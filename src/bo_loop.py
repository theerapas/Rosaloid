import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict

from .utils import LabeledSeq, Proposal, load_problem, SEQ_COL, RANDOM_SEED, DMS_ID_TO_FILE
from .embeddings import Embedder
from .surrogate import NGBSurrogate
from .acquisition import expected_improvement
from .diversity import pick_diverse
from .metrics import y_of, topk_true_in_selected
from scipy.stats import spearmanr

def get_or_compute_embeddings(embedder: Embedder, seqs: List[str], cache_prefix: str, output_dir: str):
    """
    Load embeddings from cache if available, else compute and save.
    Returns: dict mapping seq -> embedding
    """
    cache_emb_path = os.path.join(output_dir, f"{cache_prefix}_embeddings.npy")
    cache_seqs_path = os.path.join(output_dir, f"{cache_prefix}_embeddings_seqs.csv")
    
    emb_map = None
    
    if os.path.exists(cache_emb_path) and os.path.exists(cache_seqs_path):
        saved_seqs = pd.read_csv(cache_seqs_path)["seq"].tolist()
        if saved_seqs == seqs:
            print(f"[Cache] Loading embeddings from {cache_emb_path}")
            E_ALL = np.load(cache_emb_path, mmap_mode=None)
            emb_map = {s: e for s, e in zip(seqs, E_ALL)}
        else:
            print("[Cache] Sequence mismatch. Recomputing...")
            
    if emb_map is None:
        print("[Cache] Computing embeddings...")
        E_ALL = embedder.encode(seqs)
        emb_map = {s: e for s, e in zip(seqs, E_ALL)}
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        np.save(cache_emb_path, E_ALL)
        pd.DataFrame({"seq": seqs}).to_csv(cache_seqs_path, index=False)
        print(f"[Cache] Saved to {cache_emb_path}")
        
    return emb_map

def emb_lookup(seq_list: List[str], emb_map: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([emb_map[s] for s in seq_list]).astype(np.float32, copy=False)

def propose_from_pool(
    surrogate: NGBSurrogate,
    labeled: List[LabeledSeq],
    candidate_pool: List[str],
    emb_map: Dict[str, np.ndarray],
    batch_size: int = 24,
    xi: float = 0.01,
    min_hd_between: int = 2
) -> List[Proposal]:
    
    # 1. Train surrogate
    X_train = emb_lookup([x.seq for x in labeled], emb_map)
    y_train = np.array([x.fitness for x in labeled], dtype=float)
    surrogate.fit(X_train, y_train)
    
    # 2. Score pool (exclude potential duplicates if any remain)
    tested = set(x.seq for x in labeled)
    cand = [s for s in candidate_pool if s not in tested]
    if len(cand) == 0: return []
    
    X_cand = emb_lookup(cand, emb_map)
    mu, sigma = surrogate.predict(X_cand)
    best_y = float(np.max(y_train))
    ei = expected_improvement(mu, sigma, best_y, xi=xi)
    
    # 3. Create proposals
    proposals = [Proposal(seq=c, mu=float(m), sigma=float(s), ei=float(e))
                 for c, m, s, e in zip(cand, mu, sigma, ei)]
    
    # 4. Diversity selection
    return pick_diverse(proposals, k=batch_size, min_hd=min_hd_between)

def run_bo_loop(
    dms_id: str,
    output_dir: str,
    rounds: int = 5,
    batch_size: int = 24,
    low_n: int = 96,
    xi: float = 0.05,
    min_hd: int = 1,
    model_family: str = "esm2",
    device: str = "cuda"
):
    print(f"--- Starting BO for {dms_id} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    wt, dms = load_problem(dms_id, data_dir="data") # Assuming data dir is mapped correctly or handled in utils
    print(f"Ref WT len: {len(wt)}, DMS size: {len(dms)}")
    
    # 2. Initial Split
    # Deterministic split based on seed
    dms = dms.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    seed96 = dms.iloc[:low_n].copy()
    pool_df = dms.iloc[low_n:].copy()
    
    labeled_data = [LabeledSeq(seq=row[SEQ_COL], fitness=float(row["fitness"])) 
                    for _, row in seed96.iterrows()]
    candidate_pool = pool_df[SEQ_COL].tolist()
    truth_map = dict(zip(dms[SEQ_COL], dms["fitness"]))
    
    # Save initial
    seed96.to_csv(os.path.join(output_dir, f"{dms_id}_initial96.csv"), index=False)
    
    # 3. Setup Models & Embeddings
    embedder = Embedder(family=model_family, device=device)
    surrogate = NGBSurrogate(pca_dim=16)
    
    all_seqs = dms[SEQ_COL].tolist()
    emb_map = get_or_compute_embeddings(embedder, all_seqs, dms_id, output_dir)
    
    # 4. BO Loop
    results = []
    
    for r in range(1, rounds + 1):
        print(f"\n[Round {r}] Generating proposals...")
        
        # Propose
        proposals = propose_from_pool(
            surrogate, labeled_data, candidate_pool, emb_map, 
            batch_size=batch_size, xi=xi, min_hd_between=min_hd
        )
        
        # Evaluate (simulated)
        # In a real loop, we would ask user to synthesize. Here we look up truth.
        new_labeled = []
        for p in proposals:
            true_y = truth_map.get(p.seq, np.nan)
            new_labeled.append(LabeledSeq(seq=p.seq, fitness=true_y))
        
        # Metrics
        current_pool_list = list(set(candidate_pool) - set([p.seq for p in proposals])) # remaining pool
        # Note: Hit@K is usually defined on the *whole* pool or the *initial* pool.
        # The notebook calculated Hit@24 and Hit@96 on the proposals selected in this batch.
        # Let's verify expectations: "topk_true_in_selected"
        # We need the full pool to define what is "true top k". 
        # The notebook used the *remaining* candidate pool (?) or the full original pool?
        # Notebook: `Round-0 labeled: 96 | Candidate pool: 13765`
        # It seems the "pool" passed to metrics is the set of ALL candidates available for selection?
        # Actually strictly speaking, Hit@K checks if we found the global maxima.
        # We'll use the current `candidate_pool` + `labeled_data` (effectively all_dms_seqs minus initial seed?) 
        # as the reference for "True Top". Or just all_dms_seqs.
        # Let's use `pool_df` (the initial pool) as the reference for "Top K" to be consistent.
        
        selected_seqs = [p.seq for p in proposals]
        
        # Calculate measurements
        y_true_selected = np.array([x.fitness for x in new_labeled])
        y_pred_selected = np.array([p.mu for p in proposals])
        
        # Update state
        labeled_data.extend(new_labeled)
        candidate_pool = [c for c in candidate_pool if c not in selected_seqs] # remove selected
        
        # Metrics Calculation
        # Best so far
        best_so_far = max(x.fitness for x in labeled_data)
        
        # Spearman on this batch
        if len(y_true_selected) > 1:
            rho, _ = spearmanr(y_pred_selected, y_true_selected)
        else:
            rho = 0.0
            
        # Diversity
        # Mean pairwise hamming in the *selected batch*
        from .diversity import mean_pairwise_hamming
        div = mean_pairwise_hamming(selected_seqs)
        
        # Hit@K (checking if we found top K from the *initial* candidate pool)
        # We check if any of the *selected* sequences are in the top K of the *initial pool*
        initial_pool_seqs = pool_df[SEQ_COL].tolist()
        hit24 = topk_true_in_selected(selected_seqs, initial_pool_seqs, truth_map, k=24)
        hit96 = topk_true_in_selected(selected_seqs, initial_pool_seqs, truth_map, k=96)
        
        row = {
            "round": r,
            "best_so_far": best_so_far,
            "avg_selected": np.mean(y_true_selected),
            "diversity_mean_hamming": div,
            "Hit@24": hit24,
            "Hit@96": hit96,
            "Spearman": rho,
            "n_measured_total": len(labeled_data)
        }
        results.append(row)
        print(f"Stats: {row}")
        
        # Save proposals
        props_df = pd.DataFrame([{
            "seq": p.seq, "mu": p.mu, "sigma": p.sigma, "ei": p.ei, "true_fitness": truth_map.get(p.seq)
        } for p in proposals])
        props_df.to_csv(os.path.join(output_dir, f"{dms_id}_bo_proposals_round{r}.csv"), index=False)
        
    # Save validation stats
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, f"{dms_id}_bo_metrics.csv"), index=False)
    
    return results_df
