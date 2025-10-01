# src/esm_feats.py
import os, hashlib, json, re
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import numpy as np
import pandas as pd
import esm  # from fair-esm

AA_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# ---------- paths ----------
CACHE_DIR = Path("data/cache")
EMB_DIR   = CACHE_DIR / "embeddings"
ZSHOT_DIR = CACHE_DIR / "zeroshot"
EMB_DIR.mkdir(parents=True, exist_ok=True)
ZSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- model loader (singleton) ----------
_model_ctx = {}

def load_esm1v(model_tag: str = "esm1v_t33_650M_UR90S_1", device: str = None):
    """
    Load ESM-1v model once. Returns (model, alphabet, batch_converter, device).
    """
    if "esm1v" in _model_ctx:
        return _model_ctx["esm1v"]

    model, alphabet = getattr(esm.pretrained, model_tag)()
    model.eval()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    _model_ctx["esm1v"] = (model, alphabet, batch_converter, device)
    return _model_ctx["esm1v"]

# ---------- utilities ----------
def _seq_key(seq: str) -> str:
    """Stable filename-safe key for a sequence."""
    h = hashlib.sha1(seq.encode()).hexdigest()
    return f"{h}_{len(seq)}"

def parse_mutant_str(mutant: str) -> List[Tuple[str, int, str]]:
    """
    'A1P:D2N' -> [('A',1,'P'), ('D',2,'N')]
    Positions are 1-based.
    """
    toks = []
    for tok in mutant.split(":"):
        m = AA_RE.match(tok)
        if not m:
            raise ValueError(f"Bad token: {tok}")
        a, pos, b = m.group(1), int(m.group(2)), m.group(3)
        toks.append((a, pos, b))
    return toks

def seq_to_mutant(wt_seq: str, mut_seq: str) -> str:
    """
    Diff mutant sequence against WT -> ProteinGym-style tokens.
    If identical, return a no-op token (A1A) so downstream code still works.
    """
    assert len(wt_seq) == len(mut_seq), "WT and mutated seq must be same length"
    toks = []
    for i, (a, b) in enumerate(zip(wt_seq, mut_seq), start=1):
        if a != b:
            toks.append(f"{a}{i}{b}")
    if not toks:
        a0 = wt_seq[0]
        toks = [f"{a0}1{a0}"]
    return ":".join(toks)

# ---------- embeddings ----------
@torch.no_grad()
def get_embedding(seq: str, model_tag: str = "esm1v_t33_650M_UR90S_1") -> np.ndarray:
    """
    Mean-pooled last-layer token representations (exclude BOS/EOS).
    Cached to disk as .npy in data/cache/embeddings/.
    Returns np.ndarray shape [d].
    """
    key = _seq_key(seq)
    npy_path = EMB_DIR / f"{key}.npy"
    if npy_path.exists():
        return np.load(npy_path)

    model, alphabet, batch_converter, device = load_esm1v(model_tag)
    batch = [("seq", seq)]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[33], return_contacts=False)
    # per-token reps: [B, L, C], we want the last layer 33
    reps = out["representations"][33][0]  # [L, C]
    # Exclude BOS (index 0) and EOS (last index)
    reps = reps[1:-1]
    emb = reps.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    np.save(npy_path, emb)
    return emb

# ---------- zero-shot mutation score ----------
@torch.no_grad()
def zero_shot_score(wt_seq: str, mutant: str, model_tag: str = "esm1v_t33_650M_UR90S_1") -> float:
    """
    ESM-1v masked-marginal log-odds:
      sum_i [ log p(mutAA_i | WT context) - log p(WTAA_i | WT context) ].
    Cached per (WT, mutant) to data/cache/zeroshot/.
    """
    key = _seq_key(wt_seq + "|" + mutant)
    json_path = ZSHOT_DIR / f"{key}.json"
    if json_path.exists():
        return float(json.loads(json_path.read_text())["score"])

    muts = parse_mutant_str(mutant)
    model, alphabet, batch_converter, device = load_esm1v(model_tag)

    # Tokenize WT once
    batch = [("wt", wt_seq)]
    _, _, wt_tokens = batch_converter(batch)
    wt_tokens = wt_tokens.to(device)

    mask_idx = alphabet.mask_idx
    # For letter -> token index
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in alphabet.standard_toks}

    total = 0.0
    for (old, pos1, new) in muts:
        # BOS offset: tokens have [BOS] + seq + [EOS]
        tok_pos = pos1  # because BOS at index 0 -> seq pos i at token i (1-based aligns)
        # Sanity: check WT AA matches
        wt_aa = wt_seq[pos1 - 1]
        assert wt_aa == old, f"WT mismatch at {pos1}: expected {old}, got {wt_aa}"

        # Mask that position in a copy
        masked = wt_tokens.clone()
        masked[0, tok_pos] = mask_idx  # mask the i-th residue

        out = model(masked, repr_layers=[], return_contacts=False)
        logits = out["logits"][0, tok_pos]        # [Vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        idx_mut = aa_to_idx[new]
        idx_wt  = aa_to_idx[old]
        total += float(log_probs[idx_mut] - log_probs[idx_wt])

    json_path.write_text(json.dumps({"score": total}))
    return total

# ---------- batch helpers ----------
def embed_dataframe(df: pd.DataFrame, seq_col: str = "mutated_sequence",
                    model_tag: str = "esm1v_t33_650M_UR90S_1") -> pd.DataFrame:
    """
    Add columns emb_path and optionally flatten to emb_* if you want.
    Keeps caching; safe to re-run.
    """
    embs = []
    for s in df[seq_col].tolist():
        e = get_embedding(s, model_tag=model_tag)
        embs.append(e)

    E = np.stack(embs, axis=0)  # [N, d]
    # Store a separate matrix file and just note its path in df to keep CSV light
    key = hashlib.sha1(("|".join(df[seq_col].tolist())).encode()).hexdigest()
    mat_path = EMB_DIR / f"matrix_{key}.npy"
    np.save(mat_path, E)

    df_out = df.copy()
    df_out["embedding_path"] = str(mat_path)
    return df_out

def zero_shot_dataframe(df: pd.DataFrame, wt_seq: str, mutant_col: str = "mutant",
                        model_tag: str = "esm1v_t33_650M_UR90S_1") -> pd.DataFrame:
    scores = []
    for m in df[mutant_col].tolist():
        s = zero_shot_score(wt_seq, m, model_tag=model_tag)
        scores.append(s)
    out = df.copy()
    out["esm1v_zero_shot"] = scores
    return out

# ===== Mutant-context zero-shot (batched) =====

@torch.no_grad()
def zero_shot_dataframe_mutantctx_batched(
    df: pd.DataFrame,
    wt_seq: str,
    mutant_col: str = "mutant",
    seq_col: str = "mutated_sequence",
    model_tag: str = "esm1v_t33_650M_UR90S_1",
    batch_size_seqs: int = 64,     # how many mutant sequences per outer chunk
    batch_size_masks: int = 256,   # how many masked copies per inner forward
) -> pd.DataFrame:
    """
    Standard ESM-1v zero-shot for mutation effects (mutant context, MASKED):
      For each variant:
        1) build the mutant sequence,
        2) for each mutated site i, create a copy with position i masked,
        3) run model on all masked copies (batched),
        4) sum [log p(mutAA_i) - log p(wtAA_i)] at each site.

    Returns df copy with new column: 'esm1v_zero_shot_mc'.
    """
    model, alphabet, batch_converter, device = load_esm1v(model_tag)
    mask_idx = alphabet.mask_idx
    aa_list = list(alphabet.standard_toks)
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in aa_list}

    # Ensure we have tokens list and mutant sequences
    if mutant_col in df.columns:
        toks_list = [parse_mutant_str(str(m)) for m in df[mutant_col].tolist()]
        seqs = df[seq_col].tolist() if seq_col in df.columns else [None]*len(toks_list)
    else:
        assert seq_col in df.columns, f"Need either '{mutant_col}' or '{seq_col}'"
        seqs = df[seq_col].tolist()
        toks_list = [parse_mutant_str(seq_to_mutant(wt_seq, s)) for s in seqs]

    # Rebuild mutant sequences if missing
    def build_mutant_seq(muts):
        seq = list(wt_seq)
        for (old, pos, new) in muts:
            assert seq[pos-1] == old, f"WT mismatch at {pos}"
            seq[pos-1] = new
        return "".join(seq)

    seqs = [
        s if isinstance(s, str) and len(s) == len(wt_seq) else build_mutant_seq(toks_list[i])
        for i, s in enumerate(seqs)
    ]

    N = len(seqs)
    scores = np.zeros(N, dtype=np.float32)

    # Process sequences in chunks to control memory
    for a in range(0, N, batch_size_seqs):
        b = min(a + batch_size_seqs, N)
        seq_chunk = seqs[a:b]
        toks_chunk = toks_list[a:b]

        # Pre-tokenize mutant sequences (unmasked) once
        data = [("seq", s) for s in seq_chunk]
        _, _, mutant_tok = batch_converter(data)            # [B, L+2]
        mutant_tok = mutant_tok.to(device)

        # Build a big list of masked copies across this chunk
        # We’ll store tuples to map back: (row_idx, token_tensor_index, pos, old, new)
        masked_batches = []
        meta = []   # (row_global_idx, pos, old, new) per masked sample

        for j, muts in enumerate(toks_chunk):
            if len(muts) == 0:
                # No change: define score 0 via a no-op
                continue
            # for each mutated site: copy tokens and mask just that site
            for (old, pos, new) in muts:
                t = mutant_tok[j].clone()
                # BOS at index 0, residue i at token index i
                t[pos] = mask_idx
                masked_batches.append(t.unsqueeze(0))
                meta.append((a + j, pos, old, new))

        if not meta:
            continue

        # Stack into manageable sub-batches
        masked_stack = torch.cat(masked_batches, dim=0)  # [M_total, L+2]
        M_total = masked_stack.size(0)
        for s in range(0, M_total, batch_size_masks):
            e = min(s + batch_size_masks, M_total)
            toks = masked_stack[s:e].to(device)
            out = model(toks, repr_layers=[], return_contacts=False)
            logits = out["logits"]  # [m, L+2, V]
            logp = torch.log_softmax(logits, dim=-1)

            # Accumulate contributions
            for k in range(s, e):
                row_idx, pos, old, new = meta[k]
                lp = logp[k - s, pos, :]   # masked position
                scores[row_idx] += float(lp[aa_to_idx[new]] - lp[aa_to_idx[old]])

    out = df.copy()
    out["esm1v_zero_shot_mc"] = scores
    return out



# ===== Pseudo-likelihood (PLL) scoring and PLLΔ =====
# ===== Memory-safe PLL and PLLΔ (with AMP + auto backoff) =====
import torch, numpy as np, pandas as pd
from math import ceil

@torch.no_grad()
def pll_score_sequences(
    seqs: list,
    model_tag: str = "esm1v_t33_650M_UR90S_1",
    seq_batch: int = 8,          # start conservative for 12GB
    pos_batch: int = 8,          # smaller pos_batch => lower peak mem
    use_amp: bool = True,
) -> np.ndarray:
    """
    PLL(x) = sum_j log p(x_j | sequence with position j masked).
    Computes in chunks over sequences (seq_batch) and positions (pos_batch).
    AMP halves activation memory on GPU; shrinking batches does NOT change accuracy.
    """
    model, alphabet, batch_converter, device = load_esm1v(model_tag)
    model.eval()
    mask_idx = alphabet.mask_idx
    scores = np.zeros(len(seqs), dtype=np.float32)

    # ---- use CUDA checks that don't assume device is a torch.device ----
    on_cuda = torch.cuda.is_available()

    # small speed boost on Ampere+ if CUDA
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    for i in range(0, len(seqs), seq_batch):
        chunk = seqs[i:i+seq_batch]
        _, _, tok = batch_converter([("s", s) for s in chunk])   # [B, L+2]
        tok = tok.to(device)   # 'device' can be "cuda" or a torch.device
        B, L2 = tok.shape
        L = L2 - 2

        for pstart in range(1, L+1, pos_batch):
            pend = min(pstart + pos_batch - 1, L)
            m = pend - pstart + 1

            rep = tok.unsqueeze(1).repeat(1, m, 1).reshape(B*m, L2)
            for j, p in enumerate(range(pstart, pend+1)):
                rep[j::m, p] = mask_idx

            # mixed precision only if CUDA is available
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(use_amp and on_cuda)):
                out = model(rep, repr_layers=[], return_contacts=False)
                logits = out["logits"]  # [B*m, L+2, V]

            # reshape just like before
            logits = logits.view(B, m, L2, -1)
            true_tok = tok[:, 1:-1]  # [B, L]
            for j, p in enumerate(range(pstart, pend+1)):
                lp = logits[:, j, p, :]             # [B, V]
                maxv = lp.max(dim=1, keepdim=True).values
                lse  = (lp - maxv).exp().sum(dim=1).log() + maxv.squeeze(1)
                add  = lp.gather(1, true_tok[:, p-1:p]).squeeze(1) - lse
                scores[i:i+B] += add.detach().cpu().numpy()

            del rep, logits
            if on_cuda:
                torch.cuda.empty_cache()

    return scores.astype(np.float32)

@torch.no_grad()
def pll_delta_dataframe(
    df: pd.DataFrame,
    wt_seq: str,
    seq_col: str = "mutated_sequence",
    model_tag: str = "esm1v_t33_650M_UR90S_1",
    seq_batch: int = 8,
    pos_batch: int = 8,
    use_amp: bool = True,
) -> pd.DataFrame:
    seqs = df[seq_col].astype(str).tolist()
    pll_mut = pll_score_sequences(seqs, model_tag=model_tag, seq_batch=seq_batch, pos_batch=pos_batch, use_amp=use_amp)
    pll_wt  = pll_score_sequences([wt_seq], model_tag=model_tag, seq_batch=1,   pos_batch=pos_batch, use_amp=use_amp)[0]
    out = df.copy()
    out["pll_delta"] = pll_mut - pll_wt
    return out

def pll_delta_dataframe_safe(
    df: pd.DataFrame,
    wt_seq: str,
    seq_col: str = "mutated_sequence",
    model_tag: str = "esm1v_t33_650M_UR90S_1",
    try_grid = ((8,8),(8,4),(6,4),(4,4),(4,2),(2,2),(1,2),(1,1)),
    use_amp: bool = True,
) -> pd.DataFrame:
    """
    Tries a grid of (seq_batch, pos_batch) until it fits in GPU memory.
    Same outputs as pll_delta_dataframe (no accuracy loss).
    """
    last_err = None
    for sb, pb in try_grid:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return pll_delta_dataframe(df, wt_seq, seq_col=seq_col, model_tag=model_tag,
                                       seq_batch=sb, pos_batch=pb, use_amp=use_amp)
        except RuntimeError as e:
            msg = str(e)
            last_err = e
            if "CUDA out of memory" in msg or "CUBLAS" in msg or "CUDA error" in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"PLLΔ OOM even with tiny batches. Last error: {last_err}")
