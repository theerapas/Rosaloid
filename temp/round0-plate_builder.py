# %%
import pandas as pd
# ---- run me first in every notebook ----
import sys, os
from pathlib import Path

# Ensure we are at the project root (the folder that contains `src/`)
# If your notebook sits in the root, this is already correct.
root = Path.cwd()

# If your notebook lives somewhere else, climb up until we see 'src'
while not (root / "src").exists() and root.parent != root:
    root = root.parent

# Put project root and src/ on sys.path
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

print("Project root:", root)
print("Has src?:", (root / "src").exists())

# Force a clean import of the latest file
import importlib, src.esm_feats as esm_feats
importlib.reload(esm_feats)

# See what the module actually exports
print([n for n in dir(esm_feats) if "emb" in n or "shot" in n or "load" in n])

# %%
# round0-plate.py — concise Round-0 plate builder
# Inputs (already produced by round0.py):
#   - round0_additive_top.csv
#   - round0_plldelta_top.csv
#   - round0_mutantctx_top.csv
# Each CSV must have: mutated_sequence,num_subs,score,method
#
# Outputs (per plate tag: additive / plldelta / mutant_ctx / union):
#   - round0_<tag>_panel.csv  (picked variants + controls/replicates)
#   - round0_<tag>_plate.csv  (well map A01..H12)
#
# Diversity uses FPS in ESM embedding space. If embedding fails, it falls back to score-only.

from pathlib import Path
import os, math, random
import numpy as np
import pandas as pd

# ------------------------ config ------------------------
TOP_FILES = [
    ("round0_additive_top.csv",  "additive"),
    ("round0_plldelta_top.csv",  "plldelta"),
    ("round0_mutantctx_top.csv", "mutant_ctx"),
]
MAKE_UNION_PLATE = True
GLOBAL_UNIQUE    = False          # if True, prevent the same seq from appearing on >1 plate

FINAL_N        = 96
DIVERSE_N      = 84               # from top list via FPS (or score fallback)
WT_REPS        = 4
BLANKS         = 4
TECH_REP_PAIRS = 2                # number of variants to replicate twice (2 pairs -> 4 wells)

STRATIFY_BY_SUBS = True           # balance 1-mut and 2-mut in the DIVERSE_N picks
RNG_SEED         = 7

# Embedding backend (expects your existing helpers; keep names consistent with your project)
DEVICE = os.environ.get("ESM_DEVICE", "cuda")  # "cuda" or "cpu"
try:
    from src.esm_feats import load_esm1v, embed_dataframe
    ESM_OK = True
except Exception:
    ESM_OK = False

# -------------------- small utilities -------------------
def ensure_schema(df: pd.DataFrame, label: str) -> pd.DataFrame:
    need = ["mutated_sequence","num_subs","score","method"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[{label}] missing columns: {miss}")
    out = df[need].copy()
    out["mutated_sequence"] = out["mutated_sequence"].astype(str)
    out["num_subs"] = pd.to_numeric(out["num_subs"], errors="coerce")
    out["score"]    = pd.to_numeric(out["score"], errors="coerce")
    out["method"]   = out["method"].astype(str)
    # uniq on sequence, keep highest score
    out = (out.sort_values("score", ascending=False)
               .drop_duplicates(subset=["mutated_sequence"], keep="first")
               .reset_index(drop=True))
    return out

def wells_96():
    rows = "ABCDEFGH"
    cols = [f"{c:02d}" for c in range(1,13)]
    return [f"{r}{c}" for r in rows for c in cols]

def _load_vec(path: str) -> np.ndarray:
    v = np.load(path)
    if v.ndim == 1: 
        return v
    if v.ndim == 2:  # token x dim -> mean-pool
        return v.mean(axis=0)
    return v.reshape(-1)

def _normalize_rows(X: np.ndarray):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = (norms > 0).ravel()
    Xn = np.zeros_like(X, dtype=float)
    # safe divide only where valid
    Xn[valid] = X[valid] / norms[valid]
    return Xn, valid

def _fps_cosine(X: np.ndarray, k: int, seed: int = 7) -> np.ndarray:
    """Farthest Point Sampling on cosine distance with uniqueness and NaN safety."""
    n = X.shape[0]
    k = min(k, n)
    if k <= 0: 
        return np.array([], dtype=int)

    Xn, valid = _normalize_rows(X)
    if valid.sum() < 2:
        # too few valid vectors -> fall back to first k unique indices
        return np.arange(k, dtype=int)

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n))
    sel = np.zeros(n, dtype=bool)
    order = []

    # initialize min-distances to +inf; selected points set to -inf to avoid reselection
    min_dist = np.full(n, np.inf, dtype=float)

    def update_with(j):
        # 1 - cosine sim = 1 - dot(x_i, x_j)
        d = 1.0 - (Xn @ Xn[j])
        # sanitize
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
        np.minimum(min_dist, d, out=min_dist)
        min_dist[j] = -np.inf  # never pick j again

    # seed
    sel[start] = True
    order.append(start)
    update_with(start)

    for _ in range(1, k):
        j = int(np.argmax(min_dist))
        if not np.isfinite(min_dist[j]) or min_dist[j] == -np.inf:
            # degenerate: fill with any remaining unused index
            rem = np.flatnonzero(~sel)
            if len(rem) == 0:
                break
            j = int(rem[0])
        sel[j] = True
        order.append(j)
        update_with(j)

    return np.array(order[:k], dtype=int)

def pick_diverse(df: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    """Pick up to n diverse sequences via FPS; fall back to score-only if embeddings misbehave."""
    df = (df.sort_values("score", ascending=False)
            .drop_duplicates(subset=["mutated_sequence"], keep="first")
            .reset_index(drop=True))
    if len(df) <= n:
        return df.copy()

    # --- embedding path (aligned to df) ---
    if ESM_OK:
        try:
            load_esm1v(device=DEVICE)
            uniq = pd.DataFrame({"mutated_sequence": pd.unique(df["mutated_sequence"].astype(str))})
            emb_uniq = embed_dataframe(uniq.copy())  # must have mutated_sequence, embedding_path
            if "embedding_path" not in emb_uniq.columns:
                raise RuntimeError("embed_dataframe() did not return 'embedding_path'")
            seq2vec = {s: _load_vec(p) for s, p in zip(emb_uniq["mutated_sequence"].astype(str),
                                                        emb_uniq["embedding_path"])}
            seqs = df["mutated_sequence"].astype(str).tolist()
            X = np.vstack([seq2vec[s] for s in seqs])

            # stratified FPS if requested
            def take(mask_bool: np.ndarray, k: int, s: int):
                if k <= 0 or mask_bool.sum() == 0: 
                    return np.array([], dtype=int)
                local_idx = np.flatnonzero(mask_bool)
                pick_local = _fps_cosine(X[local_idx], k, s)
                return local_idx[pick_local]

            chosen = np.array([], dtype=int)
            if STRATIFY_BY_SUBS and "num_subs" in df.columns:
                m1 = (df["num_subs"] == 1).values
                m2 = (df["num_subs"] == 2).values
                n1 = min(n // 2, int(m1.sum()))
                n2 = n - n1
                idx1 = take(m1, n1, seed)
                idx2 = take(m2, n2, seed + 1)
                chosen = np.concatenate([idx1, idx2])

                short = n - len(chosen)
                if short > 0:
                    rem = np.ones(len(df), dtype=bool); rem[chosen] = False
                    idxr = take(rem, short, seed + 2)
                    chosen = np.concatenate([chosen, idxr])
            else:
                allmask = np.ones(len(df), dtype=bool)
                chosen = take(allmask, n, seed)

            # enforce uniqueness & shuffle lightly
            chosen = np.unique(chosen)
            out = df.iloc[chosen].sample(frac=1.0, random_state=seed).reset_index(drop=True)
            return out

        except Exception as e:
            print(f"[pick_diverse] embedding path failed ({e}); falling back to score-only.")

    # --- fallback: score-only selection (still stratified) ---
    if STRATIFY_BY_SUBS and "num_subs" in df.columns:
        one = df[df["num_subs"] == 1].nlargest(n // 2, "score")
        two = df[df["num_subs"] == 2].nlargest(n - len(one), "score")
        out = pd.concat([one, two], ignore_index=True)
    else:
        out = df.nlargest(n, "score")
    out = (out.sort_values("score", ascending=False)
              .drop_duplicates(subset=["mutated_sequence"], keep="first")
              .head(n)
              .reset_index(drop=True))
    return out


def add_controls_and_reps(core: pd.DataFrame, seed: int = RNG_SEED) -> pd.DataFrame:
    """Add WT/BLANK wells and tech replicate pairs to reach FINAL_N."""
    rng = random.Random(seed)
    out = core.copy()
    # tech-rep pairs (duplicate rows with a tag)
    cand = out.copy()
    if len(cand) and TECH_REP_PAIRS > 0:
        take_idx = rng.sample(range(len(cand)), k=min(TECH_REP_PAIRS, len(cand)))
        reps = []
        for i in take_idx:
            row = cand.iloc[i].copy()
            row["selected_reason"] = "tech_rep"
            row["replicate_of"] = row["mutated_sequence"]
            reps.append(row)
            reps.append(row.copy())  # two extra wells for the pair
        if reps:
            out = pd.concat([out, pd.DataFrame(reps)], ignore_index=True)
    # WT & blanks
    def mk(tag):
        return pd.DataFrame([{
            "mutated_sequence": tag,
            "num_subs": 0,
            "score": np.nan,
            "method": "control",
            "selected_reason": "control",
            "replicate_of": "",
            "is_WT": 1 if tag=="WT" else 0,
            "is_blank": 1 if tag=="BLANK" else 0,
        }])
    if WT_REPS > 0:
        out = pd.concat([out, pd.concat([mk("WT")]*WT_REPS, ignore_index=True)], ignore_index=True)
    if BLANKS  > 0:
        out = pd.concat([out, pd.concat([mk("BLANK")]*BLANKS, ignore_index=True)], ignore_index=True)
    # pad/truncate to FINAL_N
    if len(out) < FINAL_N:
        pad = FINAL_N - len(out)
        filler = core.head(min(pad, len(core))).copy()
        filler["selected_reason"] = filler.get("selected_reason", "diverse")
        out = pd.concat([out, filler], ignore_index=True)
    out = out.head(FINAL_N).reset_index(drop=True)
    # annotate flags if missing
    if "is_WT" not in out.columns:    out["is_WT"] = (out["mutated_sequence"]=="WT").astype(int)
    if "is_blank" not in out.columns: out["is_blank"] = (out["mutated_sequence"]=="BLANK").astype(int)
    return out

def write_panel_and_plate(df: pd.DataFrame, tag: str):
    """Write panel and plate CSVs, assign wells with light randomization."""
    panel = df.copy()
    # Ensure selected_reason
    if "selected_reason" not in panel.columns:
        panel["selected_reason"] = "diverse"
    if "replicate_of" not in panel.columns:
        panel["replicate_of"] = ""
    panel.to_csv(f"round0_{tag}_panel.csv", index=False)

    # Place wells: shuffle a bit to reduce edge bias; scatter controls
    rng = random.Random(RNG_SEED)
    wells = wells_96()
    # Split controls and designs
    controls = panel[(panel["is_WT"]==1) | (panel["is_blank"]==1)]
    designs  = panel[(panel["is_WT"]==0) & (panel["is_blank"]==0)]
    designs = designs.sample(frac=1.0, random_state=RNG_SEED)  # shuffle designs

    # Put a couple of controls in edges/corners, rest interleaved
    layout = []
    # Simple strategy: interleave 8 blocks of 12, put one control at start of each odd block if available
    d_iter = iter(designs.to_dict("records"))
    c_iter = iter(controls.to_dict("records"))
    for i in range(96):
        use_ctrl = (i % 12 == 0) and (len(layout) < len(controls))  # coarse scatter
        if use_ctrl:
            try:
                layout.append(next(c_iter))
                continue
            except StopIteration:
                pass
        try:
            layout.append(next(d_iter))
        except StopIteration:
            # fill with any remaining control
            try:
                layout.append(next(c_iter))
            except StopIteration:
                # fallback: duplicate first row (shouldn't happen)
                layout.append(panel.iloc[0].to_dict())

    plate = pd.DataFrame(layout).copy()
    plate.insert(0, "well", wells)
    plate.to_csv(f"round0_{tag}_plate.csv", index=False)
    print(f"[{tag}] wrote round0_{tag}_panel.csv and round0_{tag}_plate.csv")

# -------------------- main plate builder -----------------
def build_plate(top_csv: str, tag: str, forbidden: set[str] | None = None):
    forbidden = forbidden or set()
    src = Path(top_csv)
    if not src.exists():
        print(f"[{tag}] missing: {src.name} — skipped.")
        return set()
    df = ensure_schema(pd.read_csv(src), tag)
    # remove forbidden (already used on other plates when GLOBAL_UNIQUE)
    df = df[~df["mutated_sequence"].isin(forbidden)].reset_index(drop=True)
    if len(df) == 0:
        print(f"[{tag}] no candidates after filtering — skipped.")
        return set()

    # 1) choose diverse core set from the ranked list
    core = pick_diverse(df, DIVERSE_N, seed=RNG_SEED)
    core["selected_reason"] = "diverse"
    core["replicate_of"] = ""
    core["is_WT"] = 0
    core["is_blank"] = 0

    # 2) add controls + tech reps; pad/truncate to 96
    panel = add_controls_and_reps(core, seed=RNG_SEED)

    # 3) write panel+plate
    write_panel_and_plate(panel, tag)

    return set(panel.loc[(panel["is_WT"]==0)&(panel["is_blank"]==0), "mutated_sequence"].tolist())

# --------------------------- run -------------------------
if __name__ == "__main__":
    random.seed(RNG_SEED)
    used = set()

    # Per-method plates
    for top_path, label in TOP_FILES:
        taken = build_plate(top_path, label, forbidden=used if GLOBAL_UNIQUE else set())
        if GLOBAL_UNIQUE:
            used |= taken

    # Optional union/ensemble plate from the union of method tops (deduped by seq)
    if MAKE_UNION_PLATE:
        frames = []
        for p, lbl in TOP_FILES:
            p = Path(p)
            if p.exists():
                frames.append(ensure_schema(pd.read_csv(p), lbl))
        if frames:
            uni = pd.concat(frames, ignore_index=True)
            uni = (uni.sort_values("score", ascending=False)
                      .drop_duplicates(subset=["mutated_sequence"], keep="first")
                      .reset_index(drop=True))
            tmp = "round0_union_top.csv"
            uni.to_csv(tmp, index=False)
            forbid = used if GLOBAL_UNIQUE else set()
            build_plate(tmp, "union", forbidden=forbid)
        else:
            print("[union] no sources; skipped.")



