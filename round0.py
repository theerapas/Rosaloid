# %%
import pandas as pd, numpy as np
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

from src.esm_feats import embed_dataframe, load_esm1v


# %%
# ---- knobs ----
BATCH = 96          # use 24 if you want tiny
SHORTLIST = 5*BATCH # diversify beyond the strict top-k
RADIUS = 2          # trust radius (num_subs <= RADIUS)

# 1) load zero-shot table and apply trust radius
zs = pd.read_csv("gfp_dms_with_zeroshot.csv").reset_index(drop=True)
pool = zs.query("num_subs <= @RADIUS").copy()

# 2) shortlist by zero-shot score (higher is better)
short = (pool.sort_values("esm1v_zero_shot", ascending=False)
              .head(SHORTLIST)
              .reset_index(drop=True))

# 3) ensure embeddings for the shortlist (uses cache; fast on 4070S)
_ = load_esm1v(device="cuda")   # falls back to cpu if needed
short = embed_dataframe(short)  # adds 'embedding_path' and writes matrix
E = np.load(short["embedding_path"].iloc[0])   # rows align with 'short' order

# 4) diversity pick via farthest-point sampling in embedding space
def farthest_point_sampling(X, k):
    rng = np.random.default_rng(0)
    idx = [int(rng.integers(0, len(X)))]
    dist = np.full(len(X), np.inf, dtype=np.float32)
    for _ in range(1, k):
        d = np.linalg.norm(X - X[idx[-1]], axis=1)
        dist = np.minimum(dist, d)
        idx.append(int(np.argmax(dist)))
    return np.array(idx)

sel = farthest_point_sampling(E, BATCH)
round0 = short.iloc[sel].copy()

# (optional) quick sanity prints
print("Round-0 mean zero-shot:", round0["esm1v_zero_shot"].mean())
print("Avg Hamming to WT:", (round0["mutated_sequence"]
                             .apply(lambda s: sum(a!=b for a,b in zip(s, round0.iloc[0]['mutated_sequence'])))
                             .mean()))

round0.to_csv("round0_batch.csv", index=False)
print(f"Saved round0_batch.csv with {len(round0)} sequences.")

# %%
# Round 0 panel builder (explicit schema; standardized outputs)
from pathlib import Path
import pandas as pd
import numpy as np

import os
from pathlib import Path

ROUND0 = Path.cwd()  # you are running from data/round0
os.environ["ESM_CACHE_DIR"] = str((ROUND0.parent / "data" / "cache").resolve())
os.environ["ESM_EMB_DIR"]   = str((ROUND0.parent / "data" / "cache" / "embeddings").resolve())
os.environ["ESM_ZSHOT_DIR"] = str((ROUND0.parent / "data" / "cache" / "zeroshot").resolve())

print("Embedding cache ->", os.environ["ESM_EMB_DIR"])

# ---------------------------
# Config (paths & knobs)
# ---------------------------
IN_DIR  = (Path.cwd().parent / "zeroshot_prior").resolve()
OUT_DIR = Path("").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADD_PATH = IN_DIR / "gfp_dms_with_additive_zeroshot.csv"
PLL_PATH = IN_DIR / "gfp_dms_with_plldelta.csv"
CTX_PATH = IN_DIR / "gfp_dms_with_zeroshot_mutantctx.csv"

# How many per method
N_PER_METHOD   = 96
# Filter to <= MAX_SUBS if the column exists (and/or we can compute it)
COMPATIBLE_ONLY = True
MAX_SUBS        = 2

# Output files
OUT_ADD   = OUT_DIR / "round0_additive_top.csv"
OUT_PLL   = OUT_DIR / "round0_plldelta_top.csv"
OUT_CTX   = OUT_DIR / "round0_mutantctx_top.csv"
OUT_UNION = OUT_DIR / "round0_union_panel.csv"

# ---------------------------
# Explicit schemas per input
# Tell the code EXACTLY which columns to use.
# If your file uses different names, edit here (only here).
# ---------------------------
SCHEMAS = {
    "additive": {
        "path":  ADD_PATH,
        "key":   "mutated_sequence",   # <-- input column name with the FULL sequence
        "score": "esm1v_zero_shot_add",
        "subs":  "num_subs",           # if missing, we'll try to compute or skip filtering
        "label": "additive",
    },
    "plldelta": {
        "path":  PLL_PATH,
        "key":   "mutated_sequence",
        "score": "pll_delta",
        "subs":  "num_subs",
        "label": "plldelta",
    },
    "mutantctx": {
        "path":  CTX_PATH,
        "key":   "mutated_sequence",
        "score": "esm1v_zero_shot_mc",
        "subs":  "num_subs",
        "label": "mutant_ctx",
    },
}

# --- add these imports at the top ---
from src.esm_feats import load_esm1v, embed_dataframe  # your helpers

# --- add these knobs near your config ---
DIVERSIFY = True          # turn FPS on/off
SHORTLIST_MULT = 5        # shortlist = 5×96 before FPS
FPS_SEED = 0
DEVICE = "cuda"           # or "cpu"

# --- add FPS helper somewhere above "Load & prepare" ---
def farthest_point_sampling(X, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(0, len(X)))]
    dist = np.full(len(X), np.inf, dtype=np.float32)
    for _ in range(1, min(k, len(X))):
        d = np.linalg.norm(X - X[idx[-1]], axis=1)
        dist = np.minimum(dist, d)
        idx.append(int(np.argmax(dist)))
    return np.array(idx, dtype=int)

from pathlib import Path
import os, numpy as np

emb_dir = Path(os.environ["ESM_EMB_DIR"]).resolve()
print("EMB_DIR =", emb_dir, "| exists:", emb_dir.exists(), "| is_dir:", emb_dir.is_dir())

probe = emb_dir / "___probe.npy"
try:
    with open(os.fspath(probe), "wb") as f:
        np.save(f, np.array([1,2,3], dtype=np.float32))
    probe.unlink(missing_ok=True)
    print("Probe write: OK")
except Exception as e:
    print("Probe write: FAILED ->", e)

def diversify_top(df, n, label):
    """Assumes mutated_sequence is the FULL AA sequence. If it's a mut-code, map to sequence first."""
    short = (df.sort_values("score", ascending=False)
               .drop_duplicates(subset=["mutated_sequence"], keep="first")
               .head(n * SHORTLIST_MULT)
               .reset_index(drop=True))
    tmp = short.copy()
    tmp["mutated_sequence"] = tmp["mutated_sequence"]  # mirror for the embedder
    _ = load_esm1v(device=DEVICE)
    tmp = embed_dataframe(tmp)  # adds 'embedding_path', writes one .npy for this shortlist
    E = np.load(Path(tmp["embedding_path"].iloc[0]), mmap_mode="r")
    sel = farthest_point_sampling(E, n, seed=FPS_SEED)
    out = tmp.iloc[sel].copy()
    # return in your original schema
    out = out.rename(columns={"mutated_sequence": "mutated_sequence"})
    out = out[["mutated_sequence", "num_subs", "score"]]
    out["method"] = label
    return out.reset_index(drop=True)


# Optional: provide WT if you ever want to compute num_subs from sequences.
# Leave WT_SEQ=None to disable.
WT_SEQ = None  # e.g., "MSKGEELF..."

# ---------------------------
# Helpers (minimal, no guessing)
# ---------------------------
def _require_cols(df, cols, where):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required column(s): {missing}. "
                         f"Available: {list(df.columns)}")

def _maybe_make_num_subs(df, subs_col, key_col):
    """Ensure a numeric 'num_subs'. If subs_col missing and WT given, compute; else NaN."""
    out = df.copy()
    if subs_col in out.columns:
        out = out.rename(columns={subs_col: "num_subs"})
        out["num_subs"] = pd.to_numeric(out["num_subs"], errors="coerce")
        return out

    if WT_SEQ is not None and key_col in out.columns:
        if out[key_col].map(len).eq(len(WT_SEQ)).all():
            out["num_subs"] = out[key_col].map(lambda s: sum(a != b for a, b in zip(s, WT_SEQ)))
            return out

    # fallback: keep NaN; filtering step will skip if all-NaN
    out["num_subs"] = np.nan
    return out

def _prep_one(schema):
    path   = schema["path"]
    key    = schema["key"]
    score  = schema["score"]
    subs   = schema["subs"]
    label  = schema["label"]

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    _require_cols(df, [key, score], f"[{label}] {path.name}")

    # normalize columns
    df = _maybe_make_num_subs(df, subs, key)
    out = df[[key, "num_subs", score]].copy()
    out.columns = ["mutated_sequence", "num_subs", "score"]
    out["method"] = label

    # strong typing
    out["mutated_sequence"] = out["mutated_sequence"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")

    print(f"[{label}] key='{key}' | score='{score}' | subs='{subs}' | rows={len(out)}")
    return out

def _filter_compatible(df):
    if df["num_subs"].isna().all():
        print("WARNING: num_subs all-NaN; skipping <= filter.")
        return df
    return df.loc[df["num_subs"] <= MAX_SUBS].copy()

def _top_n_unique(df, n):
    ranked = df.sort_values("score", ascending=False)
    uniq = ranked.drop_duplicates(subset=["mutated_sequence"], keep="first")
    return uniq.head(n).reset_index(drop=True)

def _brief(df, name):
    smin = np.nanmin(df['score'].values) if len(df) else np.nan
    smax = np.nanmax(df['score'].values) if len(df) else np.nan
    known = int((~df['num_subs'].isna()).sum())
    print(f"{name}: {len(df):4d} rows | score [{smin:.3g}, {smax:.3g}] | num_subs known {known}/{len(df)}")

# ---------------------------
# Load, filter, select
# ---------------------------
add_df  = _prep_one(SCHEMAS["additive"])
pll_df  = _prep_one(SCHEMAS["plldelta"])
ctx_df  = _prep_one(SCHEMAS["mutantctx"])

if COMPATIBLE_ONLY:
    add_df = _filter_compatible(add_df)
    pll_df = _filter_compatible(pll_df)
    ctx_df = _filter_compatible(ctx_df)

# ---------------------------
# Select round 0 sets
# ---------------------------
if DIVERSIFY:
    add_top = diversify_top(add_df, N_PER_METHOD, "additive")
    pll_top = diversify_top(pll_df, N_PER_METHOD, "plldelta")
    ctx_top = diversify_top(ctx_df, N_PER_METHOD, "mutant_ctx")
else:
    add_top = _top_n_unique(add_df, N_PER_METHOD)
    pll_top = _top_n_unique(pll_df, N_PER_METHOD)
    ctx_top = _top_n_unique(ctx_df, N_PER_METHOD)

add_top.to_csv(OUT_ADD, index=False)
pll_top.to_csv(OUT_PLL, index=False)
ctx_top.to_csv(OUT_CTX, index=False)

# ---------------------------
# Union (dedup by sequence)
# ---------------------------
union_panel = pd.concat([add_top, pll_top, ctx_top], ignore_index=True)
union_panel = (union_panel
               .sort_values(["mutated_sequence", "score"], ascending=[True, False])
               .drop_duplicates(subset=["mutated_sequence"], keep="first")
               .reset_index(drop=True))
union_panel.to_csv(OUT_UNION, index=False)

# ---------------------------
# Report
# ---------------------------
print("=== Input snapshots (post-compat filtering) ===")
_ = [_brief(add_df, "additive pool"),
     _brief(pll_df, "plldelta pool"),
     _brief(ctx_df, "mutant_ctx pool")]

print("\n=== Round 0 selections (per method) ===")
_ = [_brief(add_top, "additive top"),
     _brief(pll_top, "plldelta top"),
     _brief(ctx_top, "mutant_ctx top")]

pair_overlap = {
    ("add","pll"): len(set(add_top.mutated_sequence) & set(pll_top.mutated_sequence)),
    ("add","ctx"): len(set(add_top.mutated_sequence) & set(ctx_top.mutated_sequence)),
    ("pll","ctx"): len(set(pll_top.mutated_sequence) & set(ctx_top.mutated_sequence)),
}
tri = set(add_top.mutated_sequence) & set(pll_top.mutated_sequence) & set(ctx_top.mutated_sequence)
print(f"\nOverlaps (pairwise): {pair_overlap} | 3-way: {len(tri)}")

print("\n=== Union panel ===")
_brief(union_panel, "round0 union")
print(f"\nWrote:\n- {OUT_ADD}\n- {OUT_PLL}\n- {OUT_CTX}\n- {OUT_UNION}")



