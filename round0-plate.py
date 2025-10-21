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
# === Round-0 plates for standardized schema ===
# expected columns in each top file: mutated_sequence, num_subs, score, method
from pathlib import Path
import re, numpy as np, pandas as pd
from src.esm_feats import load_esm1v, embed_dataframe

# ------------- config -------------
TOP_FILES = [
    ("round0_additive_top.csv",  "additive"),
    ("round0_plldelta_top.csv",  "plldelta"),
    ("round0_mutantctx_top.csv", "mutant_ctx"),
]

MAKE_UNION_PLATE = True       # also build a separate ensemble plate
GLOBAL_UNIQUE    = False      # if True, prevent the same seq from appearing on >1 plate

FINAL_N          = 96
DIVERSE_N        = 84         # picked via FPS from the method's top list
WT_REPS          = 4          # WT wells
BLANKS           = 4          # blank wells
TECH_REP_PAIRS   = 2          # choose this many mutants and replicate each twice
STRATIFY_BY_SUBS = True       # balance 1-mut vs 2-mut in FPS
SEED             = 123

DEVICE           = "cuda"     # or "cpu"
WT_SEQ           = None       # set to your WT sequence string to add WT controls; None = skip

# ------------- helpers -------------
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

def make_plate_wells(n):
    rows = "ABCDEFGH"; cols = list(range(1, 13))
    return [f"{r}{c}" for r in rows for c in cols][:n]  # row-major

def farthest_point_sampling(X, k, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) == 0: return np.array([], dtype=int)
    idx = [int(rng.integers(0, len(X)))]
    dist = np.full(len(X), np.inf, dtype=np.float32)
    for _ in range(1, min(k, len(X))):
        d = np.linalg.norm(X - X[idx[-1]], axis=1)
        dist = np.minimum(dist, d)
        idx.append(int(np.argmax(dist)))
    return np.array(idx, dtype=int)

def ensure_schema(df, label):
    need = {"mutated_sequence","num_subs","score","method"}
    if not need.issubset(df.columns):
        raise ValueError(f"{label}: expected {need}, got {set(df.columns)}")
    df = df.copy()
    df["mutated_sequence"] = df["mutated_sequence"].astype(str)
    df["num_subs"] = pd.to_numeric(df["num_subs"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["method"] = df["method"].astype(str)
    # sanity: look like AA sequences?
    if not df["mutated_sequence"].map(lambda s: bool(AA_RE.fullmatch(s))).all():
        bad = df.loc[~df["mutated_sequence"].map(lambda s: bool(AA_RE.fullmatch(s))), "mutated_sequence"].head(3).tolist()
        raise ValueError(f"{label}: mutated_sequence contains non-AA rows (e.g., {bad}).")
    return df

def embed_and_pick_diverse(df, n, seed=SEED):
    if len(df) <= n:
        return df.copy()
    _ = load_esm1v(device=DEVICE)
    df2 = embed_dataframe(df.copy())                 # adds 'embedding_path'; one .npy for this batch
    if STRATIFY_BY_SUBS:
        target1 = n // 2
        target2 = n - target1
        one = df2[df2["num_subs"] == 1].reset_index(drop=True)
        two = df2[df2["num_subs"] == 2].reset_index(drop=True)

        # pick from 1-muts
        take1 = min(target1, len(one))
        if take1:
            E1 = np.load(one["embedding_path"].iloc[0], mmap_mode="r")[:len(one)]
            idx1 = farthest_point_sampling(E1, take1, seed=seed)
            pick1 = one.iloc[idx1]
        else:
            pick1 = one.iloc[:0]

        # pick from 2-muts
        take2 = min(target2, len(two))
        if take2:
            E2 = np.load(two["embedding_path"].iloc[0], mmap_mode="r")[:len(two)]
            idx2 = farthest_point_sampling(E2, take2, seed=seed)
            pick2 = two.iloc[idx2]
        else:
            pick2 = two.iloc[:0]

        chosen = pd.concat([pick1, pick2], ignore_index=True)

        # fill by score if short
        if len(chosen) < n:
            remaining = df2.loc[~df2.index.isin(chosen.index)]
            filler = remaining.sort_values("score", ascending=False).head(n - len(chosen))
            chosen = pd.concat([chosen, filler], ignore_index=True)

        return chosen.reset_index(drop=True)

    # non-stratified
    E = np.load(df2["embedding_path"].iloc[0], mmap_mode="r")
    idx = farthest_point_sampling(E, n, seed=seed)
    return df2.iloc[idx].reset_index(drop=True)

def build_plate(top_csv, label, forbidden=set()):
    df = ensure_schema(pd.read_csv(top_csv), label)
    if forbidden:
        df = df[~df["mutated_sequence"].isin(forbidden)].reset_index(drop=True)

    # unique by sequence (keep best score), then diverse pick
    df = (df.sort_values("score", ascending=False)
            .drop_duplicates(subset=["mutated_sequence"], keep="first")
            .reset_index(drop=True))

    diverse = embed_and_pick_diverse(df, min(DIVERSE_N, len(df)))
    diverse = diverse.sort_values("score", ascending=False).reset_index(drop=True)

    # controls & replicates
    rows = []
    if WT_SEQ:
        for r in range(WT_REPS):
            rows.append(dict(mutated_sequence=WT_SEQ, num_subs=0, score=np.nan,
                             method="WT_control", tag="WT"))
    for b in range(BLANKS):
        rows.append(dict(mutated_sequence="", num_subs=np.nan, score=np.nan,
                         method="blank", tag="blank"))
    if len(diverse) >= TECH_REP_PAIRS:
        reps = diverse.sample(TECH_REP_PAIRS, random_state=SEED)
        for _, rec in reps.iterrows():
            for _ in range(2):
                rows.append(dict(mutated_sequence=rec.mutated_sequence,
                                 num_subs=rec.num_subs, score=rec.score,
                                 method=f"{label}_rep", tag="tech_rep"))

    controls = pd.DataFrame(rows)
    need = max(0, FINAL_N - len(controls))
    main = diverse.head(need).copy()
    main["tag"] = "main"

    plate = pd.concat([controls, main], ignore_index=True).reset_index(drop=True)
    plate["well"] = make_plate_wells(len(plate))
    plate = plate[["well","mutated_sequence","num_subs","score","method","tag"]]

    # decision log (the candidate set before controls)
    panel = diverse.assign(tag="candidate")[["mutated_sequence","num_subs","score","method","tag"]]

    plate_out = f"round0_{label}_plate.csv"
    panel_out = f"round0_{label}_panel.csv"
    plate.to_csv(plate_out, index=False)
    panel.to_csv(panel_out, index=False)
    print(f"[{label}] wrote {plate_out} and {panel_out} | main={len(main)} + controls={len(controls)}")
    return plate, set(main["mutated_sequence"])

# ------------- build per-method -------------
used = set()
for path,label in TOP_FILES:
    p = Path(path)
    if not p.exists():
        print(f"[{label}] skip: {path} not found")
        continue
    forbid = used if GLOBAL_UNIQUE else set()
    _, chosen = build_plate(path, label, forbidden=forbid)
    if GLOBAL_UNIQUE:
        used |= chosen

# ------------- optional: separate union plate -------------
if MAKE_UNION_PLATE:
    frames = []
    for p,l in TOP_FILES:
        p = Path(p)
        if p.exists():
            frames.append(ensure_schema(pd.read_csv(p), l))
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



