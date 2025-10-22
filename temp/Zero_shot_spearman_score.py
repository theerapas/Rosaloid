# %%
# GFP Benchmarks – one cell
# =========================
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
import sys, os

# ---- Paths (edit if your files live elsewhere) ----
OURS_WT_CSV  = "gfp_dms_with_zeroshot.csv"              # has 'esm1v_zero_shot' (WT context)
OURS_MC_CSV  = "gfp_dms_with_zeroshot_mutantctx.csv"    # has 'esm1v_zero_shot_mc' (mutant context)
FRIEND_CSV   = "gfp_with_esm1v.csv"                     # friend's output
REF_CSV      = "ref_gfp.csv"                            # has 'target_seq' (WT)

TRUE_COL     = "DMS_score"
MIN_POS_N    = 10                                       # min singles per site to include in per-position stats
TOPK_LIST    = (24, 96, 384)                            # report these k's

# -------------------------

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
# Helpers
# -------------------------
def numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

def choose_pred_column(df, prefer=()):
    """Pick an ESM-ish prediction column (never DMS labels or binaries)."""
    bad = {TRUE_COL, "DMS_score_bin", "above_WT", "label", "y", "target"}
    pref = [c for c in prefer if c in df.columns]
    for c in pref:
        if c not in bad and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 2:
            return c
    # otherwise, heuristic search
    cand = [c for c in df.columns
            if c not in bad
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique(dropna=True) > 2]
    def score_name(c):
        cl = c.lower()
        score = 0
        if "esm" in cl: score += 3
        if "zero" in cl: score += 2
        if "shot" in cl: score += 1
        if "pred" in cl or "score" in cl: score += 1
        return score
    cand.sort(key=lambda c: (score_name(c), df[c].nunique(dropna=True)), reverse=True)
    if not cand:
        raise ValueError("No usable numeric prediction column found.")
    return cand[0]

AA_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
def parse_mutant_str(mutant: str):
    toks = []
    for tok in str(mutant).split(":"):
        m = AA_RE.match(tok)
        if not m: continue
        a, pos, b = m.group(1), int(m.group(2)), m.group(3)
        toks.append((a, pos, b))
    return toks

def rebuild_mutseq_from_tokens(mutant: str, wt_seq: str):
    seq = list(wt_seq)
    for (a, pos, b) in parse_mutant_str(mutant):
        assert 1 <= pos <= len(seq)
        assert seq[pos-1] == a, f"WT mismatch at {pos}"
        seq[pos-1] = b
    return "".join(seq)

def add_mutated_sequence_if_missing(df, wt_seq, mutant_col="mutant"):
    if "mutated_sequence" in df.columns: 
        return df
    if mutant_col not in df.columns:
        return df
    df = df.copy()
    df["mutated_sequence"] = df[mutant_col].map(lambda m: rebuild_mutseq_from_tokens(m, wt_seq))
    return df

def spearman_on(df, pred_col, name, true_col=TRUE_COL):
    m = df[[pred_col, true_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if m[pred_col].nunique() <= 1 or m[true_col].nunique() <= 1:
        print(f"{name}: constant/empty; skip")
        return np.nan, 0
    rho, _ = spearmanr(m[pred_col], m[true_col])
    print(f"{name}: Spearman ρ = {rho:.3f}  (n={len(m)})")
    return rho, len(m)

def kendall_on(df, pred_col, name, true_col=TRUE_COL):
    m = df[[pred_col, true_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if m[pred_col].nunique() <= 1 or m[true_col].nunique() <= 1:
        print(f"{name}: constant/empty; skip")
        return np.nan, 0
    tau, _ = kendalltau(m[pred_col], m[true_col])
    print(f"{name}: Kendall τ = {tau:.3f}  (n={len(m)})")
    return tau, len(m)

def baseline_cutoff(df, wt_seq=None):
    # If actual WT row exists, use its DMS; otherwise use median as a robust fallback.
    if wt_seq is not None and "mutated_sequence" in df.columns:
        mask = (df["mutated_sequence"] == wt_seq)
        if mask.any():
            return df.loc[mask, TRUE_COL].mean()
    return df[TRUE_COL].median()

def topk_hitrate_enrichment(df, pred_col, k_list=TOPK_LIST, wt_seq=None):
    df = df.copy()
    thr = baseline_cutoff(df, wt_seq=wt_seq)
    df["above_WT"] = numeric_series(df[TRUE_COL]) > thr
    base = df["above_WT"].mean()
    df = df.sort_values(pred_col, ascending=False).reset_index(drop=True)
    rows = []
    for k in k_list:
        k = min(k, len(df))
        hit = df.iloc[:k]["above_WT"].mean()
        enrich = (hit / base) if base > 0 else np.nan
        rows.append((k, hit, base, enrich))
    out = pd.DataFrame(rows, columns=["k", "hit_rate", "baseline_frac", "enrichment"])
    return out

def ndcg_at_k(df, pred_col, k_list=TOPK_LIST, rel_col=TRUE_COL):
    # Normalize relevance to [0,1] by min-max within the set (robust, monotonic)
    vals = numeric_series(df[rel_col]).values
    mn, mx = np.nanmin(vals), np.nanmax(vals)
    rel = (vals - mn) / (mx - mn + 1e-12)
    order = np.argsort(-numeric_series(df[pred_col]).values)
    rel_sorted = rel[order]
    def dcg(x):
        i = np.arange(1, len(x)+1)
        return np.sum((2**x - 1) / np.log2(i + 1))
    out = []
    for k in k_list:
        k = min(k, len(df))
        dcg_k = dcg(rel_sorted[:k])
        ideal = dcg(np.sort(rel)[::-1][:k])
        ndcg = dcg_k / (ideal + 1e-12)
        out.append((k, ndcg))
    return pd.DataFrame(out, columns=["k","nDCG"])

def singles_position_normalized_spearman(df, pred_col, mutant_col):
    """Spearman per position for singles with n>=MIN_POS_N; returns (table, weighted_mean_rho)."""
    sing = df[df["num_subs"] == 1].copy()
    if len(sing) == 0:
        print("No singles found.")
        return pd.DataFrame(), np.nan
    sing["pos"] = sing[mutant_col].map(lambda m: parse_mutant_str(m)[0][1] if isinstance(m,str) else np.nan)
    by = []
    for p, grp in sing.groupby("pos"):
        grp = grp[[pred_col, TRUE_COL]].dropna()
        if len(grp) >= MIN_POS_N and grp[pred_col].nunique() > 2 and grp[TRUE_COL].nunique() > 2:
            r,_ = spearmanr(grp[pred_col], grp[TRUE_COL])
            by.append((int(p), len(grp), r))
    tab = pd.DataFrame(by, columns=["pos","n","rho"]).sort_values("rho", ascending=False)
    if len(tab):
        w_avg = np.average(tab["rho"], weights=tab["n"])
    else:
        w_avg = np.nan
    return tab, w_avg

# -------------------------

# %%
# Load data
# -------------------------
ref = pd.read_csv(REF_CSV)
WT_SEQ = ref.iloc[0]["target_seq"]

wt = pd.read_csv(OURS_WT_CSV)
mc = pd.read_csv(OURS_MC_CSV)
fr = pd.read_csv(FRIEND_CSV)

# Standardize types
for df in (wt, mc, fr):
    if TRUE_COL in df.columns:
        df[TRUE_COL] = numeric_series(df[TRUE_COL])

# Ensure friend has mutated_sequence if we need intersections later
if "mutated_sequence" not in fr.columns:
    if "mutant" in fr.columns:
        fr = add_mutated_sequence_if_missing(fr, WT_SEQ, mutant_col="mutant")

# Pick prediction columns
pred_wt = "esm1v_zero_shot" if "esm1v_zero_shot" in wt.columns else choose_pred_column(wt, prefer=("esm1v_zero_shot",))
pred_mc = "esm1v_zero_shot_mc" if "esm1v_zero_shot_mc" in mc.columns else choose_pred_column(mc, prefer=("esm1v_zero_shot_mc",))
pred_fr = "esm1v_zero_shot" if "esm1v_zero_shot" in fr.columns else choose_pred_column(fr, prefer=("esm1v_zero_shot",))

print("Using columns -> WT:", pred_wt, " MC:", pred_mc, " Friend:", pred_fr)

# -------------------------

# %%
# Core benchmarks
# -------------------------
print("\n=== SPEARMAN / KENDALL ===")
spearman_on(wt, pred_wt, "Ours WT ctx: ALL")
if "num_subs" in wt.columns:
    spearman_on(wt[wt["num_subs"]==1], pred_wt, "Ours WT ctx: singles")
    spearman_on(wt[wt["num_subs"]<=2], pred_wt, "Ours WT ctx: ≤2")
kendall_on(wt, pred_wt, "Ours WT ctx: ALL (τ)")

spearman_on(mc, pred_mc, "Ours mutant ctx: ALL")
if "num_subs" in mc.columns:
    spearman_on(mc[mc["num_subs"]==1], pred_mc, "Ours mutant ctx: singles")
    spearman_on(mc[mc["num_subs"]<=2], pred_mc, "Ours mutant ctx: ≤2")
kendall_on(mc, pred_mc, "Ours mutant ctx: ALL (τ)")

spearman_on(fr, pred_fr, "Friend: ALL")
kendall_on(fr, pred_fr, "Friend: ALL (τ)")

print("\n=== TOP-K HIT-RATE / ENRICHMENT (ours mutant ctx; ≤2 if available) ===")
pool_for_topk = mc if "num_subs" not in mc.columns else mc[mc["num_subs"]<=2]
topk = topk_hitrate_enrichment(pool_for_topk, pred_mc, TOPK_LIST, wt_seq=WT_SEQ)
print(topk)

print("\n=== nDCG@k (ours mutant ctx; same pool as top-k) ===")
print(ndcg_at_k(pool_for_topk, pred_mc, TOPK_LIST, rel_col=TRUE_COL))

print("\n=== POSITION-NORMALIZED SPEARMAN on singles (ours mutant ctx) ===")
mutant_col = "any_mutant" if "any_mutant" in mc.columns else ("mutant" if "mutant" in mc.columns else None)
if mutant_col is None:
    print("No mutant token column found; skipping per-position analysis.")
else:
    tab, wavg = singles_position_normalized_spearman(mc, pred_mc, mutant_col)
    print(f"Weighted mean ρ across positions with n≥{MIN_POS_N}: {wavg:.3f}" if not np.isnan(wavg) else "No positions with sufficient singles.")
    if len(tab):
        print("\nTop 10 positions by ρ:\n", tab.head(10).to_string(index=False))
        print("\nBottom 10 positions by ρ:\n", tab.tail(10).to_string(index=False))

print("\n=== INTERSECTION (ours mutant ctx ↔ friend) ===")
if "mutated_sequence" in fr.columns and "mutated_sequence" in mc.columns:
    inter = pd.merge(mc[["mutated_sequence", TRUE_COL, pred_mc, "num_subs"]] if "num_subs" in mc.columns else mc[["mutated_sequence", TRUE_COL, pred_mc]],
                     fr[["mutated_sequence", TRUE_COL, pred_fr]],
                     on="mutated_sequence", suffixes=("_ours","_friend")).dropna()
    print("Intersection size:", len(inter))
    # Spearman on same rows
    spearman_on(inter, pred_mc, "Ours mutant ctx (intersection)", true_col=TRUE_COL+"_ours")
    spearman_on(inter, pred_fr, "Friend (intersection)", true_col=TRUE_COL+"_ours")
    # Singles on intersection (if available)
    if "num_subs" in inter.columns:
        sing_inter = inter[inter["num_subs"]==1]
        spearman_on(sing_inter, pred_mc, "Ours mutant ctx singles (intersection)", true_col=TRUE_COL+"_ours")
else:
    print("Cannot align by mutated_sequence; friend file lacks that column.")

# %%
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

# Files
mc = pd.read_csv("gfp_dms_with_zeroshot_mutantctx.csv")      # our mutant-context ≤2 table
full = pd.read_csv("gfp_dms.csv")                            # includes WT row (num_subs==0)
ref  = pd.read_csv("ref_gfp.csv"); WT_SEQ = ref.iloc[0]["target_seq"]

# Find the true WT DMS score
WT_rows = full[full["mutated_sequence"] == WT_SEQ]
assert len(WT_rows) >= 1, "Couldn't find WT row in gfp_dms.csv"
WT_score = WT_rows["DMS_score"].mean()

# Define label: above WT
pool = mc.copy()  # ≤2 subs pool
pool["above_WT"] = pool["DMS_score"] > WT_score

# Helper: top-k hit-rate using true WT threshold
def topk_hit(df, score_col, k):
    d = df.sort_values(score_col, ascending=False).head(min(k, len(df)))
    return (d["above_WT"]).mean()

# Baseline prevalence (random expectation)
baseline = pool["above_WT"].mean()

# Compute hit-rate and enrichment
for k in [24, 96, 384]:
    hit = topk_hit(pool, "esm1v_zero_shot_mc", k)
    enrich = hit / baseline if baseline > 0 else np.nan
    print(f"k={k:>3}  hit={hit:.3f}  baseline={baseline:.3f}  enrichment={enrich:.2f}")

# ROC-AUC and PR-AUC (probability-of-being-above-WT vs score)
y_true = pool["above_WT"].astype(int).values
y_score = pool["esm1v_zero_shot_mc"].values
print("ROC-AUC:", roc_auc_score(y_true, y_score))
print("PR-AUC :", average_precision_score(y_true, y_score))


# %%
import pandas as pd, numpy as np
from scipy.stats import spearmanr, kendalltau

# Inputs
MC_CSV   = "gfp_dms_with_zeroshot_mutantctx.csv"   # our mutant-context scores (≤2)
FULL_CSV = "gfp_dms.csv"                           # aggregated table (may lack WT)
RAW_CSV  = "GFP_AEQVI_Sarkisyan_2016.csv"          # original assay file (might have WT)
REF_CSV  = "ref_gfp.csv"
SCORE    = "DMS_score"
PRED     = "esm1v_zero_shot_mc"

mc   = pd.read_csv(MC_CSV)
full = pd.read_csv(FULL_CSV)
ref  = pd.read_csv(REF_CSV)
WT_SEQ = ref.iloc[0]["target_seq"]

def find_wt_score():
    # 1) Look for WT in aggregated table
    if "mutated_sequence" in full.columns:
        wt_rows = full[full["mutated_sequence"] == WT_SEQ]
        if len(wt_rows):
            return float(wt_rows[SCORE].mean()), "WT in gfp_dms.csv"

    # 2) Try raw assay file: empty/NaN/explicit labels sometimes denote WT
    try:
        raw = pd.read_csv(RAW_CSV)
        # candidates: mutant is NaN, empty string, or equals 'WT' (case-insensitive)
        cand = raw[(~raw["mutant"].astype(str).str.strip().astype(str).str.len().fillna(0).astype(int).astype(bool)==False)]
    except Exception:
        cand = pd.DataFrame()

    # If nothing obvious, give up
    return None, "WT not found"

wt_score, wt_source = find_wt_score()
print("WT lookup:", wt_source, "=>", wt_score)

# ------------------------------
# Rank-correlation (context)
# ------------------------------
def corr_report(df, pred, name):
    m = df[[pred, SCORE]].replace([np.inf,-np.inf], np.nan).dropna()
    if m[pred].nunique() <= 1 or m[SCORE].nunique() <= 1:
        print(f"{name}: constant/empty; skip")
        return
    r,_ = spearmanr(m[pred], m[SCORE])
    t,_ = kendalltau(m[pred], m[SCORE])
    print(f"{name}: Spearman ρ={r:.3f}, Kendall τ={t:.3f} (n={len(m)})")

corr_report(mc, PRED, "Ours mutant ctx (ALL)")
if "num_subs" in mc.columns:
    corr_report(mc[mc["num_subs"]==1], PRED, "Ours mutant ctx (singles)")
    corr_report(mc[mc["num_subs"]<=2], PRED, "Ours mutant ctx (≤2)")

# ------------------------------
# Top-k hit-rate / enrichment
# ------------------------------
def topk_hit_enrich(df, pred, k, thr):
    d = df.sort_values(pred, ascending=False).head(min(k, len(df)))
    hit = (d[SCORE] > thr).mean()
    base = (df[SCORE] > thr).mean()
    enrich = hit / base if base > 0 else np.nan
    return hit, base, enrich

def report_topk(df, pred, label, ks=(24,96,384), thr=None):
    print(f"\nTop-k vs {label}:")
    for k in ks:
        h,b,e = topk_hit_enrich(df, pred, k, thr)
        print(f"k={k:>3}  hit={h:.3f}  baseline={b:.3f}  enrichment={e:.2f}")

pool = mc if "num_subs" not in mc.columns else mc[mc["num_subs"]<=2].copy()

# A) If we have a real WT score, use it
if wt_score is not None:
    report_topk(pool, PRED, "WT threshold", thr=wt_score)

# B) Always also show median/Q80/Q90 thresholds for context
thr_med = pool[SCORE].median()
thr_q80 = pool[SCORE].quantile(0.80)
thr_q90 = pool[SCORE].quantile(0.90)

report_topk(pool, PRED, "median threshold", thr=thr_med)
report_topk(pool, PRED, "80th percentile",  thr=thr_q80)
report_topk(pool, PRED, "90th percentile",  thr=thr_q90)


# %%
# =========================
# PLL vs Masked-Marginal Bench (same rows)
# =========================
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score

PLL_CSV = "gfp_dms_with_plldelta.csv"
MC_CSV  = "gfp_dms_with_zeroshot_mutantctx.csv"
WT_CSV  = "gfp_dms_with_zeroshot.csv"
FRIEND  = "gfp_with_esm1v.csv"  # optional

TRUE = "DMS_score"
TOPK = (24, 96, 384)

def _num(s): return pd.to_numeric(s, errors="coerce")

def spearman_tau(df, col, name):
    m = df[[col, TRUE]].replace([np.inf,-np.inf], np.nan).dropna()
    if m[col].nunique() <= 1 or m[TRUE].nunique() <= 1:
        print(f"{name}: constant/empty"); return np.nan, np.nan, 0
    r,_ = spearmanr(m[col], m[TRUE]); t,_ = kendalltau(m[col], m[TRUE])
    print(f"{name}: Spearman={r:.3f}  Kendall={t:.3f}  (n={len(m)})")
    return r,t,len(m)

def topk_table(df, score_col, ks=TOPK, thr=None, label=""):
    out=[]
    base = (df[TRUE] > thr).mean() if thr is not None else np.nan
    d = df.sort_values(score_col, ascending=False)
    for k in ks:
        k = min(k, len(d))
        hit = (d.iloc[:k][TRUE] > thr).mean() if thr is not None else np.nan
        enr = (hit/base) if (thr is not None and base>0) else np.nan
        out.append((label, score_col, k, hit, base, enr))
    return pd.DataFrame(out, columns=["label","score","k","hit","baseline","enrichment"])

def ndcg_at_k(df, score_col, ks=TOPK):
    vals = _num(df[TRUE]).values
    mn, mx = np.nanmin(vals), np.nanmax(vals)
    rel = (vals - mn) / (mx - mn + 1e-12)
    order = np.argsort(-_num(df[score_col]).values)
    rel_sorted = rel[order]
    def dcg(x):
        i = np.arange(1, len(x)+1)
        return np.sum((2**x - 1) / np.log2(i + 1))
    out=[]
    for k in ks:
        k = min(k, len(df))
        dcg_k = dcg(rel_sorted[:k])
        ideal = dcg(np.sort(rel)[::-1][:k])
        out.append((score_col, k, dcg_k/(ideal+1e-12)))
    return pd.DataFrame(out, columns=["score","k","nDCG"])

def zscore_within(df, col, by):
    g = df.groupby(by)[col]
    mu = g.transform("mean"); sd = g.transform("std").replace(0, np.nan)
    return (df[col]-mu)/sd

# --- Load files ---
pll = pd.read_csv(PLL_CSV)
mc  = pd.read_csv(MC_CSV)
wt  = pd.read_csv(WT_CSV)
for df in (pll, mc, wt):
    if TRUE in df.columns: df[TRUE] = _num(df[TRUE])

# Align everyone onto PLL rows (same variants for fair comparison)
bench = pll[["mutated_sequence", TRUE, "num_subs", "pll_delta"]].copy()
bench = bench.merge(mc[["mutated_sequence","esm1v_zero_shot_mc"]], on="mutated_sequence", how="left")
bench = bench.merge(wt[["mutated_sequence","esm1v_zero_shot"]], on="mutated_sequence", how="left")

# Optional friend
try:
    fr = pd.read_csv(FRIEND)
    if "mutated_sequence" in fr.columns and "esm1v_zero_shot" in fr.columns:
        bench = bench.merge(fr[["mutated_sequence","esm1v_zero_shot"]].rename(columns={"esm1v_zero_shot":"friend_zs"}),
                            on="mutated_sequence", how="left")
    else:
        bench["friend_zs"] = np.nan
except Exception:
    bench["friend_zs"] = np.nan

# --- Basic correlations (ALL & singles) ---
print("=== Spearman / Kendall (same rows as PLL file) ===")
for col,label in [("pll_delta","PLLΔ"),
                  ("esm1v_zero_shot_mc","MM mutant-ctx"),
                  ("esm1v_zero_shot","MM WT-ctx"),
                  ("friend_zs","Friend (if any)")]:
    if col not in bench.columns: continue
    spearman_tau(bench, col, f"{label} - ALL")
    sing = bench[bench["num_subs"]==1]
    if len(sing): spearman_tau(sing, col, f"{label} - singles")

# --- Hamming-class calibration (z-score within num_subs) + ensemble ---
bench["pll_z"]  = zscore_within(bench, "pll_delta", "num_subs")
bench["mmc_z"]  = zscore_within(bench, "esm1v_zero_shot_mc", "num_subs")
bench["mmw_z"]  = zscore_within(bench, "esm1v_zero_shot", "num_subs")
bench["ens_z"]  = bench[["pll_z","mmc_z"]].mean(axis=1)  # simple blend

print("\n=== Correlations after Hamming-class z-norm ===")
for col in ["pll_z","mmc_z","mmw_z","ens_z"]:
    if col in bench.columns:
        spearman_tau(bench, col, f"{col} - ALL")
        sing = bench[bench["num_subs"]==1]
        if len(sing): spearman_tau(sing, col, f"{col} - singles")

# --- Top-k @ median / Q80 / Q90 (conservative since WT missing) ---
thr_med = bench[TRUE].median()
thr_q80 = bench[TRUE].quantile(0.80)
thr_q90 = bench[TRUE].quantile(0.90)

def topk_block(df, label):
    rows=[]
    for col in ["pll_delta","esm1v_zero_shot_mc","esm1v_zero_shot","ens_z"]:
        if col not in df.columns: continue
        rows.append(topk_table(df, col, TOPK, thr=thr_med, label=label+" vs median"))
        rows.append(topk_table(df, col, TOPK, thr=thr_q80, label=label+" vs Q80"))
        rows.append(topk_table(df, col, TOPK, thr=thr_q90, label=label+" vs Q90"))
    return pd.concat(rows, ignore_index=True)

print("\n=== Top-k hit-rate / enrichment (same rows; proxy thresholds) ===")
tk = topk_block(bench, "ALL")
print(tk.to_string(index=False))

# --- nDCG@k (continuous) ---
print("\n=== nDCG@k (continuous DMS) ===")
nd = []
for col in ["pll_delta","esm1v_zero_shot_mc","esm1v_zero_shot","ens_z"]:
    if col in bench.columns:
        nd.append(ndcg_at_k(bench, col, TOPK))
nd = pd.concat(nd, ignore_index=True) if len(nd) else pd.DataFrame()
print(nd.to_string(index=False))

# --- AUCs (treat “above median” as positive) ---
print("\n=== ROC-AUC / PR-AUC vs median threshold (sanity, proxy) ===")
y = (bench[TRUE] > thr_med).astype(int).values
for col in ["pll_delta","esm1v_zero_shot_mc","esm1v_zero_shot","ens_z"]:
    if col not in bench.columns: continue
    s = pd.to_numeric(bench[col], errors="coerce").fillna(0).values
    try:
        ra = roc_auc_score(y, s); ap = average_precision_score(y, s)
        print(f"{col:>20}: ROC-AUC={ra:.3f}  PR-AUC={ap:.3f}")
    except Exception as e:
        print(f"{col}: AUC error -> {e}")


# %%
from src.esm_feats import load_esm1v, zero_shot_dataframe_additive_fast

ref = pd.read_csv("ref_gfp.csv")
wt_seq = ref.iloc[0]["target_seq"]

df = pd.read_csv("GFP_AEQVI_Sarkisyan_2016.csv")  # or "gfp_dms.csv"
_ = load_esm1v(device="cuda")                      # use your RTX 4070S

df_add = zero_shot_dataframe_additive_fast(df, wt_seq, mutant_col="mutant")
df_add.to_csv("gfp_dms_with_additive_zeroshot.csv", index=False)

# %%
from scipy.stats import spearmanr

df = pd.read_csv("gfp_dms_with_additive_zeroshot.csv")
rho = spearmanr(df["esm1v_zero_shot_add"], df["DMS_score"], nan_policy="omit").correlation
print("Global Spearman (additive WT-ctx):", rho)

# singles-only sanity (often higher)
import re
is_single = df["mutant"].astype(str).str.match(r"^[A-Z]\d+[A-Z]$")
rho1 = spearmanr(df.loc[is_single, "esm1v_zero_shot_add"], df.loc[is_single, "DMS_score"]).correlation
print("Singles-only Spearman:", rho1)

# %%
import pandas as pd
from scipy.stats import spearmanr
from src.esm_feats import zero_shot_dataframe_additive_fast

ref = pd.read_csv("ref_gfp.csv"); WT = ref.iloc[0]["target_seq"]
pll = pd.read_csv("gfp_dms_with_plldelta.csv")[["mutated_sequence","DMS_score","num_subs"]].dropna()

pll_add = zero_shot_dataframe_additive_fast(pll, WT, seq_col="mutated_sequence")
rho = spearmanr(pll_add["esm1v_zero_shot_add"], pll_add["DMS_score"]).correlation
print("Spearman on PLL rows (WT-additive):", rho)


# %%
# Full-assay ESM benchmarks on GFP (Sarkisyan 2016)

import re, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

# ---- project bits
from src.esm_feats import (
    load_esm1v, parse_mutant_str, seq_to_mutant,
    zero_shot_dataframe_mutantctx_batched  # mutant-context scorer (batched)
)
# ^ mutant-ctx function is defined in your esm_feats.py and returns 'esm1v_zero_shot_mc'. 
#   It builds mutant sequences and masks only mutated sites (your friend’s recipe). 
#   Ref lines (docstring+loop) in your file.  # see chat notes

# === Additive WT-context (paper-style) ===
# If you already added zero_shot_dataframe_additive_fast earlier, use that.
# I include a local drop-in here so this cell is self-contained.

import torch, json, hashlib, os
import esm

# Simple cache dirs (same as your module’s convention)
from pathlib import Path
CACHE_DIR = Path("data/cache"); ZSHOT_DIR = CACHE_DIR / "zeroshot"
ZSHOT_DIR.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def _precompute_wt_mask_logprobs(wt_seq, model_tag="esm1v_t33_650M_UR90S_1", batch_size=96, device="cuda"):
    # cache key per WT
    key = hashlib.sha1(("WTMASK|" + wt_seq).encode()).hexdigest()
    cache = ZSHOT_DIR / f"wtmask_{key}.npz"
    if cache.exists():
        dat = np.load(cache, allow_pickle=False)
        aa_list = json.loads(dat["aa_list"].tobytes().decode("utf-8"))
        return dat["table"], aa_list

    model, alphabet, batch_converter, dev0 = load_esm1v(model_tag=model_tag, device=device)
    aa_list = list(alphabet.standard_toks)
    aa_idx  = torch.tensor([alphabet.get_idx(a) for a in aa_list], device=dev0)
    mask_idx = alphabet.mask_idx

    _, _, wt_tok = batch_converter([("WT", wt_seq)])
    wt_tok = wt_tok.to(dev0)
    L = len(wt_seq)
    table = np.zeros((L, len(aa_list)), dtype=np.float32)

    positions = list(range(1, L + 1))
    for s in range(0, L, batch_size):
        chunk = positions[s:s+batch_size]
        toks = wt_tok.repeat(len(chunk), 1)
        for b, pos in enumerate(chunk):
            toks[b, pos] = mask_idx
        out = model(toks, repr_layers=[], return_contacts=False)
        logp = torch.log_softmax(out["logits"], dim=-1)
        for b, pos in enumerate(chunk):
            table[pos-1, :] = logp[b, pos, aa_idx].detach().cpu().numpy()

    np.savez(cache, table=table, aa_list=np.frombuffer(json.dumps(aa_list).encode("utf-8"), dtype=np.uint8))
    return table, aa_list

def additive_wt_score_from_table(mutant, wt_seq, table, aa_list):
    aa_to_col = {aa:j for j,aa in enumerate(aa_list)}
    total = 0.0
    for tok in str(mutant).split(":"):
        if not tok: continue
        old, pos, new = tok[0], int(tok[1:-1]), tok[-1]
        total += float(table[pos-1, aa_to_col[new]] - table[pos-1, aa_to_col[old]])
    return total

def zero_shot_dataframe_additive_fast(df, wt_seq, mutant_col="mutant", seq_col="mutated_sequence",
                                      model_tag="esm1v_t33_650M_UR90S_1", device="cuda"):
    table, aa_list = _precompute_wt_mask_logprobs(wt_seq, model_tag=model_tag, device=device)
    if mutant_col in df.columns:
        muts = df[mutant_col].astype(str).tolist()
    else:
        assert seq_col in df.columns, f"Need '{mutant_col}' or '{seq_col}'"
        muts = [seq_to_mutant(wt_seq, s) for s in df[seq_col].astype(str)]
    scores = [additive_wt_score_from_table(m, wt_seq, table, aa_list) for m in muts]
    out = df.copy()
    out["esm1v_zero_shot_add"] = np.array(scores, dtype=np.float32)
    if mutant_col not in out.columns: out[mutant_col] = muts
    return out

# === (Optional) PLLΔ (batched, sampled) ===
# Warning: True PLL for ALL rows is heavy (≈ L masks per sequence).
# We default to a stratified sample; set SAMPLE_PLL_N=None to run all (can take a long time).
@torch.no_grad()
def pll_delta_dataframe(df, wt_seq, seq_col="mutated_sequence", model_tag="esm1v_t33_650M_UR90S_1",
                        batch_size_masks=128, device="cuda", SAMPLE_PLL_N=5000, seed=123):
    # Ensure mutant sequences
    if seq_col not in df.columns:
        assert "mutant" in df.columns, "Need mutated_sequence or mutant"
        df = df.copy()
        df[seq_col] = df["mutant"].map(lambda m: _rebuild_mutseq_from_tokens(m, wt_seq))
    L = len(wt_seq)
    # Sample for speed (stratify by num_subs if present)
    use = df
    if SAMPLE_PLL_N is not None and len(df) > SAMPLE_PLL_N:
        if "num_subs" in df.columns:
            use = (df.groupby("num_subs", group_keys=False)
                     .apply(lambda g: g.sample(min(len(g), max(50, int(SAMPLE_PLL_N * len(g)/len(df)))), random_state=seed))
                     .reset_index(drop=True))
        else:
            use = df.sample(SAMPLE_PLL_N, random_state=seed).reset_index(drop=True)
    # Model
    model, alphabet, batch_converter, dev0 = load_esm1v(model_tag=model_tag, device=device)
    mask_idx = alphabet.mask_idx

    # Precompute PLL(WT) once using WT masked table
    wt_table, aa_list = _precompute_wt_mask_logprobs(wt_seq, model_tag=model_tag, device=device)
    aa_to_col = {aa:j for j,aa in enumerate(aa_list)}
    pll_wt = float(sum(wt_table[i, aa_to_col[wt_seq[i]]] for i in range(L)))

    # Compute PLL(mutant) by masking every position on the mutant context (batched)
    seqs = use[seq_col].tolist()
    pll_m = np.zeros(len(seqs), dtype=np.float32)

    # Tokenize all mutant sequences once
    names = [("seq", s) for s in seqs]
    _, _, toks_all = batch_converter(names)
    toks_all = toks_all.to(dev0)

    # Build [B*L] masked copies in chunks
    B = toks_all.size(0)
    # We’ll iterate positions and build a batch for each position to preserve memory
    for pos in range(1, L+1):
        # mask pos for every sequence
        t = toks_all.clone()
        t[:, pos] = mask_idx
        # run in mini-batches over sequences dimension
        for a in range(0, B, batch_size_masks):
            b = min(a + batch_size_masks, B)
            out = model(t[a:b], repr_layers=[], return_contacts=False)
            logp = torch.log_softmax(out["logits"], dim=-1)
            # gather the log prob of the actual AA at pos for each seq
            # Need the residue letter for each seq at 'pos'
            letters = [seqs[i][pos-1] for i in range(a, b)]
            idxs = torch.tensor([alphabet.get_idx(ch) for ch in letters], device=dev0)
            pll_m[a:b] += logp[range(b-a), pos, idxs].detach().cpu().numpy().astype(np.float32)

    use = use.copy()
    use["pll_delta"] = pll_m - pll_wt
    return use

def _rebuild_mutseq_from_tokens(mutant, wt_seq):
    seq = list(wt_seq)
    for tok in str(mutant).split(":"):
        if not tok: continue
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", tok)
        if not m: continue
        a, pos, b = m.group(1), int(m.group(2)), m.group(3)
        assert seq[pos-1] == a, f"WT mismatch at {pos}"
        seq[pos-1] = b
    return "".join(seq)

# === Metrics ===
def _num(s): return pd.to_numeric(s, errors="coerce")

def spearman_on(df, pred, true="DMS_score", label=""):
    m = df[[pred, true]].replace([np.inf, -np.inf], np.nan).dropna()
    if m[pred].nunique() <= 1 or m[true].nunique() <= 1: 
        print(f"{label}: constant/empty"); return np.nan
    r,_ = spearmanr(m[pred], m[true]); print(f"{label}: Spearman ρ = {r:.3f} (n={len(m)})"); return r

def kendall_on(df, pred, true="DMS_score", label=""):
    m = df[[pred, true]].replace([np.inf, -np.inf], np.nan).dropna()
    if m[pred].nunique() <= 1 or m[true].nunique() <= 1: 
        print(f"{label}: constant/empty"); return np.nan
    t,_ = kendalltau(m[pred], m[true]); print(f"{label}: Kendall τ = {t:.3f} (n={len(m)})"); return t

def topk_hitrate(df, pred, ks=(24,96,384), true="DMS_score", wt_seq=None):
    # threshold = WT DMS if present in the full table, else median
    thr = df.loc[df["mutated_sequence"]==wt_seq, true].mean() if wt_seq in df["mutated_sequence"].values else df[true].median()
    base = (df[true] > thr).mean()
    D = df.sort_values(pred, ascending=False).reset_index(drop=True)
    rows=[]
    for k in ks:
        k=min(k,len(D)); hit=(D.iloc[:k][true] > thr).mean(); enr=hit/base if base>0 else np.nan
        rows.append((k, hit, base, enr))
    return pd.DataFrame(rows, columns=["k","hit_rate","baseline_frac","enrichment"])

def ndcg_at_k(df, pred, ks=(24,96,384), true="DMS_score"):
    vals = _num(df[true]).values
    mn, mx = np.nanmin(vals), np.nanmax(vals); rel = (vals - mn) / (mx - mn + 1e-12)
    order = np.argsort(-_num(df[pred]).values); rel_sorted = rel[order]
    def dcg(x): 
        i = np.arange(1, len(x)+1); return np.sum((2**x - 1) / np.log2(i + 1))
    out=[]
    for k in ks:
        k=min(k,len(df)); out.append((k, dcg(rel_sorted[:k]) / (dcg(np.sort(rel)[::-1][:k])+1e-12)))
    return pd.DataFrame(out, columns=["k","nDCG"])

# === Load assay & WT
ref = pd.read_csv("ref_gfp.csv")
WT_SEQ = ref.iloc[0]["target_seq"]
full = pd.read_csv("GFP_AEQVI_Sarkisyan_2016.csv").copy()
full["DMS_score"] = _num(full["DMS_score"])

# Ensure mutated_sequence & num_subs on FULL set
def count_subs(m):
    return 0 if (not isinstance(m,str) or m.strip()=="") else len(m.split(":"))
if "mutated_sequence" not in full.columns:
    full["mutated_sequence"] = full["mutant"].map(lambda m: _rebuild_mutseq_from_tokens(m, WT_SEQ))
full["num_subs"] = full["mutant"].map(count_subs)

# === Score computations (full) ===
_ = load_esm1v(device="cuda")  # use GPU

# (1) WT-context additive (paper-style; fast)
full_add = zero_shot_dataframe_additive_fast(full, WT_SEQ, mutant_col="mutant")

# (2) Mutant-context masked-marginal (friend, batched)
full_mc  = zero_shot_dataframe_mutantctx_batched(full, WT_SEQ, mutant_col="mutant", seq_col="mutated_sequence")

# (3) PLLΔ on a sample (change SAMPLE_PLL_N=None to run all)
pll_sample = pll_delta_dataframe(full, WT_SEQ, seq_col="mutated_sequence", SAMPLE_PLL_N=5000)

# Merge all predictions back on mutated_sequence
bench = full_add[["mutated_sequence","DMS_score","num_subs","esm1v_zero_shot_add"]].merge(
          full_mc[["mutated_sequence","esm1v_zero_shot_mc"]], on="mutated_sequence", how="left")

bench = bench.merge(pll_sample[["mutated_sequence","pll_delta"]], on="mutated_sequence", how="left")

# Save combined table
bench.to_csv("gfp_full_esm_scores.csv", index=False)
print("Saved: gfp_full_esm_scores.csv")

# === Reports ===
print("\n=== Rank Correlations (ALL / singles / ≤2) ===")
for col, label in [("esm1v_zero_shot_add","WT-additive"),
                   ("esm1v_zero_shot_mc","Mutant-ctx MM"),
                   ("pll_delta","PLLΔ (sampled)")]:
    if col not in bench.columns: continue
    spearman_on(bench, col, label+" ALL")
    kendall_on(bench,  col, label+" ALL (τ)")
    sing = bench[bench["num_subs"]==1]
    if len(sing): spearman_on(sing, col, label+" singles")
    le2 = bench[bench["num_subs"]<=2]
    if len(le2): spearman_on(le2, col, label+" ≤2")

print("\n=== Top-K hit-rate / enrichment (vs WT or median) ===")
for col in ["esm1v_zero_shot_add", "esm1v_zero_shot_mc", "pll_delta"]:
    if col not in bench.columns: continue
    tk = topk_hitrate(bench.dropna(subset=[col]), col, wt_seq=WT_SEQ)
    print(f"\n{col}"); print(tk.to_string(index=False))

print("\n=== nDCG@K (continuous DMS) ===")
for col in ["esm1v_zero_shot_add", "esm1v_zero_shot_mc", "pll_delta"]:
    if col not in bench.columns: continue
    nd = ndcg_at_k(bench.dropna(subset=[col]), col)
    print(f"\n{col}"); print(nd.to_string(index=False))



