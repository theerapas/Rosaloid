import pandas as pd, re
from scipy.stats import spearmanr

OUT = "predictpy_output.csv"
ASSAY = "DMS_score"

df = pd.read_csv(OUT)
score_cols = [c for c in df.columns if c.startswith("esm1v_t33_650M_UR90S_")]
df["esm_avg"] = (
    df[score_cols].mean(axis=1)
    if score_cols
    else df[[c for c in df.columns if c.startswith("esm1v_")][0]]
)

pat = re.compile(r"^[A-Z]\d+[A-Z]$")
df = df[df["mutant"].astype(str).str.match(pat)].copy()
df["pos"] = df["mutant"].str.extract(r"(\d+)").astype(int)

# within-position ranks (average method to handle ties gracefully)
df["rank_esm"] = df.groupby("pos")["esm_avg"].rank(method="average")
df["rank_dms"] = df.groupby("pos")[ASSAY].rank(method="average")
df["rank_dms_flip"] = df.groupby("pos")[ASSAY].rank(ascending=False, method="average")

# concat all sites and correlate ranks
r = spearmanr(df["rank_esm"], df["rank_dms"]).correlation
rf = spearmanr(df["rank_esm"], df["rank_dms_flip"]).correlation
print("Position-normalized global Spearman:", r, "(orig),", rf, "(flipped)")
