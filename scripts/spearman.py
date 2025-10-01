import pandas as pd
from scipy.stats import spearmanr
import re

OUT = "predictpy_output.csv"  # output from predict.py
SCORE_COL = "esm1v_t33_650M_UR90S_1"  # name added by predict.py
ASSAY_COL = "DMS_score"  # your column

df = pd.read_csv(OUT)

# 0) singles only & valid tokens
pat = re.compile(r"^[A-Z]\d+[A-Z]$")
df = df[df["mutant"].astype(str).str.match(pat)]

# 1) NA drop
v = df[[SCORE_COL, ASSAY_COL]].dropna().copy()

# 2) basic stats of the assay column
n = len(v)
n_unique = v[ASSAY_COL].nunique()
ties = n - n_unique
print(f"n={n}, unique DMS values={n_unique}, ties={ties}")

# 3) Spearman both signs
r = spearmanr(v[SCORE_COL], v[ASSAY_COL]).correlation
r_flip = spearmanr(v[SCORE_COL], -v[ASSAY_COL]).correlation
print("rho:", r, "rho (flipped):", r_flip)
