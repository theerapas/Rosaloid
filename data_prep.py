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
ref = pd.read_csv("DMS_substitutions.csv")
ref_gfp = ref[ref["DMS_id"] == "GFP_AEQVI_Sarkisyan_2016"].copy()

# %%
assert len(ref_gfp) == 1, "More than one or zero matches — check ref[ref.DMS_id.str.contains('GFP', case=False)]"

# %%
wt_seq = ref_gfp.iloc[0]["target_seq"]
ref_gfp.to_csv("ref_gfp.csv", index=False)

# %%
df = pd.read_csv("GFP_AEQVI_Sarkisyan_2016.csv")

# %%
def _tok_ok(tok,wt):
    a, pos, b = tok[0], int(tok[1:-1]), tok[-1]  # e.g. A42G
    return wt[pos-1] == a

# %%
ok = df["mutant"].str.split(":").apply(lambda toks: all(_tok_ok(t, wt_seq) for t in toks))
print(f"fraction of rows consistent with WT: {ok.mean():.3f}")

# %%
# Drop inconsistent rows and construct the mutated sequence
import re
pat = re.compile(r"^([A-Z])(\d+)([A-Z])$")

def apply_mutant(ref_seq, mutant_str):
    seq = list(ref_seq)
    muts = mutant_str.split(":")
    for tok in muts:
        m = pat.match(tok)
        if not m:
            raise ValueError(f"Bad token: {tok}")
        a, pos, b = m.group(1), int(m.group(2)), m.group(3)
        assert 1 <= pos <= len(seq), f"Position out of range: {pos}"
        assert seq[pos-1] == a, f"Ref mismatch at {pos}: expected {a}, got {seq[pos-1]}"
        seq[pos-1] = b
    return "".join(seq), len(muts)

df_ok = df[ok].copy()
df_ok["mutated_sequence"], df_ok["num_subs"] = zip(*df_ok["mutant"].map(lambda s: apply_mutant(wt_seq, s)))


# %%
# Handle duplicates / replicates (some DMS rows refer to the same mutated sequence)
# keep one score per unique sequence (mean across replicates is fine)
agg = (df_ok
       .groupby("mutated_sequence", as_index=False)
       .agg(DMS_score=("DMS_score", "mean"),
            any_mutant=("mutant", "first"),
            num_subs=("num_subs", "first")))

# optional: keep WT row if present; otherwise record WT_score separately
WT_score = agg.loc[agg["mutated_sequence"] == wt_seq, "DMS_score"].mean() if (agg["mutated_sequence"] == wt_seq).any() else None
if WT_score is not None:
    agg["above_WT"] = agg["DMS_score"] > WT_score


# %%
# Save a clean table and a fast score lookup
agg.to_csv("gfp_dms.csv", index=False)
score_of = dict(zip(agg["mutated_sequence"], agg["DMS_score"]))
# End of data preparation

# %%
from src.esm_feats import load_esm1v
model, alphabet, batch_converter, device = load_esm1v()
import torch
print("device:", device, "cuda_available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.version)
# Print the PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Print the CUDA version (will be None if not built with CUDA)
print(f"CUDA Version (from torch): {torch.version.cuda}")

# Check if CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

# %%
import pandas as pd
from src.esm_feats import (
    get_embedding, zero_shot_score,
    embed_dataframe, zero_shot_dataframe, load_esm1v
)


# Load your outputs from Step 1
gfp = pd.read_csv("gfp_dms.csv")              # has mutated_sequence, DMS_score, etc.
ref = pd.read_csv("ref_gfp.csv")
wt_seq = ref.iloc[0]["target_seq"]

# (A) Add zero-shot prior to the subset you'll consider for Round-0
gfp_small = gfp[gfp["num_subs"] <= 2].copy()  # Example: safe local pool
gfp_zs = zero_shot_dataframe(gfp_small, wt_seq, mutant_col="any_mutant") # adds 'esm1v_zero_shot'
gfp_zs.to_csv("gfp_dms_with_zeroshot.csv", index=False)

# (B) Cache embeddings for sequences you’re likely to touch soon (optional precompute)
# You can precompute for gfp_small, or compute on-demand later in Round-0/GP fit.
gfp_emb = embed_dataframe(gfp_small)          # writes a matrix_*.npy, adds 'embedding_path'
gfp_emb.to_csv("gfp_dms_with_embptr.csv", index=False)


# %%
from src.esm_feats import load_esm1v, zero_shot_dataframe_mutantctx_batched

# make sure model is on GPU if available
_ = load_esm1v(device="cuda")

gfp = pd.read_csv("gfp_dms.csv")
ref = pd.read_csv("ref_gfp.csv"); wtseq = ref.iloc[0]["target_seq"]

gfp2 = gfp[gfp["num_subs"] <= 2].reset_index(drop=True)
mutant_col = "any_mutant" if "any_mutant" in gfp2.columns else "mutant"

gfp_mc = zero_shot_dataframe_mutantctx_batched(gfp2, wtseq, mutant_col=mutant_col,
                                               batch_size_seqs=64, batch_size_masks=512)
gfp_mc.to_csv("gfp_dms_with_zeroshot_mutantctx.csv", index=False)
print("Saved:", len(gfp_mc))


# %%

#PLL
import importlib, src.esm_feats as ef
import pandas as pd
from scipy.stats import spearmanr

importlib.reload(ef)
from src.esm_feats import load_esm1v, pll_delta_dataframe_safe

_ = load_esm1v(device="cuda")

gfp = pd.read_csv("gfp_dms.csv")
ref = pd.read_csv("ref_gfp.csv"); wt = ref.iloc[0]["target_seq"]

subset = gfp[gfp["num_subs"] <= 2].reset_index(drop=True)   # or singles

pll_df = pll_delta_dataframe_safe(subset, wt, seq_col="mutated_sequence")
pll_df.to_csv("gfp_dms_with_plldelta.csv", index=False)
print("Saved PLLΔ:", len(pll_df))

m = pll_df[["pll_delta","DMS_score"]].dropna()
rho, _ = spearmanr(m["pll_delta"], m["DMS_score"])
print(f"PLLΔ vs DMS (subset) ρ = {rho:.3f}  (n={len(m)})")



