import re, sys, math, torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr
import os

print(os.getcwd())

# -------- paths (edit if needed) ----------
REF_CSV = "ref_gfp.csv"
ASSAY_CSV = "GFP_AEQVI_Sarkisyan_2016.csv"  # or "gfp_dms.csv"
OUT_CSV = "gfp_with_esm1v.csv"

# -------- load WT and assay ---------------
ref = pd.read_csv(REF_CSV)
assert (
    "target_seq" in ref.columns and len(ref) == 1
), "ref_gfp.csv must have a single row with target_seq"
wt_seq = ref.iloc[0]["target_seq"]
assert isinstance(wt_seq, str) and len(wt_seq) > 0, "WT sequence missing"

df = pd.read_csv(ASSAY_CSV)
need_cols = {"mutant", "DMS_score"}
assert need_cols.issubset(df.columns), f"Assay CSV must have columns: {need_cols}"

# -------- ESM-1v model + alphabet --------
import esm

model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

batch_converter = alphabet.get_batch_converter()
mask_idx = alphabet.mask_idx

# Tokenize WT once
_, _, wt_tokens = batch_converter([("WT", wt_seq)])
wt_tokens = wt_tokens.to(device)
L = wt_tokens.size(1) - 2  # exclude BOS/EOS
assert L == len(wt_seq), "Token length mismatch vs WT"

# --- helpers ---
_tok_re = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_mutant_string(mstr):
    toks = mstr.split(":")
    parsed = []
    for t in toks:
        m = _tok_re.match(t.strip())
        if not m:
            raise ValueError(f"Bad token: {t}")
        a, pos_str, b = m.groups()
        pos1 = int(pos_str)
        if not (1 <= pos1 <= len(wt_seq)):
            raise ValueError(f"Position out of range: {pos1}")
        parsed.append((a, pos1, b))
    return parsed


def aa_to_idx(aa):
    # Map residue letter -> token id in ESM alphabet
    try:
        return alphabet.get_idx(aa)
    except KeyError:
        raise ValueError(f"Unknown residue letter for ESM alphabet: {aa}")


@torch.no_grad()
def score_variant_mutated_background(mutant_str):
    """
    Multi-mutation aware zero-shot score:
    1) Apply all substitutions to WT tokens to form the mutated background.
    2) For each mutated site i: mask site i, run model, take log p(mut_i) - log p(wt_i).
    3) Sum across sites.
    """
    muts = parse_mutant_string(mutant_str)

    # Build mutated background tokens
    toks_full = wt_tokens.clone()
    for a, pos1, b in muts:
        wt_aa = wt_seq[pos1 - 1]
        if wt_aa != a:
            # Sometimes datasets still list original AA; warn but proceed.
            # raise ValueError(f"WT mismatch at {pos1}: expected {a}, WT has {wt_aa}")
            pass
        toks_full[0, pos1] = aa_to_idx(b)

    total = 0.0
    for a, pos1, b in muts:
        toks = toks_full.clone()
        toks[0, pos1] = (
            mask_idx  # mask only the current site; other mutated sites remain in context
        )
        out = model(toks, repr_layers=[], return_contacts=False)
        logits = out["logits"][0, pos1]  # distribution at masked position
        log_probs = F.log_softmax(logits, dim=-1)
        idx_mut = aa_to_idx(b)
        idx_wt = aa_to_idx(a)
        total += (log_probs[idx_mut] - log_probs[idx_wt]).item()
    return total


# -------- compute scores ----------
scores = []
bad = 0
for m in df["mutant"]:
    try:
        s = score_variant_mutated_background(m)
    except Exception as e:
        s = float("nan")
        bad += 1
    scores.append(s)

df["esm1v_zero_shot"] = scores

valid = df[~df["esm1v_zero_shot"].isna()]
rho, p = spearmanr(valid["esm1v_zero_shot"], valid["DMS_score"])
print(
    f"Spearman rho (ESM-1v zero-shot vs DMS_score): {rho:.3f}  (n={len(valid)})  skipped={bad}"
)

df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
