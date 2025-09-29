import pandas as pd

ref_gfp = pd.read_csv("ref_gfp.csv")
wt_seq = ref_gfp.iloc[0]["target_seq"]

df = pd.read_csv("GFP_AEQVI_Sarkisyan_2016.csv")  # mutant, DMS_score, DMS_score_bin


def apply_mutants(wt, mutant_str):
    seq = list(wt)
    for tok in mutant_str.split(":"):  # e.g., A42G or A42G:D190N
        a, pos, b = tok[0], int(tok[1:-1]), tok[-1]
        # Optional sanity check:
        # assert seq[pos-1] == a, f"WT mismatch at {pos}: expected {a}, found {seq[pos-1]}"
        seq[pos - 1] = b
    return "".join(seq)


df["mutated_sequence"] = df["mutant"].apply(lambda s: apply_mutants(wt_seq, s))
df.to_csv("gfp_dms.csv", index=False)  # keep this as your working dataset
