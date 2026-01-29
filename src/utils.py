import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# =========================
# Configuration / Constants
# =========================
RANDOM_SEED = 6721
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
SEQ_COL = "mutated_sequence"
RAW_SCORE_COL = "DMS_score"
USE_ZSCORE = True

# Map each DMS_id to DMS files (sequence, DMS_score)
# Assuming these are now under data/ or inputs/ (we will handle path correction)
DMS_ID_TO_FILE = {
    "BLAT_ECOLX_Deng_2012": "BLAT_ECOLX_Deng_2012.csv",
    "HMDH_HUMAN_Jiang_2019": "HMDH_HUMAN_Jiang_2019.csv",
    "RDRP_I33A0_Li_2023": "RDRP_I33A0_Li_2023.csv",
    "GFP_AEQVI_Sarkisyan_2016": "dms_le2.csv",
}

@dataclass
class LabeledSeq:
    seq: str
    fitness: float

@dataclass
class Proposal:
    seq: str
    mu: float
    sigma: float
    ei: float

def load_problem(dms_id: str, data_dir: str = "data"):
    # Read WT/meta
    ref_path = os.path.join(data_dir, "DMS_substitutions.csv")
    if not os.path.exists(ref_path):
        # Verify if it might be in inputs/ if data/ is empty
        if os.path.exists(os.path.join("inputs", "DMS_substitutions.csv")):
             ref_path = os.path.join("inputs", "DMS_substitutions.csv")
             data_dir = "inputs" # Switch to inputs if found there

    ref = pd.read_csv(ref_path)
    required_cols = {"DMS_id","target_seq","raw_DMS_directionality"}
    missing = required_cols - set(ref.columns)
    assert not missing, f"WT table missing columns: {missing}"

    row = ref.loc[ref["DMS_id"] == dms_id]
    assert len(row)==1, f"DMS_id '{dms_id}' not found or not unique in {ref_path}"
    wt = str(row.iloc[0]["target_seq"]).strip().upper()
    direction = int(row.iloc[0]["raw_DMS_directionality"])  # 1 = higher better, -1 = lower better
    assert direction in (1,-1), f"raw_DMS_directionality must be 1 or -1 for {dms_id}"

    # Map to DMS file
    assert dms_id in DMS_ID_TO_FILE, f"No DMS file mapped for {dms_id} in DMS_ID_TO_FILE"
    dms_filename = DMS_ID_TO_FILE[dms_id]
    dms_path = os.path.join(data_dir, dms_filename)
    dms = pd.read_csv(dms_path)

    # basic checks
    assert SEQ_COL in dms.columns and RAW_SCORE_COL in dms.columns, \
        f"DMS file must have columns '{SEQ_COL}' and '{RAW_SCORE_COL}'"

    # clean & filter: valid AA + same length as WT
    valid_aas = set(AA_ALPHABET)
    dms[SEQ_COL] = dms[SEQ_COL].astype(str).str.strip().str.upper()
    def is_valid_seq(s): return (len(s)==len(wt)) and all(ch in valid_aas for ch in s)
    dms = dms[dms[SEQ_COL].map(is_valid_seq)].copy()
    dms = dms[~dms[RAW_SCORE_COL].isna()].copy()

    # direction-aware score: make larger = better
    dms["score_directed"] = dms[RAW_SCORE_COL].astype(float) * float(direction)

    # normalization (z-score across this protein's dataset)
    if USE_ZSCORE:
        mu = dms["score_directed"].mean()
        sd = dms["score_directed"].std(ddof=0) + 1e-8
        dms["fitness"] = (dms["score_directed"] - mu) / sd
    else:
        dms["fitness"] = dms["score_directed"]

    return wt, dms
