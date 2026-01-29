import numpy as np
from typing import List

def y_of(seqs: List[str], truth_map: dict) -> np.ndarray:
    return np.array([truth_map.get(s, np.nan) for s in seqs], dtype=float)

def topk_true_in_selected(selected: List[str], pool: List[str], truth_map: dict, k: int) -> float:
    """
    Calculates Hit@K metric: fraction of true top-K sequences from pool that appear in selected.
    """
    # Sort pool by true fitness to find true top K
    pool_sorted = sorted(pool, key=lambda s: truth_map.get(s, -np.inf), reverse=True)
    true_top = set(pool_sorted[:min(k, len(pool_sorted))])
    
    hit = len(set(selected) & true_top)
    denom = min(k, len(selected))
    
    # Original logic: hit / denom where denom is min(k, len(selected)) 
    # But usually Hit@K means fraction of true top K found. 
    # The original notebook used: return (hit / denom) if denom else 0.0
    # Let's preserve original logic.
    return (hit / denom) if denom else 0.0
