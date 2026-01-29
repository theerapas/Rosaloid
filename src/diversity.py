import numpy as np
from typing import List
from .utils import Proposal

def hamming(s1: str, s2: str) -> int:
    assert len(s1) == len(s2), "Sequences must have the same length"
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def mean_pairwise_hamming(seqs: List[str]) -> float:
    if len(seqs) < 2: return 0.0
    acc, n = 0, 0
    for i in range(len(seqs)):
        for j in range(i+1, len(seqs)):
            acc += hamming(seqs[i], seqs[j]); n += 1
    return acc / max(n, 1)

def pick_diverse(proposals: List[Proposal], k: int, min_hd: int) -> List[Proposal]:
    """
    Greedily selects top k proposals by EI, ensuring each new selection 
    has Hamming distance >= min_hd from all previously selected.
    """
    chosen = []
    # Assumes proposals are already sorted or we sort them here
    # Here we assume caller sorts them (as in original code), but let's be safe and sort desc by EI
    sorted_props = sorted(proposals, key=lambda x: x.ei, reverse=True)
    
    for p in sorted_props:
        if len(chosen) >= k: break
        if (len(chosen) == 0) or all(hamming(p.seq, c.seq) >= min_hd for c in chosen):
            chosen.append(p)
            
    return chosen[:k]
