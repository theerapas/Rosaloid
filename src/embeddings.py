import math
import numpy as np
import torch
import esm
from tqdm.auto import tqdm
from typing import List, Optional
from .utils import AA_ALPHABET, RANDOM_SEED

class Embedder:
    """ESM-2 (650M) pooled embedding via mean over residues (exclude BOS/EOS)."""
    def __init__(self, family: str = "esm2", device: Optional[str] = None, layer:int=33, batch_size: int = 64):
        self.family, self.device, self.layer = family, device, layer
        self.batch_size = batch_size
        self._mode = "fallback"
        self._embed_dim = 1280
        self._setup()

    def _setup(self):
        try:
            if self.family == "esm2":
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                model = model.eval()
                if self.device == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda"); self.device = "cuda"
                else:
                    self.device = "cpu"
                self.model = model
                self.alphabet = alphabet
                self.batch_converter = alphabet.get_batch_converter()
                self._mode = "esm"
            else:
                raise RuntimeError("Only esm2 implemented.")
        except Exception as e:
            print(f"[Embedder] Falling back (no ESM): {e}")
            self._mode = "fallback"
            self.device = "cpu"
            self._rng = np.random.RandomState(RANDOM_SEED)

    def _positional_embed(self, seq: str) -> np.ndarray:
        # Simple fallback: average one-hot projections with random projection per position
        L = len(seq); D = self._embed_dim
        aa_to_idx = {aa:i for i,aa in enumerate(AA_ALPHABET)}
        self._pos_proj = getattr(self, "_pos_proj", None)
        if self._pos_proj is None or self._pos_proj.shape != (L, 20, D):
            self._pos_proj = self._rng.normal(0, 1, size=(L, 20, D)).astype(np.float32)
        out = np.zeros((D,), dtype=np.float32)
        for i, aa in enumerate(seq):
            j = aa_to_idx.get(aa, None)
            if j is not None: out += self._pos_proj[i, j]
        return out / max(L, 1)

    def encode(self, seqs: List[str], show_progress: bool = True) -> np.ndarray:
        if self._mode != "esm":
            it = tqdm(seqs, disable=not show_progress, desc="Embedding (fallback)")
            return np.vstack([self._positional_embed(s) for s in it])
        embs = []
        batch_size = self.batch_size
        with torch.no_grad():
            total = math.ceil(len(seqs) / batch_size)
            for start in tqdm(range(0, len(seqs), batch_size),
                              total=total, disable=not show_progress,
                              desc=f"Embedding (ESM-2)"):
                batch = [("seq", s) for s in seqs[start:start+batch_size]]
                _, _, toks = self.batch_converter(batch)
                toks = toks.to(self.device)
                out = self.model(toks, repr_layers=[self.layer], return_contacts=False)
                reps = out["representations"][self.layer]      # [B, L, D]
                pooled = reps[:, 1:-1, :].mean(dim=1).cpu().numpy()  # exclude BOS/EOS
                embs.append(pooled)
        return np.vstack(embs).astype(np.float32, copy=False)
