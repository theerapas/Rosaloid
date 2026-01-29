import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import MLE
except ImportError:
    print("[Surrogate] NGBoost not found. Please install via: pip install ngboost")
    NGBRegressor = None

from .utils import RANDOM_SEED

class NGBSurrogate:
    """Standardize -> PCA -> NGBoost(Normal) → returns (mu, sigma)."""
    def __init__(self, pca_dim: int = 16):
        if NGBRegressor is None:
            raise ImportError("NGBoost is required for NGBSurrogate")
            
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
        base = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED)
        self.ngb = NGBRegressor(
            Dist=Normal, Base=base, Score=MLE,
            natural_gradient=True, n_estimators=500, learning_rate=0.03,
            random_state=RANDOM_SEED, verbose=False
        )
        self.fitted = False

    def fit(self, X_emb: np.ndarray, y: np.ndarray):
        Z = self.scaler.fit_transform(X_emb.astype(np.float32, copy=False))
        Z = self.pca.fit_transform(Z)
        self.ngb.fit(Z, y.astype(np.float32, copy=False))
        self.fitted = True

    def predict(self, X_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.fitted, "Model not fitted yet"
        Z = self.scaler.transform(X_emb.astype(np.float32, copy=False))
        Z = self.pca.transform(Z)
        dist = self.ngb.pred_dist(Z)
        mu, sigma = dist.loc, np.clip(dist.scale, 1e-6, None)
        return mu.astype(np.float32, copy=False), sigma.astype(np.float32, copy=False)
