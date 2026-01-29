import numpy as np
from scipy.stats import norm

def expected_improvement(mu, sigma, best_y, xi=0.01):
    """
    Computes Expected Improvement (EI).
    mu, sigma: arrays of mean and std dev from surrogate
    best_y: best fitness observed so far
    xi: exploration parameter
    """
    imp = mu - best_y - xi
    Z = np.zeros_like(mu)
    nonzero = sigma > 1e-12
    Z[nonzero] = imp[nonzero] / sigma[nonzero]
    
    ei = np.zeros_like(mu)
    ei[nonzero] = imp[nonzero] * norm.cdf(Z[nonzero]) + sigma[nonzero] * norm.pdf(Z[nonzero])
    return np.maximum(ei, 0.0)
