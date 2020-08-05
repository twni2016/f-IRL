# '''
# Written by Greg Ver Steeg and Improved by us
# Refer[Original Code]: https://github.com/gregversteeg/NPEET
# '''

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree

# CONTINUOUS ESTIMATORS
def entropy(x, k=3):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        natural logarithm base
        k controls bias-variance trade-off
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n, d = x.shape
    x = add_noise(x)

    const = digamma(n) - digamma(k) + d * np.log(2) # twice the distance
    nn = query_tree(x, x, k)
    return const + d * np.log(nn).mean()

def kldiv(x, xp, k=3):
    """ KL Divergence between p and q for x~p(x), xp~q(x)
    """
    assert k < min(len(x), len(xp)), "Set k smaller than num. samples - 1"
    assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
    x, xp = np.asarray(x), np.asarray(xp)
    x, xp = x.reshape(x.shape[0], -1), xp.reshape(xp.shape[0], -1)
    n, d = x.shape
    m, _ = xp.shape
    x = add_noise(x) # fix np.log(0)=inf issue

    const = np.log(m) - np.log(n - 1)
    nn = query_tree(x, x, k)
    nnp = query_tree(xp, x, k - 1) # (m, k-1) 
    return const + d * (np.log(nnp).mean() - np.log(nn).mean())

def KDE_entropy(x, bw=1.0, kernel='gaussian'):
    x = np.asarray(x)
    n_elements, n_features = x.shape
    kde = KDE(bandwidth=bw, kernel=kernel)
    kde.fit(x)
    return -kde.score(x) / n_elements 

# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_tree(x, xp, k):
    # https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/lnc.py
    # https://github.com/scipy/scipy/issues/9890 p=2 or np.inf 
    tree = cKDTree(x)
    return tree.query(xp, k=k + 1, p=float('inf'))[0][:, k] # chebyshev distance of k+1-th nearest neighbor
