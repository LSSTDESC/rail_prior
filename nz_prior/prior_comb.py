import numpy as np
from numpy.linalg import eig, cholesky
from scipy.stats import norm
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorComb(PriorBase):
    """
    Prior for the comb model.
    """
    def __init__(self, ens, M):
        self._prior_base(ens)
        self.M = M
        zmax = np.max(self.z)
        zmin = np.min(self.z)
        dz = (zmax - zmin)/M 
        zmeans = [(zmin + dz/2) + i*dz for i in range(M)]
        self.combs = {}
        for i in np.arange(M):
            self.combs[i] = norm(zmeans[i], dz/2)

    def _find_prior(self):
        self.Ws = self._find_weights()
    
    def _find_weights(self):
        Ws = []
        for nz in self.nzs:
            W = [np.dot(nz, self.combs[i].pdf(self.z)) for i in np.arange(self.M)]
            Ws.append(W/np.sum(W)) 
        return np.array(Ws)

    def _get_prior(self):
        Ws = self.Ws
        mean = np.mean(Ws, axis=0)
        cov = np.cov(Ws.T)
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def _get_params(self):
        return self.Ws

    def _get_params_names(self):
        return ['W_{}'.format(i) for i in range(len(self.Ws.T))]
