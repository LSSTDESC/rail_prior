import numpy as np
from numpy.linalg import eig, cholesky
from scipy.stats import multivariate_normal as mvn
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorGP(PriorBase):
    """
    Prior for the moments model.
    The moments model assumes that meausred photometric distribution
    is Gaussian meaning that it can be fully described by its mean and
    covariance matrix. Conceptually, this is equavalent to a 
    Gaussian process regressio for a given p(z). The details can be found 
    in the paper: 2301.11978

    Some measured photometric distributions will possess non-invertible
    covariance matrices. If this is the case, PriorMoments will
    attempt regularize the covariance matrix by adding a small jitter
    to its eigen-values. If this fails, the covariance matrix will be
    diagonalized.
    """
    def __init__(self, ens,
                znodes=None,
                interp_mode='linear'):
        self.interp_mode = interp_mode
        self.znodes = znodes
        self._prior_base(ens)
        self._find_prior()

    def _find_prior(self):
        if self.znodes is None:
            self.znodes = self.z
            self.samples = self.nzs
        else:
            if self.interp_mode == 'linear':
                self.samples = np.array([np.interp(self.znodes, self.z, nz) for nz in self.nzs])
            elif self.interp_mode == 'wiener':
                self.samples = self._wiener_filter(self.znodes, self.z, self.nzs)

    def _wiener_filter(self, z_new, z_old, nzs):
        N, M = len(z_old), len(z_new)
        concat_nzs = []
        for nz in nzs:
            nz_new = np.interp(z_new, z_old, nz)
            concat_nz = np.append(nz_new, nz)
            concat_nzs.append(concat_nz)
        concat_cov = np.cov(np.array(concat_nzs).T)
        Koo = concat_cov[M:, M:]
        Koo_inv = np.linalg.pinv(Koo)
        Kon = concat_cov[:M, M:]
        C = Kon @ Koo_inv
        new_samples = np.array([C @ nz for nz in nzs])
        return new_samples

    def _get_prior(self):
        mean = np.mean(self.samples, axis=0)
        cov = np.cov(self.samples.T)
        cov = make_cov_posdef(cov)
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def _get_params(self):
        return self.samples.T

    def _get_params_names(self):
        return ['nz_{}'.format(i) for i in range(len(self.samples.T))]
