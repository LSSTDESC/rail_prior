import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal as mvn
from .prior_base import PriorBase


class PriorShifts(PriorBase):
    """
    Projector for the shifts model.
    The shift model assumes that all the variation in the measured
    photometric distributions can be described by a single shift in
    the position of the mean of a fiducial n(z) distribution.

    This shift is calibrated by computing the standard deviations
    of the measured photometric distributions over redshift.
    The shift prior is then given by a Gaussian distribution with
    mean 0 and variance equal to the ratio of the standard deviation
    of the standard deviations to the mean of the standard deviations.
    """
    def __init__(self, ens):
        self._prior_base(ens)
        self._find_prior()

    def _find_prior(self):
        self.shift = self._find_shift()

    def evaluate_model(self, nz, shift):
        """
        Aplies a shift to the given p(z) distribution.
        This is done by interpolating the p(z) distribution
        at the shifted z values and then evaluating it at the
        original z values.
        """
        z = nz[0]
        nz = nz[1]
        z_shift = z + shift
        pz_shift = interp1d(z_shift, nz,
                            kind='linear',
                            fill_value='extrapolate')(z)
        return [z, pz_shift]

    def _find_shift(self):
        z_ms = []
        i = 0
        for nz in self.nzs:
            m = np.mean(nz)
            eq = interp1d(self.z, nz-m, kind='linear', fill_value='extrapolate')
            z_m = fsolve(eq, np.mean(self.z))
            if self.z[0] < z_m < self.z[-1]:
                z_ms.append(z_m)
            else:
                i = i + 1
        if i > 0:
            print("Warning: {} out of {} n(z) distributions have no root.".format(i, len(self.nzs)))
        m_fid = np.mean(self.nz_mean)
        z_m_fid = fsolve(eq, np.mean(self.z))
        z_ms = np.array(z_ms).T - z_m_fid
        shift = [np.mean(z_ms), np.std(z_ms)]

        #stds = np.std(self.nzs, axis=1)  # std of each pz
        #s_stds = np.std(stds)            # std of the z-std
        #m_stds = np.mean(stds)           # mean of the z-std
        #shift = [0, s_stds / m_stds]
        return shift

    def _get_prior(self):
        m, s = self.shift
        return mvn([m], [s**2])
