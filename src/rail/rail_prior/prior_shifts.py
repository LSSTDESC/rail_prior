import numpy as np
from scipy.interpolate import interp1d
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
        self.shifts = self._find_shifts()

    def _find_shifts(self):
        mu = np.mean(self.nz_mean)
        shifts = [(np.mean(nz)-mu)/mu for nz in self.nzs]   # mean of each nz
        shifts = np.mean(self.z)*np.array(shifts)                     # std of the means
        return shifts

    def _get_prior(self):
        shifts = self.shifts
        mean = np.array([np.mean(shifts)])
        cov = np.array([[np.std(shifts)**2]])
        return mean, cov

    def _get_params(self):
        return np.array([self.shifts])

    def _get_params_names(self):
        return np.array(['delta_z'])
