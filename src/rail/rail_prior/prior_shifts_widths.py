import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal as mvn
from .prior_base import PriorBase


class PriorShiftsWidths(PriorBase):
    """
    Prior for the shifts and widths model.
    The shifts and widths model assumes that the variation in the measured
    photometric distributions can be captured by varying the mean and the
    standard deviation of a fiducial n(z) distribution.

    The calibration method was written by Tilman Tröster.
    The shift prior is given by a Gaussian distributiob with zero mean
    standard deviation the standard deviation in the mean of
    the measured photometric distributions.
    The width is calibrated by computing the standard deviations
    of the measured photometric distributions over redshift.
    The width prior is then given by a Gaussian distribution with
    mean 0 and variance equal to the ratio of the standard deviation
    of the standard deviations to the mean of the standard deviations.
    This is similar to how the shift prior is calibrated in the shift model.
    """
    def __init__(self, ens):
        self._prior_base(ens)
        self._find_prior()

    def _find_prior(self):
        self.shift = self._find_shift()
        self.width = self._find_width()

    def evaluate_model(self, nz, args):
        """
        Aplies a shift and a width to the given p(z) distribution.
        This is done by evluating the n(z) distribution at
        p((z-mu)/width + mu + shift) where mu is the mean redshift
        of the fiducial n(z) distribution and the rescaling by the width.
        Finally the distribution is normalized.
        """
        shift, width = args
        z = nz[0]
        nz = nz[1]
        nz_i = interp1d(z, nz, kind='linear', fill_value='extrapolate')
        mu = np.mean(nz)
        pdf = nz_i((z-mu)/width + mu + shift)/width
        norm = np.sum(pdf)
        return [z, pdf/norm]

    def _find_shift(self):
        mu = np.mean(self.nz_mean)
        ms = [(np.mean(nz)-mu)/mu for nz in self.nzs]   # mean of each nz
        shift = np.mean(self.z)*np.std(ms)                # std of the means
        return shift

    def _find_width(self):
        stds = np.std(self.nzs, axis=1) # std of each nz
        std_std = np.std(stds)          # std of the stds
        std_mean = np.mean(stds)        # mean of the stds
        width = std_std / std_mean  
        return width

    def _get_prior(self):
        mean = np.array([0, 1])
        cov = np.array([
            [self.shift**2, 0],
            [0, self.width**2]])
        return mean, cov
