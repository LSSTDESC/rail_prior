import numpy as np
from scipy.interpolate import interp1d
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
        self.shifts = self._find_shifts()
        self.widths = self._find_widths()

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

    def _find_shifts(self):
        mu = np.mean(self.nz_mean)
        shifts = [(np.mean(nz)-mu)/mu for nz in self.nzs]   # mean of each nz
        shifts = np.mean(self.z)*np.array(shifts)           # std of the means
        return shifts

    def _find_widths(self):
        stds = np.std(self.nzs, axis=1) # std of each nz
        std_mean = np.mean(stds)        # mean of the stds
        widths = stds / std_mean  
        return widths

    def _get_prior(self):
        m_shift = np.mean(self.shifts)
        m_width = np.mean(self.widths)
        s_shift = np.std(self.shifts)
        s_width = np.std(self.widths)
        mean = np.array([m_shift, m_width])
        cov = np.array([
            [s_shift**2, 0],
            [0, s_width**2]])
        return mean, cov

    def _get_params(self):
        return np.array([self.shifts, self.widths])
