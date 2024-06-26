import numpy as np
import qp


class PriorBase():
    """
    Base class for priors. Projectors are used to project the measured
    photometric distributions by RAIL onto the space of a given generative
    photometric model for inference.
    This class is not meant to be used directly,
    but to be subclassed by specific projectors.
    The subclasses should implement the following methods:
    - evaluate_model: given a set of parameters, evaluate the model
    - get_prior: return the prior distribution of the model given
    the meadured photometric distributions.
    """
    def __init__(self, ens):
        self._prior_base(ens)

    def _prior_base(self, ens):
        if type(ens) is qp.ensemble.Ensemble:
            z_edges = ens.metadata()['bins'][0]
            z = 0.5 * (z_edges[1:] + z_edges[:-1])
            nzs = ens.objdata()['pdfs']
        elif type(ens) is list:
            z = ens[0]
            nzs = ens[1]
        else:
            raise ValueError("Invalid ensemble type=={}".format(type(ens)))
        self.z = z
        self.nzs = self._normalize(nzs)
        self.nz_mean = np.mean(self.nzs, axis=0)
        self.nz_cov = np.cov(self.nzs, rowvar=False)
        self.prior = None

    def _normalize(self, nzs):
        norms = np.sum(nzs, axis=1)
        nzs = nzs/norms[:, None]
        return nzs

    def evaluate_model(self, nz, args):
        """
        Evaluate the model at the given parameters.
        """
        raise NotImplementedError

    def get_prior(self):
        """
        Returns the calibrated prior distribution for the model
        parameters given the measured photometric distributions.
        """
        if self.prior is None:
            self.prior = self._get_prior()
        return self.prior

    def _get_prior(self):
        raise NotImplementedError

    def sample_prior(self):
        """
        Draws a sample from the prior distribution.
        """
        prior = self.get_prior()
        return prior.rvs()

    def save_prior(self, mode="dist"):
        """
        Saves the prior distribution to a file.
        """
        prior = self.get_prior()
        if mode == "dist":
            return prior
        if mode == "file":
            np.save("prior_mean.npy", prior.mean)
            np.save("prior_cov.npy", prior.cov)
        else:
            raise NotImplementedError
