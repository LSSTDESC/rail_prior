import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import block_diag
from .prior_base import PriorBase
from .prior_shifts import PriorShifts
from .prior_shifts_widths import PriorShiftsWidths
from .prior_moments import PriorMoments
from .utils import make_cov_posdef, is_pos_def


class PriorSacc(PriorBase):

    def __init__(self, sacc_file,
                 model="Shifts",
                 compute_crosscorrs="Full"):
        if model == "Shifts":
            self.model = PriorShifts
        if model == "ShiftsWidths":
            self.model = PriorShiftsWidths
        if model == "Moments":
            self.model = PriorMoments

        self.compute_crosscorrs = compute_crosscorrs
        self.tracers = sacc_file.tracers
        self.params = self._find_params()
        self.params_names = self._get_params_names()
        self.prior_mean = None
        self.prior_cov = None
        self.prior_chol = None

    def _find_params(self):
        params = []
        for tracer_name in list(self.tracers.keys()):
            tracer = self.tracers[tracer_name]
            ens = tracer.ensemble
            model_obj = self.model(ens)
            params_sets = model_obj._get_params()
            params.append(params_sets)
        return np.array(params)

    def _get_prior(self):
        self.prior_mean = np.array([np.mean(param_sets, axis=1) for param_sets in self.params]).flatten()
        if self.compute_crosscorrs == "Full":
            params = []
            for param_sets in self.params:
                for param_set in param_sets:
                    params.append(param_set)
            params = np.array(params)
            cov = np.cov(params)
        elif self.compute_crosscorrs == "BinWise":
            covs = []
            for p in self.params:
                covs.append(np.cov(p))
            covs = np.array(covs)
            cov = block_diag(*covs)
        elif self.compute_crosscorrs == "None":
            stds = []
            for param_sets in self.params:
                for param_set in param_sets:
                    stds.append(np.std(param_set))
            cov = np.diag(stds**2)
        else:
            raise ValueError("Invalid compute_crosscorrs=={}".format(self.compute_crosscorrs))
        self.prior_cov = make_cov_posdef(cov)
        self.prior_chol = cholesky(cov)

    def _get_params_names(self):
        params_names = []
        for tracer_name in list(self.tracers.keys()):
            tracer = self.tracers[tracer_name]
            ens = tracer.ensemble
            model_obj = self.model(ens)
            params_names_set = model_obj._get_params_names()
            for param_name in params_names_set:
                param_name = tracer_name + "__" + param_name
                params_names.append(param_name)
        return np.array(params_names)
