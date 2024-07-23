import numpy as np
from numpy.linalg import cholesky
from .prior_base import PriorBase
from .prior_shifts import PriorShifts
from .prior_shifts_widths import PriorShiftsWidths
from .prior_moments import PriorMoments
from .utils import make_cov_posdef, is_pos_def


class PriorSacc(PriorBase):

    def __init__(self, sacc_file, model="Shifts"):
        if model == "Shifts":
            self.model = PriorShifts
        if model == "ShiftsWidths":
            self.model = PriorShiftsWidths
        if model == "Moments":
            self.model = PriorMoments

        self.tracers = sacc_file.tracers
        self.params = self._find_params()
        self.params_names = self._get_params_names()
        self.prior = None

    def _find_params(self):
        params = []
        for tracer_name in list(self.tracers.keys()):
            tracer = self.tracers[tracer_name]
            ens = tracer.ensemble
            model_obj = self.model(ens)
            params_sets = model_obj._get_params()
            for params_set in params_sets:
                params.append(params_set)

        return np.array(params)

    def _get_prior(self):
        params = self.params
        mean = np.mean(params, axis=1)
        cov = np.cov(params)
        cov = make_cov_posdef(cov)
        chol = cholesky(cov)
        return mean, cov, chol

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
