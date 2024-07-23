from .prior_shifts import PriorShifts
from .prior_shifts_widths import PriorShiftsWidths
from .prior_moments import PriorMoments


class PriorSacc():

    def __init__(self, sacc_file, model="Shifts"):
        if model == "Shifts":
            self.model = PriorShifts
        if model == "ShiftsWidths":
            self.model = PriorShiftsWidths
        if model == "Moments":
            self.model = PriorMoments

        self.tracers = sacc_file.tracers
        self.params = self._find_params()

    def _find_params(self):
        params = []
        for tracer in self.tracers:
            ens = self.tracers[tracer].ensemble
            model_obj = self.model(ens)
            params.append(model_obj._get_params())
        return params
