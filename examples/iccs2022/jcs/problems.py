import numpy as np

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

from pymoo.core.problem import Problem
import examples.optimization.evol.opt_utils as opt_utils


class UnconditionalOptimization(Problem):

    def __init__(self, order, theta_max_degrees, dx_wl, dz_wl, **kwargs):
        super().__init__(n_var=2 * (order[0] + order[1]), n_obj=1, xl=-100.0, xu=100.0, vtype=float, **kwargs)
        self.order = order
        self.theta_max_degrees = theta_max_degrees
        self.dx_wl = dx_wl
        self.dz_wl = dz_wl

    def _evaluate(self, x, out, *args, **kwargs):
        res = np.empty(x.shape[0])
        for i in range(0, x.shape[0]):
            res[i] = opt_utils.fit_func_ga(x[i], self.dx_wl, self.dz_wl, self.order, self.theta_max_degrees)
        out["F"] = res