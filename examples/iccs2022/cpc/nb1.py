import numpy as np

import pyximport
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pso import PSO

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

from pymoo.core.problem import Problem
import examples.optimization.evol.opt_utils as opt_utils
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.termination import get_termination


class UnconditionalOptimization(Problem):

    def __init__(self, order, theta_max_degrees, dx_wl, dz_wl):
        super().__init__(n_var=2 * (order[0] + order[1]), n_obj=1, xl=-100.0, xu=100.0)
        self.order = order
        self.theta_max_degrees = theta_max_degrees
        self.dx_wl = dx_wl
        self.dz_wl = dz_wl

    def _evaluate(self, x, out, *args, **kwargs):
        res = np.empty(x.shape[0])
        for i in range(0, x.shape[0]):
            res[i] = opt_utils.fit_func_ga(x[i], self.dx_wl, self.dz_wl, self.order, self.theta_max_degrees)
        out["F"] = res


problem = UnconditionalOptimization(
    order=(6, 7),
    theta_max_degrees=22,
    dx_wl=50,
    dz_wl=0.25
)

algorithm_de = DE(
    pop_size=20,
    sampling=LHS(),
    variant="DE/best/1/exp",
    CR=0.9,
    dither="vector",
    jitter=True,
)

algorithm = PSO(pop_size=50, max_velocity_rate=0.001, w=0.9)

termination = get_termination("n_gen", 100000000)

res = minimize(problem,
               algorithm,
               seed=1,
               termination=termination,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))