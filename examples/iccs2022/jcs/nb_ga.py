import numpy as np

import pyximport
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.factory import get_reference_directions

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

from pymoo.core.problem import Problem
import examples.optimization.evol.opt_utils as opt_utils
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.termination import get_termination
from problems import UnconditionalOptimization
from pymoo.operators.selection.rnd import RandomSelection

from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair



from pymoo.algorithms.soo.nonconvex.ga import GA


problem = UnconditionalOptimization(
    order=(6, 7),
    theta_max_degrees=22,
    dx_wl=50,
    dz_wl=0.25
)

mutation = PolynomialMutation(prob=0.0, eta=20, repair=RoundingRepair())

algorithm = GA(
    pop_size=100,
    #selection=RandomSelection(),
    mutation=mutation,
    eliminate_duplicates=True)

termination = get_termination("n_gen", 100000000)

res = minimize(problem,
               algorithm,
               seed=1,
               termination=termination,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))