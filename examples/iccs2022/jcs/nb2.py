from pymoo.algorithms.hyperparameters import SingleObjectiveSingleRun, HyperparameterProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.mixed import MixedVariableGA

from problems import UnconditionalOptimization

from pymoo.core.parameters import get_params
from pymoo.operators.sampling.lhs import LHS

from pymoo.algorithms.soo.nonconvex.es import ES


problem = UnconditionalOptimization(
    order=(6, 7),
    theta_max_degrees=22,
    dx_wl=50,
    dz_wl=0.25
)

algorithm = DE(
    pop_size=50,
    dither="scalar",
    sampling=LHS(),
)

termination = get_termination("n_gen", 500)

performance = SingleObjectiveSingleRun(problem, termination=termination, seed=1)

res = minimize(HyperparameterProblem(algorithm, performance),
               Optuna(),
               termination=('n_evals', 10),
               seed=1,
               verbose=True)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

termination = get_termination("n_gen", 100000000)

res = minimize(problem, algorithm, termination=termination, seed=1, verbose=True)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))