from pymoo.algorithms.hyperparameters import SingleObjectiveSingleRun, HyperparameterProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

from problems import UnconditionalOptimization


problem = UnconditionalOptimization(
    order=(6, 7),
    theta_max_degrees=22,
    dx_wl=50,
    dz_wl=0.25
)

algorithm = DE()

n_evals = 500

performance = SingleObjectiveSingleRun(problem, termination=("n_evals", n_evals), seed=1)

res = minimize(HyperparameterProblem(algorithm, performance),
               Optuna(),
               termination=('n_evals', 50),
               seed=1,
               verbose=True)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

res = minimize(problem, algorithm, termination=("n_evals", n_evals), seed=1, verbose=True)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))