import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
from scipy.optimize import differential_evolution

from examples.optimization.uwa.utils import fit_func


dx = 10
order = (6, 7)
theta_max_degrees = 10
xi_bounds = (-0.1, 0)
bounds_ga = [(-10, 10)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=fit_func,
    args=(dx, order, xi_bounds),
    bounds=bounds_ga,
    popsize=50,
    disp=True,
    mutation=(0.5, 1),
    recombination=1.0,
    strategy='randtobest1exp',
    tol=1e-9,
    maxiter=2000,
    polish=False,
    workers=-1
)

print(result_ga)