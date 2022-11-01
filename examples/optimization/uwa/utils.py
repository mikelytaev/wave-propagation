import cmath as cm
import math as fm
from itertools import zip_longest

import numpy as np

from examples.optimization.evol import opt_utils as opt_utils


def fit_func(coefs_arr, dx, order, xi_bounds):
    k0 = 2 * fm.pi
    num_coefs, den_coefs = opt_utils.opt_coefs_to_coefs_ga(coefs_arr, order)
    max_err = 0.0
    for xi in np.linspace(xi_bounds[0], xi_bounds[1], 20):
        p = 1.0
        for a, b in zip_longest(num_coefs, den_coefs, fillvalue=0.0j):
            p *= (1 + xi * a) / (1 + xi * b)

        ev = cm.exp(1j*k0*dx*(cm.sqrt(1+xi)-1))
        max_err = max(max_err, abs(p - ev))

    return max_err
