import numpy as np

import pyximport

import examples.optimization.uwa.pade_opt.utils

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

import utils as utils

from examples.jcs.d2_error import *
import math as fm
from propagators._utils import *

pade_order = (8, 8)
k0 = 2 * fm.pi
dx_wl = 5
coefs, a0 = pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0, dx=dx_wl, a0=-0.3)
pade_coefs_num = np.array([a[0] for a in coefs])
pade_coefs_den = np.array([a[1] for a in coefs])

xi = -0.4
error = examples.optimization.uwa.pade_opt.utils.tau_error(xi, xi, dx_wl, pade_coefs_num, pade_coefs_den, a0)
