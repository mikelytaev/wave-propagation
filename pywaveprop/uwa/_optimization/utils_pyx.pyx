import numpy as np
cimport numpy as np

from math import pi
from cmath import sqrt, exp


def tau_error(double k0, complex xi1, complex xi2, double dx_m, np.ndarray[complex, ndim=1] pade_coefs_num, np.ndarray[complex, ndim=1] pade_coefs_den, complex c0):
    cdef complex product = c0
    cdef complex a_i = 0
    cdef Py_ssize_t i
    for i in range(0, len(pade_coefs_den)):
        if i < len(pade_coefs_num):
            a_i = pade_coefs_num[i]
        else:
            a_i = 0
        product *= (1 + a_i * xi2) / (1 + pade_coefs_den[i] * xi2)

    cdef complex ex = exp(1j * k0 * dx_m * (sqrt(1.0 + xi1) - 1.0))
    return abs(ex - product).real
