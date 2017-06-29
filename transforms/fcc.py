__author__ = 'Mikhail'

import numpy as np
import cmath as cm
import math as fm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# python implementation of Filon–Clenshaw–Curtis rules
# Dominguez V., Graham I. G., Smyshlyaev V. P.
# Stability and error estimates for Filon–Clenshaw–Curtis rules for highly oscillatory integrals
#  //IMA Journal of Numerical Analysis. – 2011. – Т. 31. – №. 4. – С. 1253-1280.


def chebyshev_weights_asymptotics(k: complex, n: int, tol=1e-16):
    sink = cm.sin(k)
    cosk = cm.cos(k)
    flag = 1

    coef = np.array([1,
                      k,
                      -3 * k**2,
                      (-15 * k**2 + 4 * n**2) * k,
                      (105 * k**2 - 60 * n**2) * k**2,
                      -(-945 * k**4 + 840 * k**2 * n**2 - 16 * n**4) * k,
                      -(-12600 * k**4 * n**2 + 1008 * k**2 * n**4 + 10395 * k**6),
                      (207900 * k**4 * n**2 - 35280 * k**2 * n**4 + 64 * n**6 - 135135 * k**6) * k])

    coef /= (2.0*n)**(2.0*np.arange(1, len(coef)+1)-1)
    coef[::2] = 2 * coef[::2]*sink
    coef[1::2] = 2 * coef[1::2] * cosk

    val = sum(coef[::-1])
    # Last coefficient is used as estimator of the error
    if abs(coef[-1]) > tol:
        flag = 0
    return val, flag


def chebyshev_weights(k: complex, n: int):
    rho = np.zeros((n + 1, 1))
    rho[0] = 2.0 / k * cm.sin(k)
    rho[1] = -4.0 * cm.cos(k) / k + 4.0 * cm.sin(k) / k**2

    # n0 coefficients are computed using the three term forward recurrence
    n0 = min(max(fm.ceil(abs(k)), 2), n)
    gamma = [-2 * cm.cos(k) / k, 2 * cm.sin(k) / k]
    for ji in range(3, n0+1):
        rho[ji-1] = 2 * gamma(ji % 2 + 1) + 2 * (-1) ^ ji * (ji - 1) / k * rho[ji - 2] + rho[ji - 3]

    if n0 < n:
        # the remainder terms, if required, are computed by solving a tridiagonal linear system
        flag = 0
        ji = 0
        nMax = max(fm.ceil(abs(k)), n)
        # Compute \rho(nMax) for nMax sufficiently large using an asymptotic formula
        while flag == 0 & ji < 5:
            nMax = fm.ceil(nMax) * 2
            rho_nMax_plus1, flag = chebyshev_weights_asymptotics(k, nMax / 2)
            ji = ji + 1

        # Assembly the tridiagonal matrix
        d1 = 2 * np.arange(n0 + 1, nMax + 2).T / k
        if n0 % 2 == 0:
            d1[::2] = -d1[::2]
        else:
            d1[1::2] = -d1[1::2]
        unos = np.ones((len(d1)-1, 1))
        m = diags([-unos, d1, unos], [-1, 0, 1])

        # Right hand side
        b = np.zeros((nMax - n0 + 1, 1))
        if n0 % 2 == 1:
            b[1::2] = 2 * gamma[0]
            b[0::2] = 2 * gamma[1]
        else:
            b[1::2] = 2 * gamma[1]
            b[0::2] = 2 * gamma[2]
        b[0] += rho[n0-1]
        b[-1] -= rho_nMax_plus1

        # Solving the linear system
        aux = spsolve(m, b)
        # save the values
        rho[n0:n] = aux[0:n-n0]

    # correct rho as a complex number the entries at even positions are pure imaginary
    rho[1::] = rho[1::2] * 1j
    w = np.zeros((n+1, 1));
    w[0] = gamma[1]
    w[1:] = np.multiarray(np.arange(1, n+1).T / (1j*k), rho[0:n+1])
    w[1::2] += gamma[0]*1j
    w[2::2] += gamma[1]

    return w, rho
