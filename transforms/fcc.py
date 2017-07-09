__author__ = 'Mikhail'

import numpy as np
import cmath as cm
import math as fm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.fftpack import idct
from numpy.linalg import norm
import logging

# python implementation of Filon–Clenshaw–Curtis rules
# Dominguez V., Graham I. G., Smyshlyaev V. P.
# Stability and error estimates for Filon–Clenshaw–Curtis rules for highly oscillatory integrals
#  //IMA Journal of Numerical Analysis. – 2011. – Т. 31. – №. 4. – С. 1253-1280.


def chebyshev_weights_asymptotics(k: complex, n: int, tol=1e-16):
    sink = cm.sin(k)
    cosk = cm.cos(k)
    flag = 1

    coef = np.array([1+0j,
                      k,
                      -3 * k**2,
                      (-15 * k**2 + 4 * n**2) * k,
                      (105 * k**2 - 60 * n**2) * k**2,
                      -(-945 * k**4 + 840 * k**2 * n**2 - 16 * n**4) * k,
                      -(-12600 * k**4 * n**2 + 1008 * k**2 * n**4 + 10395 * k**6),
                      (207900 * k**4 * n**2 - 35280 * k**2 * n**4 + 64 * n**6 - 135135 * k**6) * k])

    coef /= ((2.0+0j)*n)**(2.0*np.arange(1, len(coef)+1)-1)
    coef[::2] = 2 * coef[::2]*sink
    coef[1::2] = 2 * coef[1::2] * cosk

    val = sum(coef[::-1])
    # Last coefficient is used as estimator of the error
    if abs(coef[-1]) > tol:
        flag = 0
    return val, flag


def chebyshev_weights(k: complex, n: int):
    rho = np.zeros(n + 1)*0j
    rho[0] = 2.0 / k * cm.sin(k)
    rho[1] = -4.0 * cm.cos(k) / k + 4.0 * cm.sin(k) / k**2

    # n0 coefficients are computed using the three term forward recurrence
    n0 = min(max(fm.ceil(abs(k)), 2), n)
    gamma = [-2 * cm.cos(k) / k, 2 * cm.sin(k) / k]
    for ji in range(3, n0+1):
        rho[ji-1] = 2 * gamma[ji % 2] + 2 * (-1) ** ji * (ji - 1) / k * rho[ji - 2] + rho[ji - 3]

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
        d1 = (2+0j) * np.arange(n0 + 1, nMax + 2).T / k
        if n0 % 2 == 0:
            d1[::2] = -d1[::2]
        else:
            d1[1::2] = -d1[1::2]
        unos = np.ones(len(d1)-1)
        m = diags([-unos, d1, unos], [-1, 0, 1])

        # Right hand side
        b = np.zeros(nMax - n0 + 1)*1j
        if n0 % 2 == 1:
            b[1::2] = 2 * gamma[0]
            b[0::2] = 2 * gamma[1]
        else:
            b[1::2] = 2 * gamma[1]
            b[0::2] = 2 * gamma[0]
        b[0] += rho[n0-1]
        b[-1] -= rho_nMax_plus1

        # Solving the linear system
        aux = spsolve(m, b)
        # save the values
        rho[n0:n] = aux[0:n-n0]

    # correct rho as a complex number the entries at even positions are pure imaginary
    rho[1::2] = rho[1::2] * 1j
    w = np.zeros(n+1)*0j
    w[0] = gamma[1]
    w[1:] = -np.arange(1, n+1) / (1j*k) * rho[0:n]
    w[1::2] += gamma[0]*1j
    w[2::2] += gamma[1]

    return w, rho


def cheb_grid(a, b, n):
    return a + (b - a) * (np.cos(np.linspace(0, fm.pi, n + 1)) + 1) / 2


class FCCFourier:

    def __init__(self, domain_size, x_n, kn: np.array):
        self.x_n = x_n
        self.kn = -kn
        self.fw = np.zeros((len(kn), x_n + 1))*0j
        for k_i in range(1, len(kn)+1):
            k = -kn[k_i-1].real
            m = x_n
            knew = k * domain_size / 2
            if abs(knew) < 1:
                if m % 2 == 0:
                    m_end = m + 1
                else:
                    m_end = m
                w = np.zeros(m+1)
                w[0] = 2
                w[2:m_end:2] = 2.0 / (1 - (np.arange(2, m_end+1, 2)**2))
                xi = np.cos(np.arange(0, m+1) * fm.pi / m)
                w = idct(w, type=1) / (len(w) - 1)
                # Correction for the first & last term
                w[[0, -1]] = 0.5 * w[[0, -1]]
                self.fw[k_i - 1, :] = w * np.exp(1j * knew * xi) * domain_size / 2
            else:
                w, rho = chebyshev_weights(knew, m)
                w = idct(w, type=1) / (len(w) - 1)
                # Correction for the first & last term
                w[[0, -1]] = 0.5 * w[[0, -1]]
                self.fw[k_i - 1, :] = domain_size / 2 * w

    def forward(self, f: np.array, x_a, x_b):
        return np.exp(1j*self.kn.real*(x_b+x_a)/2) * (
            (self.fw * np.exp(np.array([-self.kn.imag]).T.dot(np.array([cheb_grid(x_a, x_b, self.x_n)])))).dot(f))


class FCCAdaptiveFourier:

    def __init__(self, domain_size, kn: np.array, x_n=15, rtol=1e-3):
        self.domain_size = domain_size
        self.kn = kn
        self.x_n = x_n
        self.rtol = rtol
        self.fcc_integrators_dict = {}
        self.fcc_integrators_dict[1] = FCCFourier(self.domain_size, self.x_n, self.kn)

    def forward(self, f, x_a, x_b):
        vect_f = np.vectorize(f)
        i_val = self.fcc_integrators_dict[1].forward(vect_f(cheb_grid(x_a, x_b, self.x_n)), x_a, x_b)
        return self._rec_forward(vect_f, x_a, x_b, i_val)

    def _rec_forward(self, vect_f, x_a, x_b, i_val):
        logging.debug('FCCAdaptiveFourier [' + str(x_a) + '; ' + str(x_b) + ']')
        if (x_b - x_a) < 1e-14:
            return i_val
        x_c = (x_a + x_b) / 2
        index = round(self.domain_size / (x_c - x_a))
        if not (index in self.fcc_integrators_dict):
            self.fcc_integrators_dict[index] = FCCFourier(x_c - x_a, self.x_n, self.kn)
        i1_val = self.fcc_integrators_dict[index].forward(vect_f(cheb_grid(x_a, x_c, self.x_n)), x_a, x_c)
        i2_val = self.fcc_integrators_dict[index].forward(vect_f(cheb_grid(x_c, x_b, self.x_n)), x_c, x_b)
        if norm(i_val - i1_val - i2_val) / norm(i_val) < self.rtol:
            return i1_val + i2_val
        else:
            return self._rec_forward(vect_f, x_a, x_c, i1_val) + self._rec_forward(vect_f, x_c, x_b, i2_val)
