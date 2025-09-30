__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'

import numpy as np
import cmath as cm
import math as fm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.fftpack import idct
from numpy.linalg import norm
import logging

""" Adaptive Filon–Clenshaw–Curtis (FCC) quadrature
for details of FCC see
Dominguez V., Graham I. G., Smyshlyaev V. P.
Stability and error estimates for Filon–Clenshaw–Curtis rules for highly oscillatory integrals //
IMA Journal of Numerical Analysis. – 2011. – Т. 31. – №. 4. – С. 1253-1280.
"""

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
        m = diags([-unos, d1, unos], [-1, 0, 1], format='csr')

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


def chebyshev_grid(a, b, n):
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
        if f.ndim == 1:
            f = f[:, np.newaxis]
        return np.tile(np.exp(1j*self.kn.real*(x_b+x_a)/2)[:, np.newaxis], (1, f.shape[1])) * (
            (self.fw * np.exp(-self.kn[:, np.newaxis].imag.dot(chebyshev_grid(x_a, x_b, self.x_n)[np.newaxis, :]))).dot(f))


class FCCAdaptiveFourier:
    """Adaptive Fourier transform by Filon–Clenshaw–Curtis quadrature
    """

    def __init__(self, domain_size, kn: np.array, x_n=15, rtol=1e-3):
        """
        :param domain_size: length of the integration domain (x_b - x_a)
        :param kn: array of spectral domain points
        :param x_n: number of Chebyshev points at one step adaptive integration step
        :param rtol: relative tolerance
        """
        self.domain_size = domain_size
        self.kn = kn
        self.x_n = x_n
        self.rtol = rtol
        self.fcc_integrators_dict = {1: FCCFourier(self.domain_size, self.x_n, self.kn)}

    def forward(self, f, x_a, x_b):
        assert abs((x_b - x_a) / self.domain_size - 1) < self.rtol, 'wrong integration domain size'
        i_val = self.fcc_integrators_dict[1].forward(np.array([f(a) for a in chebyshev_grid(x_a, x_b, self.x_n)]), x_a, x_b)
        return self._rec_forward(f, x_a, x_b, i_val)

    def transform(self, f, x_a, x_b):
        """
        compute the Fourier transformation of function f:
        \frac{1}{\sqrt{2\pi}}\int\limits_{x_{a}}^{x_{b}}f(x)e^{-ik_{x}x}dx
        :param f: np.array vector valued function
        :param x_a: left integration bound, (x_b - x_a) = domain_size
        :param x_b: right integration bound, (x_b - x_a) = domain_size
        :return:
        """
        return 1/cm.sqrt(2*cm.pi) * self.forward(f, x_a, x_b)

    def _rec_forward(self, f, x_a, x_b, i_val):
        logging.debug('FCCAdaptiveFourier [' + str(x_a) + '; ' + str(x_b) + ']')
        if (x_b - x_a) < 1e-14:
            return i_val
        x_c = (x_a + x_b) / 2
        index = round(self.domain_size / (x_c - x_a))
        if index not in self.fcc_integrators_dict:
            self.fcc_integrators_dict[index] = FCCFourier(x_c - x_a, self.x_n, self.kn)
        left_val = self.fcc_integrators_dict[index].forward(np.array([f(a) for a in chebyshev_grid(x_a, x_c, self.x_n)]), x_a, x_c)
        right_val = self.fcc_integrators_dict[index].forward(np.array([f(a) for a in chebyshev_grid(x_c, x_b, self.x_n)]), x_c, x_b)
        if norm(i_val - left_val - right_val) < self.rtol * norm(i_val):
            return left_val + right_val
        else:
            return self._rec_forward(f, x_a, x_c, left_val) + self._rec_forward(f, x_c, x_b, right_val)
