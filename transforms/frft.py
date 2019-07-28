"""
Implementation of the Fourier transform method from
Bailey D. H., Swarztrauber P. N. A fast method for the numerical evaluation of continuous Fourier and Laplace
transforms //SIAM Journal on Scientific Computing. – 1994. – Т. 15. – №. 5. – С. 1105-1110.
"""
import numpy as np
import cmath as cm

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


def get_fcft_grid(m, b):
    return np.arange(0, m) * b / m - b / 2


def frft(x, alpha):
    m = x.shape[-1]
    if len(x.shape) == 1:
        x = x.reshape(1, m)
    y = np.zeros((x.shape[0], 2*x.shape[1]), dtype=complex)
    y[:, 0:m] = x * np.exp(-cm.pi * 1j * np.arange(0, m) ** 2 * alpha)
    z = np.zeros((x.shape[0], 2*x.shape[1]), dtype=complex)
    z[:, 0:m] = np.exp(cm.pi * 1j * np.arange(0, m) ** 2 * alpha)
    z[:, m:2 * m] = np.exp(cm.pi * 1j * (np.arange(m, 2 * m) - 2 * m) ** 2 * alpha)
    w = np.fft.ifft((np.fft.fft(y) * np.fft.fft(z)))
    return np.exp(-cm.pi * 1j * np.arange(0, m) ** 2 * alpha) * w[:, 0:m]


def fcft(f_x, a, b):
    """
    computes discrete Fourier Transform for input points f_x
    1/\sqrt{2 \pi} \int\limits_{-a/2}^{a/2} f(t)\exp (-itx_{k})dt
    :param f_x: input function values in points np.linspace(-a/2, a/2, m)
    """
    m = f_x.shape[-1]
    delta = a * b / (2 * cm.pi * m ** 2)
    beta = a / m
    w = frft(np.exp(cm.pi * 1j * np.arange(0, m) * m * delta) * f_x, delta)
    return 1 / cm.sqrt(2 * cm.pi) * beta * np.exp(cm.pi * 1j * (np.arange(0, m) - m / 2) * m * delta) * w


def ifcft(f_x, b, a):
    """
    computes inverse discrete Fourier Transform for input points f_x
    """
    return fcft(f_x, b, -a)