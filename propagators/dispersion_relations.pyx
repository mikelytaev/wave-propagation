import numpy as np
cimport numpy as np

from math import sin, pi
from cmath import log, sqrt



__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


def discrete_k_x(double k, double dx, np.ndarray[complex, ndim=1] pade_coefs_num, np.ndarray[complex, ndim=1] pade_coefs_den, double theta_degrees):
    cdef double sin_theta = sin(theta_degrees * pi / 180)**2
    cdef complex sum = 0
    cdef complex a_i = 0
    cdef Py_ssize_t i
    for i in range(0, len(pade_coefs_den)):
        if i < len(pade_coefs_num):
            a_i = pade_coefs_num[i]
        else:
            a_i = 0
        sum += log(1 - a_i * sin_theta) - log(1 - pade_coefs_den[i] * sin_theta)

    return k - 1j / dx * sum


def discrete_k_x(double k, double dx, double dz, np.ndarray[complex, ndim=1] pade_coefs_num, np.ndarray[complex, ndim=1] pade_coefs_den, double theta_degrees):
    cdef double v = (2 / (k*dz) * sin(k * sin(theta_degrees * pi / 180) * dz / 2))**2
    cdef complex sum = 0
    cdef complex a_i = 0
    cdef Py_ssize_t i
    for i in range(0, len(pade_coefs_den)):
        if i < len(pade_coefs_num):
            a_i = pade_coefs_num[i]
        else:
            a_i = 0
        sum += log(1 - a_i * v) - log(1 - pade_coefs_den[i] * v)

    return k - 1j / dx * sum


def k_x(double k, double kz):
    if abs(kz) < k:
        return sqrt(k**2 - kz**2)
    else:
        return 1j * sqrt(kz**2 - k**2)


def k_x_abs_error_point(double k0, double dx, np.ndarray[complex, ndim=1] pade_coefs_num, np.ndarray[complex, ndim=1] pade_coefs_den, double theta_degrees):
    cdef double kz = k0 * sin(theta_degrees * pi / 180)
    dk_x = discrete_k_x(k0, dx, pade_coefs_num, pade_coefs_den, theta_degrees)
    return abs(dk_x - k_x(k0, kz)).real


def k_x_abs_error_point(double k0, double dx, double dz, np.ndarray[complex, ndim=1] pade_coefs_num, np.ndarray[complex, ndim=1] pade_coefs_den, double theta_degrees):
    cdef double kz = k0 * sin(theta_degrees * pi / 180)
    dk_x = discrete_k_x(k0, dx, dz, pade_coefs_num, pade_coefs_den, theta_degrees)
    return abs(dk_x - k_x(k0, kz)).real


def k_x_abs_error_range(double k0, double dx, np.ndarray[complex, ndim=1] pade_coefs_num, np.ndarray[complex, ndim=1] pade_coefs_den, double theta_max_degrees, int iters):
    cdef double error = 0
    cdef Py_ssize_t i
    cdef double theta_degrees
    cdef double t
    for i in range(0, iters):
        theta_degrees = theta_max_degrees * i / (iters - 1)
        t = k_x_abs_error_point(k0, dx, pade_coefs_num, pade_coefs_den, theta_degrees)
        if t > error:
            error = t
    return error


def k_x_abs_error_range(double k0, double dx, double dz, np.ndarray[complex, ndim=1] pade_coefs_num, np.ndarray[complex, ndim=1] pade_coefs_den, double theta_max_degrees, int iters):
    cdef double error = 0
    cdef Py_ssize_t i
    cdef double theta_degrees
    cdef double t
    for i in range(0, iters):
        theta_degrees = theta_max_degrees * i / (iters - 1)
        t = k_x_abs_error_point(k0, dx, dz, pade_coefs_num, pade_coefs_den, theta_degrees)
        if t > error:
            error = t
    return error
