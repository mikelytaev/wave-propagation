import cmath
from itertools import zip_longest
from typing import List, Any

import mpmath
import cmath as cm
import numpy as np


def pade_sqrt_coefs(n):
    n_arr = np.arange(1, n+1)
    a_n = 2 / (2*n + 1) * np.sin(n_arr * cm.pi / (2 * n + 1))**2
    b_n = np.cos(n_arr * cm.pi / (2 * n + 1)) ** 2
    return a_n, b_n


def pade_sqrt(z, a_n, b_n, alpha=0):
    alpha = alpha * cm.pi / 180
    return cm.exp(1j*alpha/2) * (1 + sum([a * ((1 + z) * cm.exp(-1j*alpha) - 1) / (1 + b * ((1 + z) * cm.exp(-1j*alpha) - 1)) for (a, b) in zip(a_n, b_n)]))


def pade_propagator_coefs(*, pade_order, diff2, k0, dx, spe=False, alpha=0):
    """

    :param pade_order: order of Pade approximation, tuple, for ex (7, 8)
    :param diff2:
    :param k0:
    :param dx:
    :param spe:
    :param alpha: rotation angle, see F. A. Milinazzo et. al. Rational square-root approximations for parabolic equation algorithms. 1997. Acoustical Society of America.
    :return:
    """

    mpmath.mp.dps = 63
    if spe:
        def sqrt_1plus(x):
            return 1 + x / 2
    elif alpha == 0:
        def sqrt_1plus(x):
            return mpmath.mp.sqrt(1 + x)
    else:
        a_n, b_n = pade_sqrt_coefs(pade_order[1])

        def sqrt_1plus(x):
            return pade_sqrt(x, a_n, b_n, alpha)

    def propagator_func(s):
        return mpmath.mp.exp(1j * k0 * dx * (sqrt_1plus(diff2(s)) - 1))

    t = mpmath.taylor(propagator_func, 0, pade_order[0] + pade_order[1] + 20)
    p, q = mpmath.pade(t, pade_order[0], pade_order[1])
    pade_coefs = list(zip_longest([-1 / complex(v) for v in mpmath.polyroots(p[::-1], maxsteps=2000)],
                                       [-1 / complex(v) for v in mpmath.polyroots(q[::-1], maxsteps=2000)],
                                       fillvalue=0.0j))
    return pade_coefs


def discrete_k_x(k, dx, pade_coefs, dz, kz, order=2):
    if order == 2:
        d_2 = cm.sin(kz * dz / 2) ** 2
    else:
        d_2 = cm.sin(kz * dz / 2) ** 2 + 1 / 3 * cm.sin(kz * dz / 2) ** 4
    sum = 0
    for (a_i, b_i) in pade_coefs:
        sum += cm.log((1 - 4 * a_i / (k * dz) ** 2 * d_2)) - cm.log((1 - 4 * b_i / (k * dz) ** 2 * d_2))

    return k - 1j / dx * sum


def discrete_k_x2(k, dx, pade_coefs, dz, kz, order=2):
    if order == 2:
        d_2 = cm.sin(kz * dz / 2) ** 2
    else:
        d_2 = cm.sin(kz * dz / 2) ** 2 + 1 / 3 * cm.sin(kz * dz / 2) ** 4
    mult = 1
    for (a_i, b_i) in pade_coefs:
        mult *= (1 - 4 * a_i / (k * dz) ** 2 * d_2) / (1 - 4 * b_i / (k * dz) ** 2 * d_2)

    return k - 1j / dx * cm.log(mult)


def discrete_exp(k, dx, pade_coefs, dz, kz, order=2):
    if order == 2:
        d_2 = cm.sin(kz*dz/2)**2
    else:
        d_2 = cm.sin(kz*dz/2)**2 + 1/3*cm.sin(kz*dz/2)**4
    mult = 1
    for (a_i, b_i) in pade_coefs:
        mult *= (1-4*a_i/(k*dz)**2 * d_2) / (1-4*b_i/(k*dz)**2 * d_2)

    return cm.exp(1j*k*dx) * mult


def optimal_params(max_angle, threshold, dx=None, dz=None, pade_order=None, z_order=4):
    k0 = 2*cm.pi
    res = (None, None, None)
    cur_min = 1e100

    if pade_order:
        pade_orders = [pade_order]
    else:
        pade_orders = [(7, 8), (6, 7), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (1, 1)]

    if dx:
        dxs = [dx]
    else:
        dxs = np.concatenate((np.linspace(0.1, 1, 10), np.linspace(2, 10, 9), np.linspace(20, 100, 9), np.linspace(200, 1000, 9), np.linspace(2000, 10000, 9)))

    if dz:
        dzs = [dz]
    else:
        dzs = np.concatenate((np.array([0.01, 0.05]), np.linspace(0.1, 2, 20)))

    dxs.sort()
    dzs.sort()
    for pade_order in pade_orders:
        for dx in dxs:
            if z_order <= 4:
                coefs = pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0, dx=dx, spe=False)
            for dz in dzs:
                if z_order > 4:
                    coefs = pade_propagator_coefs(pade_order=pade_order, diff2=lambda s: mpmath.acosh(1 + (k0 * dz) ** 2 * s / 2) ** 2 / (k0 * dz) ** 2, k0=k0, dx=dx, spe=False)
                error = 0
                for al in np.linspace(0, max_angle, 5):
                    kz = k0 * cm.sin(al * cm.pi / 180)
                    if z_order <= 4:
                        d_exp = discrete_exp(k0, dx, coefs, dz, kz, order=z_order)
                    else:
                        d_exp = discrete_exp(k0, dx, coefs, dz, kz, order=2)
                    error += abs(cm.phase(d_exp) - cm.phase(
                        cm.exp(1j * cm.sqrt(k0 ** 2 - kz ** 2) * dx)))
                error /= 5
                val = pade_order[1] / (dx * dz)
                if error < threshold and val < cur_min:
                    res = (dx, dz, pade_order)
                    cur_min = val

    return res


def reflection_coef(eps1, eps2, theta, polarz):
    """
    :param eps1: permittivity in medium 1
    :param eps2: permittivity in medium 2
    :param theta: angle between incident wave and normal to surface in degrees
    :param polarz: polarization of the wave
    :return: reflection coefficient
    """
    eps_r = eps2 / eps1
    theta = theta * cm.pi / 180
    if polarz.upper() == 'H':
        return (cm.cos(theta) - cm.sqrt(eps_r - cm.sin(theta)**2)) / (cm.cos(theta) + cm.sqrt(eps_r - cm.sin(theta)**2))
    else:
        return (cm.sqrt(eps_r - cm.sin(theta) ** 2) - eps_r * cm.cos(theta)) / (cm.sqrt(eps_r - cm.sin(theta) ** 2) + eps_r * cm.cos(theta))


def brewster_angle(eps1, eps2):
    """
    :param eps1: permittivity in medium 1
    :param eps2: permittivity in medium 2
    :return: brewster angle between incident wave and normal to the surface in degrees
    """
    return 90 - cm.asin(1 / cm.sqrt(eps2 / eps1 + 1)) * 180 / cm.pi


def sqr_eq(a, b, c):
    c1 = (-b + cm.sqrt(b**2 - 4 * a * c)) / (2 * a)
    c2 = (-b - cm.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return c2 if abs(c1) > abs(c2) else c1


def lentz(cont_frac_seq, tol=1e-20):
    """
    Lentz W. J. Generating Bessel functions in Mie scattering calculations using continued fractions
    //Applied Optics. – 1976. – 15. – №. 3. – P. 668-671.
    :param cont_frac_seq: continued fraction sequence
    :param tol: absolute tolerance
    """
    num = cont_frac_seq(2) + 1.0 / cont_frac_seq(1)
    den = cont_frac_seq(2)
    y = cont_frac_seq(1) * num / den
    i = 3
    while abs(num / den - 1) > tol:
        num = cont_frac_seq(i) + 1.0 / num
        den = cont_frac_seq(i) + 1.0 / den
        y = y * num / den
        i += 1

    return y


def d2a_n_eq_ba_n(b):
    c1 = (b+2-cm.sqrt(b**2+4*b))/2
    c2 = 1.0 / c1
    return [c1, c2][abs(c1) > abs(c2)]