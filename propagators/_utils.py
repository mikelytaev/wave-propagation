import cmath
from itertools import zip_longest
from typing import List, Any
from scipy.special import ive, iv

import mpmath
import cmath as cm
import math as fm
import numpy as np


def pade_sqrt_coefs(n):
    n_arr = np.arange(1, n+1)
    a_n = 2 / (2*n + 1) * np.sin(n_arr * cm.pi / (2 * n + 1))**2
    b_n = np.cos(n_arr * cm.pi / (2 * n + 1)) ** 2
    return a_n, b_n


def pade_sqrt(z, a_n, b_n, alpha=0):
    alpha = alpha * cm.pi / 180
    return cm.exp(1j*alpha/2) * (1 + sum([a * ((1 + z) * cm.exp(-1j*alpha) - 1) / (1 + b * ((1 + z) * cm.exp(-1j*alpha) - 1)) for (a, b) in zip(a_n, b_n)]))


def pade_propagator_coefs(*, pade_order, diff2, k0, dx, spe=False, alpha=0, a0=0.0):
    """
    Pade approximation of the exponential propagator of the form \prod_{l=1}^{p}\frac{1+a_{l}L}{1+b_{l}L}
    :param pade_order: order of Pade approximation, tuple, for ex (7, 8)
    :param diff2:
    :param k0: wavenumber
    :param dx: longitudinal grid step (m)
    :param spe: use standard narrow angle parabolic equation (Schröder's equation)
    :param alpha: rotation angle, see F. A. Milinazzo et. al. Rational square-root approximations for parabolic equation algorithms. 1997. Acoustical Society of America.
    :return: [(a_1, b_1), (a_2, b_2),...(a_p, b_p)], if pade_order[0]!=pade_order[1] extra coefs. are filled with 0
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

    t = mpmath.taylor(propagator_func, a0, pade_order[0] + pade_order[1] + 2)
    p, q = mpmath.pade(t, pade_order[0], pade_order[1])
    p0 = p[-1]
    q0 = q[-1]
    p = [t / p0 for t in p]
    q = [t / q0 for t in q]
    try:
        p_roots = mpmath.polyroots(p[::-1], maxsteps=5000)
        q_roots = mpmath.polyroots(q[::-1], maxsteps=5000)
    except:
        return list(zip_longest([0j] * (len(p)-1), [0j] * (len(q)-1), fillvalue=0.0j)), 1.0+0j
    pade_coefs = list(zip_longest([-1 / complex(v) / (1 - a0 * (-1 / complex(v))) for v in p_roots],
                                       [-1 / complex(v) / (1 - a0 * (-1 / complex(v))) for v in q_roots],
                                       fillvalue=0.0j))
    c0 = p0 / q0
    for t in p_roots:
        c0 *= -t * (1 - a0 * (-1 / complex(t)))
    for t in q_roots:
        c0 /= -t * (1 - a0 * (-1 / complex(t)))
    return pade_coefs, complex(c0)


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


def k_x(k, kz):
    if abs(kz) < k:
        return cm.sqrt(k**2 - kz**2)
    else:
        return 1j * cm.sqrt(kz**2 - k**2)


def discrete_exp(k, dx, pade_coefs, dz, kz, order=2):
    if order == 2:
        d_2 = cm.sin(kz*dz/2)**2
    else:
        d_2 = cm.sin(kz*dz/2)**2 + 1/3*cm.sin(kz*dz/2)**4
    mult = 1
    for (a_i, b_i) in pade_coefs:
        mult *= (1-4*a_i/(k*dz)**2 * d_2) / (1-4*b_i/(k*dz)**2 * d_2)

    return cm.exp(1j*k*dx) * mult


def optimal_params_m(max_angle_deg, max_distance_wl, threshold, dx_wl=None, dz_wl=None, pade_order=None, z_order=4):
    k0 = 2*cm.pi
    res = (None, None, None)
    cur_min = 1e100

    if pade_order:
        pade_orders = [pade_order]
    else:
        pade_orders = [(7, 8), (6, 7), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (1, 1)]

    if dx_wl:
        dxs = [dx_wl]
    else:
        dxs = np.concatenate((#np.linspace(0.001, 0.01, 10),
                              #np.linspace(0.02, 0.1, 9),
                              #np.linspace(0.2, 1, 9),
                              #np.linspace(2, 10, 9),
                              np.linspace(20, 100, 9),
                              np.linspace(200, 1000, 9),
                              np.linspace(1100, 1900, 9),
                              np.linspace(2000, 10000, 9)))

    if dz_wl:
        dzs = [dz_wl]
    else:
        dzs = np.concatenate((np.array([0.001, 0.009]),
                              np.array([0.01, 0.09]),
                             np.linspace(0.1, 9, 90)))

    dxs.sort()
    dzs.sort()
    for pade_order in pade_orders:
        for dx_wl in dxs:
            updated = False
            if z_order <= 4:
                coefs, c0 = pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0, dx=dx_wl, spe=False)
            for dz_wl in dzs:
                if z_order > 4:
                    coefs, c0 = pade_propagator_coefs(pade_order=pade_order,
                                                  diff2=lambda s: mpmath.acosh(1 + (k0 * dz_wl) ** 2 * s / 2) ** 2 / (k0 * dz_wl) ** 2,
                                                  k0=k0, dx=dx_wl, spe=False)

                errors = []
                for al in np.linspace(0, max_angle_deg, 20):
                    kz = k0 * cm.sin(al * cm.pi / 180)
                    if z_order <= 4:
                        discrete_kx = discrete_k_x(k0, dx_wl, coefs, dz_wl, kz, order=z_order)
                    else:
                        discrete_kx = discrete_k_x(k0, dx_wl, coefs, dz_wl, kz, order=2)
                    real_kx = cm.sqrt(k0 ** 2 - kz ** 2)
                    errors += [abs(real_kx - discrete_kx) / k0]

                val = pade_order[1] / (dx_wl * dz_wl)
                error = max(errors)

                if error >= threshold * dx_wl / max_distance_wl:
                    break

                if error < threshold * dx_wl / max_distance_wl and val < cur_min:
                    res = (dx_wl, dz_wl, pade_order)
                    cur_min = val
                    updated = True

            if not updated:
                break

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
        return -(cm.sqrt(eps_r - cm.sin(theta) ** 2) - eps_r * cm.cos(theta)) / (cm.sqrt(eps_r - cm.sin(theta) ** 2) + eps_r * cm.cos(theta))


def brewster_angle(eps1, eps2):
    """
    :param eps1: permittivity in medium 1
    :param eps2: permittivity in medium 2
    :return: brewster angle between incident wave and normal to the surface in degrees
    """
    return 90 - cm.asin(1 / cm.sqrt(eps2 / eps1 + 1)) * 180 / cm.pi


def Miller_Brown_factor(theta, k0, rms_m):
    gamma = 2 * k0 * rms_m * cm.sin(theta * cm.pi / 180)
    #rho = i0e(0.5 * abs(gamma)**2)
    #print("gamma = " + str(gamma))
    #rho = cm.exp(-0.5 * gamma**2)
    #rho = 1 + (-0.5 * gamma**2) + (-0.5 * gamma**2)**2
    arg = -0.5 * gamma**2
    rho = ive(0, -arg)
    #rho = (1 + (1/2)*arg + (1/9)*arg**2 + (1/72)*arg**3) / (1 - (1/2)*arg + (1/9)*arg**2 - (1/72)*arg**3)
    #print("theta = " + str(theta) + " sin(theta) = " + str(cm.sin(theta * cm.pi / 180)) + " rho = " + str(rho))
    return rho


class MillerBrownFactor:

    def __init__(self, n):
        self.n = n
        mpmath.mp.dps = 100

        def func(x):
            return mpmath.exp(-x) * mpmath.besseli(0, x)

        t = mpmath.taylor(func, 0, 2*n+1)
        self.p, self.q = mpmath.pade(t, n, n)
        # self.pade_coefs = list(zip_longest([-1 / complex(v) for v in mpmath.polyroots(p[::-1], maxsteps=2000)],
        #                               [-1 / complex(v) for v in mpmath.polyroots(q[::-1], maxsteps=2000)],
        #                               fillvalue=0.0j))
        #self.pade_roots_num = [complex(v) for v in mpmath.polyroots(self.p[::-1], maxsteps=5000)]
        #self.pade_roots_den = [complex(v) for v in mpmath.polyroots(self.q[::-1], maxsteps=5000)]
        self.pade_coefs_num = [complex(v) for v in self.p]
        self.pade_coefs_den = [complex(v) for v in self.q]
        self.taylor_coefs = [complex(v) for v in t]

        a = [self.q[-1]] + [b + c for b, c in zip(self.q[:-1:], self.p)]
        self.a_roots = [complex(v) for v in mpmath.polyroots(a[::-1], maxsteps=5000)]

    def roots(self, r):
        a = [r*p + q for p, q in zip(self.p, self.q)]
        rootss = [v for v in mpmath.polyroots(a[::-1], maxsteps=5000)]
        #b =  mpmath.polyval(self.p[::-1], rootss[0]) + mpmath.polyval(self.q[::-1], rootss[0])
        return [complex(v) for v in rootss]

    def factor(self, theta, k0, rms_m, k_z2=None):
        #theta = 0.2
        gamma = 2 * k0 * rms_m * cm.sin(theta * cm.pi / 180)
        arg = 0.5 * gamma ** 2
        if k_z2:
            arg = complex(2 * rms_m**2 * k_z2)
        #res = 1
        # for (a, b) in self.pade_coefs:
        #     res *= (1 + a * arg) / (1 + b * arg)
        # num = np.prod([1 + a * arg for a in self.pade_coefs_num])
        # den = np.prod([1 + b * arg for b in self.pade_coefs_den])

        #return np.polyval(self.taylor_coefs[::-1], arg)
        # if arg.real < 0:
        #     arg = -arg.real + 1j*arg.imag
        #res = cm.exp(-arg)

        num = np.polyval(self.pade_coefs_num[::-1], arg)
        den = np.polyval(self.pade_coefs_den[::-1], arg)
        res = num / den
        return complex(res)
        #return cm.exp(-abs(arg))
        #return ive(0, arg) * cm.exp(-1j * arg.imag)
        #return (cm.atan(-cm.log10(-arg + 1e-10) * 3) / (cm.pi / 2) + 1) / 2


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


def d_k_x(*, k_z, dx, dz, pade_order, z_order, alpha=0):
    k0 = 2 * cm.pi
    if z_order > 4:
        z_order = 2
        diff2 = lambda s: mpmath.acosh(1 + (k0 * dz) ** 2 * s / 2) ** 2 / (k0 * dz) ** 2
    else:
        diff2 = lambda s: s

    if hasattr(k_z, "__len__") and not hasattr(dx, "__len__") and not hasattr(dz, "__len__"):
        coefs = pade_propagator_coefs(pade_order=pade_order, diff2=diff2, k0=k0, dx=dx, alpha=alpha)
        return np.array([discrete_k_x(k=k0, dx=dx, dz=dz, pade_coefs=coefs, kz=kz, order=z_order) for kz in k_z])

    if not hasattr(k_z, "__len__") and hasattr(dx, "__len__") and not hasattr(dz, "__len__"):
        res = []
        for dx_val in dx:
            coefs = pade_propagator_coefs(pade_order=pade_order, diff2=diff2, k0=k0, dx=dx_val, alpha=alpha)
            res += [discrete_k_x(k=k0, dx=dx_val, dz=dz, pade_coefs=coefs, kz=k_z, order=z_order)]
        return np.array(res)

    if not hasattr(k_z, "__len__") and not hasattr(dx, "__len__") and hasattr(dz, "__len__"):
        coefs = pade_propagator_coefs(pade_order=pade_order, diff2=diff2, k0=k0, dx=dx, alpha=alpha)
        return np.array([discrete_k_x(k=k0, dx=dx, dz=v, pade_coefs=coefs, kz=k_z, order=z_order) for v in dz])