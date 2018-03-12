import logging
import math as fm
import pickle
from itertools import zip_longest

import mpmath
import pyximport
import scipy.linalg as la

from rwp.WPDefs import *
from rwp.environment import *
from transforms.fcc import FCCAdaptiveFourier

pyximport.install(setup_args={"include_dirs": np.get_include()})
from rwp.contfrac import *
from rwp.propagators import *

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class PadePropagator:

    def __init__(self, env: EMEnvironment, wavelength=1.0, pade_order=(1, 2), spe=False, dx_wl=100, dz_wl=1, tol=1e-11):
        self.env = env
        self.k0 = (2 * cm.pi) / wavelength
        self.wavelength = wavelength
        self.n_z = fm.ceil((self.env.z_max - self.env.z_min) / (dz_wl * wavelength)) + 1
        self.z_computational_grid, self.dz = np.linspace(self.env.z_min, self.env.z_max, self.n_z, retstep=True)
        self.dx = dx_wl * wavelength
        self.pade_order = pade_order
        self.spe = spe
        self.tol = tol

        mpmath.mp.dps = 63

        if self.env.N_profile is None and isinstance(self.env.lower_boundary, (ImpedanceBC, TransparentConstBS)) and \
                isinstance(self.env.upper_boundary, (ImpedanceBC, TransparentConstBS)):
            def diff2(s):
                return mpmath.acosh(1 + (self.k0 * self.dz) ** 2 * s / 2) ** 2 / (self.k0 * self.dz) ** 2
        else:
            def diff2(s):
                return s

        if spe:
            def sqrt_1plus(x):
                return 1 + x / 2
        else:
            def sqrt_1plus(x):
                return mpmath.mp.sqrt(1+x)

        def propagator_func(s):
            return mpmath.mp.exp(1j*self.k0*self.dx*(sqrt_1plus(diff2(s))-1))

        t = mpmath.taylor(propagator_func, 0, pade_order[0]+pade_order[1])
        p, q = mpmath.pade(t, pade_order[0], pade_order[1])
        self.pade_coefs = list(zip_longest([-1/complex(v) for v in mpmath.polyroots(p[::-1], maxsteps=200)],
                                           [-1/complex(v) for v in mpmath.polyroots(q[::-1], maxsteps=200)], fillvalue=0.0j))

    def _Crank_Nikolson_propagate(self, a, b, het, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        """
        Performs one Crank-Nikolson propagation step
        :param a: right-hand parameter
        :param b: left-hand parameter
        :param het: heterogeneity vector
        :param initial: initial value vector
        :param lower_bound: lower_bound[0]*u_0 + lower_bound[1]*u_1 = lower_bound[2]
        :param upper_bound: upper_bound[0]*u_{n-1} + upper_bound[1]*u_{n} = upper_bound[2]
        :return:
        """
        return np.array(Crank_Nikolson_propagator(self.k0 * self.dz, a, b, het, initial, lower_bound, upper_bound))
        # d_2 = 1/(self.k0*self.dz)**2 * diags([np.ones(self.n_z-1), -2*np.ones(self.n_z), np.ones(self.n_z-1)], [-1, 0, 1])
        # left_matrix = eye(self.n_z) + b*(d_2 + diags([het], [0]))
        # right_matrix = eye(self.n_z) + a*(d_2 + diags([het], [0]))
        # rhs = right_matrix * initial
        # left_matrix[0, 0], left_matrix[0, 1], rhs[0] = lower_bound
        # left_matrix[-1, -2], left_matrix[-1, -1], rhs[-1] = upper_bound
        # return spsolve(left_matrix, rhs)

    def _calc_nlbc(self, n_x, beta, diff_eq_solution):
        num_roots, den_roots = list(zip(*self.pade_coefs))
        m_size = len(self.pade_coefs)
        tau = 1.001
        if max(self.pade_order) == 1:
            def nlbc_transformed(t):
                return diff_eq_solution(self.k0 ** 2 * ((1 - t) / (-num_roots[0] + den_roots[0] * t) - beta) * self.dz ** 2)
        else:
            def nlbc_transformed(t):
                matrix_a = np.diag(den_roots, 0) - np.diag(num_roots[1:], -1)
                matrix_a[0, -1] = -num_roots[0]
                matrix_a[0, 0] *= t
                matrix_b = np.diag(-np.ones(m_size), 0) + np.diag(np.ones(m_size - 1), -1) + 0j
                matrix_b[0, -1] = 1
                matrix_b[0, 0] *= t
                matrix_b -= beta * matrix_a
                w, vr = la.eig(matrix_b, matrix_a, right=True)
                r = np.diag([diff_eq_solution(a * self.k0 ** 2 * self.dz**2) for a in w])
                res = vr.dot(r).dot(la.inv(vr))
                return res.reshape(m_size**2)

        fcca = FCCAdaptiveFourier(2 * fm.pi, -np.arange(0, n_x), rtol=self.tol)

        return (tau**np.repeat(np.arange(0, n_x)[:, np.newaxis], m_size ** 2, axis=1) / (2*fm.pi) *
                fcca.forward(lambda t: nlbc_transformed(tau * cm.exp(1j*t)), 0, 2*fm.pi)).reshape((n_x, m_size, m_size))

    def calc_lower_nlbc(self, n_x):
        beta = self.env.n2m1_profile(0, self.env.z_min)
        if isinstance(self.env.lower_boundary, TransparentConstBS):
            return self._calc_nlbc(n_x, beta, diff_eq_solution=lambda v: 1 / d2a_n_eq_ba_n(v))
        elif isinstance(self.env.lower_boundary, TransparentLinearBS):
            c = self.k0 ** 2 * self.env.lower_boundary.mu_n2m1 * self.dz ** 3
            return self._calc_nlbc(n_x, beta, lambda v: 1 / bessel_ratio(c=c, d=v, j=0, tol=self.tol))

    def calc_upper_nlbc(self, n_x):
        beta = self.env.n2m1_profile(0, self.env.z_max)
        if isinstance(self.env.upper_boundary, TransparentConstBS):
            return self._calc_nlbc(n_x, beta, diff_eq_solution=lambda v: 1 / d2a_n_eq_ba_n(v))
        elif isinstance(self.env.upper_boundary, TransparentLinearBS):
            c = -self.k0 ** 2 * self.env.upper_boundary.mu_n2m1 * self.dz ** 3
            return self._calc_nlbc(n_x, beta, lambda v: bessel_ratio(c=c, d=v, j=0, tol=self.tol))

    def propagate(self, initials: list, polarz, n_x, *, direction=1, lower_nlbc=np.array([]), upper_nlbc=np.array([]), n_dx_out=1, n_dz_out=1):
        x_computational_grid = np.arange(0, n_x) * self.dx
        field = Field(x_computational_grid[::n_dx_out], self.z_computational_grid[::n_dz_out], precision=self.tol)
        reflected = [np.empty(0)] * n_x
        if direction == 1:
            phi = initials[0]
        else:
            phi = self.z_computational_grid * 0j

        field.field[0, :] = phi[::n_dz_out]
        phi_0, phi_J = np.zeros((n_x, max(self.pade_order)), dtype=complex), np.zeros((n_x, max(self.pade_order)), dtype=complex)

        if isinstance(self.env.lower_boundary, TransparentBS) and len(lower_nlbc) < n_x:
            lower_nlbc = self.calc_lower_nlbc(n_x)
        if isinstance(self.env.upper_boundary, TransparentBS) and len(upper_nlbc) < n_x:
            upper_nlbc = self.calc_upper_nlbc(n_x)

        if direction == 1:
            iterator = enumerate(x_computational_grid[1:], start=1)
        else:
            iterator = enumerate(x_computational_grid[-2::-1], start=1)
            initials = initials[::-1]

        if isinstance(self.env.terrain, KnifeEdges):
            kn_dict = {}
            for i in range(0, len(self.env.terrain.edge_range)):
                x_i = int(round(self.env.terrain.edge_range[i] / self.dx))
                if direction != 1:
                    x_i = n_x - x_i - 1
                kn_dict[x_i] = int(round(self.env.terrain.edge_height[i] / self.dz))

        for x_i, x in iterator:
            het = self.env.n2m1_profile(x, self.z_computational_grid)
            terr_i = int(round(self.env.terrain(x - direction * self.dx / 2) / self.dz))
            terr_inext = int(round(self.env.terrain(x + direction * self.dx / 2) / self.dz))
            # process boundary conditions
            if isinstance(self.env.lower_boundary, TransparentBS):
                lower_convolution = np.einsum('ijk,ik->j', lower_nlbc[1:x_i], phi_0[x_i-1:0:-1])
            if isinstance(self.env.upper_boundary, TransparentBS):
                upper_convolution = np.einsum('ijk,ik->j', upper_nlbc[1:x_i], phi_J[x_i-1:0:-1])
            for pc_i, (a, b) in enumerate(self.pade_coefs):
                if isinstance(self.env.lower_boundary, TransparentBS):
                    lower_bound = -lower_nlbc[0, pc_i, pc_i], 1, lower_convolution[pc_i] + lower_nlbc[0, pc_i].dot(phi_0[x_i])
                elif isinstance(self.env.lower_boundary, ImpedanceBC):
                    alpha1, alpha2 = self.env.lower_boundary(self.wavelength, polarz)
                    gamma0 = 2 * (-alpha1 + alpha2 * self.dz) / (self.k0 * self.dz)**2 + alpha1*het[0]
                    gamma1 = 2 * alpha1 / (self.k0 * self.dz)**2 + alpha1*het[0]
                    lower_bound = alpha1 + b * gamma0, alpha1 + b * gamma1, \
                                  (alpha1 + a * gamma0) * phi[0] + (alpha1 + a * gamma1) * phi[1]

                if isinstance(self.env.upper_boundary, TransparentBS):
                    upper_bound = 1, -upper_nlbc[0, pc_i, pc_i], upper_convolution[pc_i] + upper_nlbc[0, pc_i].dot(phi_J[x_i])
                elif isinstance(self.env.upper_boundary, ImpedanceBC):
                    upper_bound = 0, 1, 0

                # process terrain and propagate
                phi = np.concatenate((np.zeros(terr_i),
                                      self._Crank_Nikolson_propagate(a, b, het[terr_i::], phi[terr_i::],
                                                                     lower_bound=lower_bound, upper_bound=upper_bound)))

                phi_0[x_i, pc_i], phi_J[x_i, pc_i] = phi[0], phi[-1]

            if isinstance(self.env.terrain, KnifeEdges):
                if x_i in kn_dict:
                    reflected[x_i] = np.copy(phi[0:kn_dict[x_i]])
                    phi[0:kn_dict[x_i]] = 0
                if initials[x_i].size > 0:
                    phi[0:kn_dict[x_i]] = -initials[x_i]
            else:
                if initials[x_i].size > 0:
                    phi[terr_inext:terr_i] = -initials[x_i]
                reflected[x_i] = phi[terr_i:terr_inext]

            if divmod(x_i, n_dx_out)[1] == 0:
                field.field[divmod(x_i, n_dx_out)[0], :] = phi[::n_dz_out]
                logging.debug('SSPade propagation x = ' + str(x))

        if direction == 1:
            return field, reflected
        else:
            field.field = field.field[::-1, :]
            return field, reflected[::-1]


class SSPadePropagationTask:

    def __init__(self, *, src: Source, env: EMEnvironment, two_way=False, max_range_m=100000, dx_wl=100, dz_wl=1,
                 n_dx_out=1, n_dz_out=1, pade_order=(1, 2), spe=False, nlbc_manager_path='nlbc', tol=1e-11):
        self.src = src
        self.env = env
        self.two_way = two_way
        self.max_range_m = max_range_m
        self.n_dx_out = n_dx_out
        self.n_dz_out = n_dz_out
        self.nlbc_manager_path = nlbc_manager_path
        self.propagator_fw = PadePropagator(env=self.env, wavelength=src.wavelength,
                                            pade_order=pade_order, spe=spe, dx_wl=dx_wl, dz_wl=dz_wl, tol=tol)

    def calculate(self):
        n_x = fm.ceil(self.max_range_m / self.propagator_fw.dx) + 1
        lower_nlbc_fw, upper_nlbc_fw = NLBCManager(self.nlbc_manager_path).get_NLBC(self.propagator_fw, n_x)
        initials_fw = [np.empty(0)] * n_x
        initials_fw[0] = np.array([self.src(a) for a in self.propagator_fw.z_computational_grid])
        field_fw, reflected_fw = self.propagator_fw.propagate(polarz=self.src.polarz, initials=initials_fw, n_x=n_x,
                                                              direction=1, lower_nlbc=lower_nlbc_fw,
                                                              upper_nlbc=upper_nlbc_fw, n_dx_out=self.n_dx_out,
                                                              n_dz_out=self.n_dz_out)

        if self.two_way:
            field_bw, reflected_bw = self.propagator_fw.propagate(polarz=self.src.polarz, initials=reflected_fw,
                                                                  direction=-1, n_x=n_x, lower_nlbc=lower_nlbc_fw,
                                                                  upper_nlbc=upper_nlbc_fw, n_dx_out=self.n_dx_out,
                                                                  n_dz_out=self.n_dz_out)
            field_fw.field += field_bw.field
        return field_fw


class NLBCManager:

    def __init__(self, name='nlbc'):
        self.file_name = name
        import os
        if os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.nlbs_dict = pickle.load(f)
        else:
            self.nlbs_dict = {}

    def get_NLBC(self, propagator: PadePropagator, n_x):
        lower_nlbc, upper_nlbc = None, None
        if isinstance(propagator.env.lower_boundary, TransparentBS):
            beta = propagator.env.n2m1_profile(0, propagator.env.z_min)
            q = 'lower', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, propagator.spe, beta, propagator.env.lower_boundary
            if q not in self.nlbs_dict or self.nlbs_dict[q].shape[0] < n_x:
                self.nlbs_dict[q] = propagator.calc_lower_nlbc(n_x)
            lower_nlbc = self.nlbs_dict[q]

        if isinstance(propagator.env.upper_boundary, TransparentBS):
            beta = propagator.env.n2m1_profile(0, propagator.env.z_max)
            q = 'upper', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, propagator.spe, beta, propagator.env.upper_boundary
            if q not in self.nlbs_dict or self.nlbs_dict[q].shape[0] < n_x:
                self.nlbs_dict[q] = propagator.calc_upper_nlbc(n_x)
            upper_nlbc = self.nlbs_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbs_dict, f)

        return lower_nlbc, upper_nlbc



def d2a_n_eq_ba_n(b):
    c1 = (b+2-cm.sqrt(b**2+4*b))/2
    c2 = 1.0 / c1
    return [c1, c2][abs(c1) > abs(c2)]


# def bessel_ratio(c, d, j, tol):
#     return lentz(lambda n: (-1)**(n+1) * 2.0 * (j + (2.0 + d) / c + n - 1) * (c / 2.0))


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