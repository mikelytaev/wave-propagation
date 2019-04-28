import logging

import math as fm
import numpy as np
from copy import deepcopy

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from rwp.field import Field
from rwp.environment import *
from rwp.antennas import Source


class CrankNicolsonPropagator:

    def __init__(self, env: Troposphere, src: Source, type='greene', dx_wl=100, dz_wl=1, tol=1e-11):
        self.env = env
        self.wavelength = src.wavelength
        self.freq_hz = 3e8 / self.wavelength
        self.k0 = (2 * cm.pi) / self.wavelength
        self.src = src
        self.window_height = self.env.z_max
        self.n_z = fm.ceil((self.env.z_max + self.window_height) / (dz_wl * self.wavelength)) + 1
        self.z_computational_grid, self.dz = np.linspace(0, self.env.z_max + self.window_height,
                                                         self.n_z, retstep=True)
        self.dx = dx_wl * self.wavelength
        self.tol = tol
        if type.lower() == 'greene':
            self.chi1, self.chi2, self.chi3, self.chi4 = 0.99987, 0.79624, 1.0, 0.30102
        elif type.lower() == 'claerbout':
            self.chi1, self.chi2, self.chi3, self.chi4 = 1.0, 0.75, 1.0, 0.25
        else:
            self.chi1, self.chi2, self.chi3, self.chi4 = 1.0, 0.5, 1.0, 0.0

        self.window = np.ones(self.n_z)
        self.window[np.nonzero(self.z_computational_grid > self.env.z_max)] = \
            np.cos((self.z_computational_grid[np.nonzero(self.z_computational_grid > self.env.z_max)] - self.env.z_max) /
                   self.window_height * cm.pi / 2)

    def _c(self, dx):
        return 1 - 1j*self.k0 * dx * (self.chi2 - self.chi4) / (2 * self.chi4)

    def _a(self, j: np.ndarray, m):
        x = m * self.dx
        n1 = self.env.n2m1_profile(m * self.dx, j * self.dz + self.env.terrain(x), self.freq_hz)
        n2 = self.env.n2m1_profile((m - 1) * self.dx, j * self.dz + self.env.terrain(x), self.freq_hz)
        return self.k0**2 * self.dz**2 * (n1 + n2) / 2

    def _D(self, j: np.ndarray, m, dx):
        return self._c(dx) * (self._a(j, m) - 2 + 2*self.chi3 * self.k0**2 * self.dz**2 /
                              (2*self.chi4 - 1j*self.k0*dx*(self.chi2 - self.chi4)))

    def _S(self, m):
        x = m * self.dx
        return (self.env.terrain(x + self.dx / 2) - self.env.terrain(x - self.dx / 2)) / self.dx

    def _d(self, m, dx):
        return -12 * (self._c(dx) - 5j*self.k0 * self._S(m)**2 * dx) - self.dz**2 *\
               (24*self.k0**2*self._S(m)**2 + ((self.chi1 - self.chi3) - (self.chi2 - self.chi4)*self._S(m)**2)
                *12*dx*1j*self.k0**3/self.chi4)

    def _R(self, j: np.ndarray, m, dx, sign):
        return 32 * (self._c(dx) - 1j * self.k0 * self._S(m)**2 * dx) + sign * self._S(m) * \
               ((12 - 8*self._a(j, m))*dx/self.dz + 32*1j*self.k0*self.dz -
                (2*(self.chi2 - self.chi4)-self.chi3 + self.chi4*self._S(m)**2)*8*self.k0**2*dx*self.dz/self.chi4)

    def _T(self, j: np.ndarray, m, dx, sign):
        return -2*(self._c(dx) - 1j * self.k0 * self._S(m)**2*dx) + sign * self._S(m) * \
               ((self._a(j, m)-6)*dx/self.dz - 4*1j*self.k0*self.dz -
                (2*(self.chi2 - self.chi4)-self.chi3+self.chi4*self._S(m)**2)*self.k0**2*dx*self.dz/self.chi4)

    def _eta(self, m):
        eps = self.env.ground_material.complex_permittivity(self.freq_hz)
        if self.src.polarz == 'H':
            return cm.sqrt(eps - 1)
        else:
            return cm.sqrt(eps - 1) / eps

    def _xi(self, m):
        return 1j * self.k0 * self._eta(m) * self.dz

    def _phi(self, j: np.ndarray, m, dx):
        return self._T(j, m, dx, +1)

    def _varphi(self, j: np.ndarray, m, dx):
        res = self._R(j, m, dx, +1)
        ind_1 = np.nonzero(j == 1)
        res[ind_1] += -(3 * self._R(j[ind_1], m, dx, -1) + (6 * self._xi(m) + self._xi(m)**3) * self._T(j[ind_1], m, dx, -1)) \
            / (45 - 42 * self._xi(m) + 18 * self._xi(m) ** 2 - 4 * self._xi(m)**3)
        return res

    def _alpha(self, j: np.ndarray, m, dx):
        res = 24 * self._D(j, m, dx) + self._d(m, dx)
        ind_1 = np.nonzero(j == 1)
        res[ind_1] += (48 * self._R(j[ind_1], m, dx, -1) + (45 + 54 * self._xi(m) + 18 * self._xi(m)**2 + 12 * self._xi(m)**3) * self._T(j[ind_1], m, dx, -1)) \
            / (45 - 42 * self._xi(m) + 18 * self._xi(m) ** 2 - 4 * self._xi(m)**3)
        ind_2 = np.nonzero(j == 2)
        res[ind_2] -= 3 * self._T(j[ind_2], m, dx, -1) / (45 - 42 * self._xi(m) + 18 * self._xi(m) ** 2 - 4 * self._xi(m)**3)
        return res

    def _theta(self, j: np.ndarray, m, dx):
        res = self._R(j, m, dx, -1)
        ind_2 = np.nonzero(j == 2)
        res[ind_2] += (48 * self._T(j[ind_2], m, dx, -1)) / (45 - 42 * self._xi(m) + 18 * self._xi(m) ** 2 - 4 * self._xi(m)**3)
        return res

    def _vartheta(self, j: np.ndarray, m, dx):
        return self._T(j, m, dx, -1)

    def _matrix_A(self, m, dx):
        main_diag = self._alpha(np.arange(1, self.n_z), m, dx)
        u1_diag = self._varphi(np.arange(1, self.n_z - 1), m, dx)
        u2_diag = self._phi(np.arange(1, self.n_z - 2), m, dx)
        l1_diag = self._theta(np.arange(2, self.n_z), m, dx)
        l2_diag = self._vartheta(np.arange(3, self.n_z), m, dx)
        return diags([main_diag, u1_diag, u2_diag, l1_diag, l2_diag], [0, 1, 2, -1, -2])

    def _transform_field(self, field: np.ndarray, x_grid, z_grid):
        t_field = deepcopy(field)
        for x_i, x in enumerate(x_grid[1:], start=1):
            t_field[x_i][np.nonzero(z_grid < self.env.terrain(x))] = 0
            ind = np.nonzero(z_grid >= self.env.terrain(x))
            t_field[x_i][ind] = field[x_i][0:len(ind[0])]

        return t_field

    def propagate(self, initial, polarz, n_x, n_dx_out=1, n_dz_out=1):
        x_computational_grid = np.arange(0, n_x) * self.dx
        z_max_i = np.nonzero(self.z_computational_grid < self.env.z_max)[0].shape[0]
        field = Field(x_computational_grid[::n_dx_out], self.z_computational_grid[:z_max_i:n_dz_out],
                      freq_hz=self.freq_hz, precision=self.tol)
        phi = np.array([initial.aperture(a) for a in self.z_computational_grid + self.env.terrain(0)])
        field.field[0, :] = phi[:z_max_i:n_dz_out]

        for x_i, x in enumerate(x_computational_grid[1:], start=1):
            A = self._matrix_A(x_i, self.dx)
            B = self._matrix_A(x_i, -self.dx)

            u = phi * np.exp(-1j*self.k0*(self._S(x_i) * self.z_computational_grid + self._S(x_i)**2 * (x - self.dx) + (x - self.dx)))
            u[1::] = spsolve(A, B.dot(u[1::]))
            u[0] = (48 * u[1] - 3 * u[2]) / (45 - 42 * self._xi(x_i) + 18 * self._xi(x_i) ** 2 - 4 * self._xi(x_i)**3)
            phi = u * np.exp(1j*self.k0*(self._S(x_i) * self.z_computational_grid + self._S(x_i)**2 * x + x)) * self.window

            if divmod(x_i, n_dx_out)[1] == 0:
                field.field[divmod(x_i, n_dx_out)[0], :] = phi[:z_max_i:n_dz_out]
                logging.debug('Guo Zhou Long Propagator propagation x = ' + str(x))

        field.field = self._transform_field(field.field, x_computational_grid[::n_dx_out], self.z_computational_grid[:z_max_i:n_dz_out])
        return field


class CrankNicolsonPropagationTask:
    """
    Crank-Nicolson shift map parabolic equation for radio wave propagation
    Greene, Claerbout and narrow angle parabolic equation methods are implemented
    See the following paper for details
    Guo Q., Zhou C., Long Y. Greene Approximation Wide-Angle Parabolic Equation for Radio Propagation
    //IEEE Transactions on Antennas and Propagation. – 2017. – Vol. 65. – N. 11. – pp. 6048-6056.
    """
    def __init__(self, *, src: Source, env: Troposphere, type='greene', max_range_m=100000, dx_wl=100, dz_wl=1,
                 n_dx_out=1, n_dz_out=1, tol=1e-11):
        self.src = deepcopy(src)
        self.env = env
        self.max_range_m = max_range_m
        self.n_dx_out = n_dx_out
        self.n_dz_out = n_dz_out
        self.propagator = CrankNicolsonPropagator(env=self.env, src=src, dx_wl=dx_wl, dz_wl=dz_wl,
                                                  type=type, tol=tol)

    def calculate(self):
        n_x = fm.ceil(self.max_range_m / self.propagator.dx) + 1
        field = self.propagator.propagate(polarz=self.src.polarz, initial=self.src, n_x=n_x, n_dx_out=self.n_dx_out,
                                          n_dz_out=self.n_dz_out)
        return field