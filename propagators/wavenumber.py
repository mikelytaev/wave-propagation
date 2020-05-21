from dataclasses import dataclass
from copy import deepcopy
import cmath as cm
import numpy as np
from propagators.math_utils import *
from transforms.fcc_fourier import FCCAdaptiveFourier
import types
from scipy.linalg import sqrtm, solve


@dataclass
class WaveNumberIntegratorParams:
    fcc_tol: float
    x_grid_m: np.ndarray = None
    z_computational_grid_m: np.ndarray = None
    z_out_grid_m: np.ndarray = None
    lower_refl_coef: types.FunctionType = lambda k_x: 0
    upper_refl_coef: types.FunctionType = lambda k_x: 0
    alpha: float = 1e-3
    alpha_compensate: bool = False
    min_p_k0: float = None
    max_p_k0: float = 1.1
    het: types.FunctionType = None
    

class WaveNumberIntegrator:

    def __init__(self, k0: float, initial_func, params: WaveNumberIntegratorParams):
        self.params = deepcopy(params)
        self.initial_func = deepcopy(initial_func)
        self.k0 = k0
        self.dz = self.params.z_computational_grid_m[1] - self.params.z_computational_grid_m[0]

        if callable(self.initial_func):
            d2_m = np.diag(np.ones(len(self.params.z_computational_grid_m) - 1), -1) + \
                   np.diag(np.ones(len(self.params.z_computational_grid_m) - 1), -1) + \
                   np.diag(-2 * np.ones(len(self.params.z_computational_grid_m)), 0)
            self.q1 = self.initial_func(self.params.z_computational_grid_m)
            self.q2 = sqrtm(np.diag(np.eye(len(self.params.z_computational_grid_m))) + d2_m / self.k0 ** 2).dot(self.q1)

        if self.params.het:
            #assert np.all(self.params.z_computational_grid_m == self.params.z_out_grid_m)
            assert len(self.params.z_computational_grid_m) == len(self.params.z_out_grid_m)
            _, self.het_v = np.meshgrid(self.params.z_out_grid_m, self.params.het(self.params.z_computational_grid_m), indexing='ij')

    def green_function(self, z, zsh, k_x):
        """

        :param z: array or scalar
        :param zsh: array of scalar
        :param k_x: scalar
        :return:
        """
        zv, zshv = np.meshgrid(z, zsh, indexing='ij')
        gamma = cm.sqrt(self.k0**2*(1+1j*self.params.alpha) - k_x**2)
        h = self.params.z_computational_grid_m[-1]
        theta = cm.acos(k_x / self.k0)*180/cm.pi
        r_0 = self.params.lower_refl_coef(theta)
        r_h = self.params.upper_refl_coef(theta)
        mult = 1/(2j*gamma*(1-r_0*r_h*np.exp(2j*gamma*h)))

        res = np.exp(1j*gamma*abs(zv - zshv))
        if abs(r_0) > 1e-12:
            res += r_0 * np.exp(1j*gamma*(zv + zshv))

        if abs(r_h) > 1e-12:
            res += r_h * np.exp(2j*gamma*h) * np.exp(-1j * gamma * (zv + zshv))

        if abs(r_0) > 1e-12 and abs(r_h) > 1e-12:
            res += r_0 * r_h * np.exp(2j*gamma*h) * np.exp(-1j*gamma*abs(zv - zshv))

        res *= mult
        return np.squeeze(res)

    def _rhs(self, k_x):
        if isinstance(self.initial_func, DeltaFunction):
            res = self.green_function(self.params.z_out_grid_m, self.initial_func.x_c, k_x)
        elif callable(self.initial_func):
            r = 1j * k_x * self.q1 + self.q2
            res = self.green_function(self.params.z_out_grid_m, self.params.z_computational_grid_m, k_x).dot(r) * self.dz

        if self.params.het:
            a = self.green_function(self.params.z_out_grid_m, self.params.z_computational_grid_m,
                                    k_x) * self.het_v * self.dz
            res = solve(np.eye(len(self.params.z_computational_grid_m)) + a, res, overwrite_a=True, overwrite_b=True)

        return res

    def calculate(self):
        if not self.params.min_p_k0:
            self.params.min_p_k0 = -self.params.max_p_k0
        min_int = self.k0 * self.params.min_p_k0
        max_int = self.k0 * self.params.max_p_k0
        fcca = FCCAdaptiveFourier(max_int-min_int, -self.params.x_grid_m, rtol=self.params.fcc_tol, x_n=15)
        mult = 1 if min_int < 0 else 2
        res = mult / cm.sqrt(2*cm.pi) * fcca.forward(lambda k_x: self._rhs(k_x), min_int, max_int)
        if self.params.alpha_compensate:
            xv, _ = np.meshgrid(self.params.x_grid_m, self.params.z_out_grid_m, indexing='ij')
            res *= np.exp(self.k0*self.params.alpha/2 * xv)
        return res
