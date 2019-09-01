from dataclasses import dataclass
from copy import deepcopy
import cmath as cm
import numpy as np
from propagators.math_utils import *
from transforms.fcc_fourier import FCCAdaptiveFourier


@dataclass
class WaveNumberIntegratorParams:
    alpha: float
    fcc_tol: float
    x_grid_m: np.ndarray = None
    z_grid_m: np.ndarray = None


class WaveNumberIntegrator:

    def __init__(self, k0: float, q_func, params: WaveNumberIntegratorParams):
        self.params = deepcopy(params)
        self.q_func = deepcopy(q_func)
        self.k0 = k0
        self.q = 0

    def _gamma(self, k_x):
        alpha = self.params.alpha
        a = np.sqrt((np.sqrt((self.k0 ** 2 - k_x ** 2) ** 2 + (alpha * self.k0 ** 2) ** 2) - (self.k0 ** 2 - k_x ** 2)) / 2)
        d = -np.sqrt((np.sqrt((self.k0 ** 2 - k_x ** 2) ** 2 + (alpha * self.k0 ** 2) ** 2) + (self.k0 ** 2 - k_x ** 2)) / 2)
        return a + 1j*d

    def green_function_free_space(self, z, zsh, k_x):
        zv, zshv, k_xv = np.meshgrid(z, zsh, k_x, indexing='ij')
        tgvp = self._gamma(k_xv)
        gv = -1 / (2 * tgvp) * np.exp(-tgvp * np.abs(zv - zshv))
        return np.squeeze(gv)

    def green_function_lower_boundary(self, z, zsh, k_x):
        zv, zshv, k_xv = np.meshgrid(z, zsh, k_x, indexing='ij')
        tgvp = self._gamma(k_xv)
        gv = -1 / (2 * tgvp) * np.exp(-tgvp * np.abs(zv - zshv)) + 1 / (2 * tgvp) * np.exp(-tgvp * (zv + zshv))
        return np.squeeze(gv)

    def _rhs(self, k_x):
        if isinstance(self.q_func, DeltaFunction):
            return self.green_function_lower_boundary(self.params.z_grid_m, self.q_func.x_c, k_x)
        elif callable(self.q_func):
            dz = self.params.z_grid_m[1] - self.params.z_grid_m[0]
            self.green_function_lower_boundary(self.params.z_grid_m, self.params.z_grid_m, k_x) * self.q_func(self.params.z_grid_m) * dz

    def calculate(self):
        fcca = FCCAdaptiveFourier(2 * self.k0 *1.1, -self.params.x_grid_m, rtol=self.params.fcc_tol)
        res = fcca.forward(lambda x: self._rhs(x), -self.k0*1.1, self.k0*1.1)
        return res