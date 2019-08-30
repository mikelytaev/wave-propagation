from dataclasses import dataclass
from copy import deepcopy
import cmath as cm
import numpy as np
from transforms.fcc_fourier import FCCAdaptiveFourier


@dataclass
class WaveNumberIntegratorParams:
    alpha: float
    fcc_tol: float
    x_grid_m: np.ndarray = None
    z_grid_m: np.ndarray = None


class WaveNumberIntegrator:

    def __init__(self, wavelength: float, params: WaveNumberIntegratorParams):
        self.params = deepcopy(params)
        self.k0 = 2 * cm.pi / wavelength
        self.q = 0
        self.z_c = 50

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
        gv = -1 / (2 * tgvp) * np.exp(-tgvp * np.abs(zv - zshv)) + 1 / (2 * tgvp) * np.exp(-tgvp * (zv - zshv))
        return np.squeeze(gv)

    def _rhs(self, k_x):
        self.green_function_lower_boundary(self.params.z_grid_m, self.z_c, k_x)

    def calculate(self):
        fcca = FCCAdaptiveFourier(2 * self.k0, -self.params.x_grid_m, rtol=self.params.fcc_tol)
        res = fcca.forward(self._rhs, -self.k0, self.k0)
        return res