import cmath as cm

import numpy as np

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class Field:
    def __init__(self, x_grid, z_grid, precision=1e-6):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.field = np.zeros((x_grid.size, z_grid.size), dtype=complex)
        self.precision = precision


class Source:
    pass


class GaussSource(Source):

    def __init__(self, wavelength, height, beam_width, eval_angle, polarz):
        self.height = height
        self.k0 = 2 * cm.pi / wavelength
        self.wavelength = wavelength
        self.beam_width = beam_width
        self.eval_angle = eval_angle
        self.polarz = polarz
        self._ww = cm.sqrt(2 * cm.log(2)) / (self.k0 * cm.sin(beam_width * cm.pi / 180 / 2))

    def _ufsp(self, z):
        return 1 / (cm.sqrt(cm.pi) * self._ww) * cm.exp(-1j * self.k0 * cm.sin(self.eval_angle * cm.pi / 180) * z) * \
               cm.exp(-((z - self.height) / self._ww) ** 2)

    def _ufsn(self, z):
        return 1 / (cm.sqrt(cm.pi) * self._ww) * cm.exp(-1j * self.k0 * cm.sin(self.eval_angle * cm.pi / 180) * (-z)) * \
               cm.exp(-((-z - self.height) / self._ww) ** 2)

    def __call__(self, *args, **kwargs):
        if self.polarz.upper() == 'H':
            return self._ufsp(args[0]) - self._ufsn(args[0])
        else:
            return self._ufsp(args[0]) + self._ufsn(args[0])
