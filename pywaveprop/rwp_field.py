"""
JAX-based field classes for tropospheric radio wave propagation results.
"""
from copy import deepcopy
from typing import List, Optional

import jax.numpy as jnp
import numpy as np


class RWPField:
    """Tropospheric radio wave propagation field result from JAX computation."""

    def __init__(self, x_grid, z_grid, freq_hz, field=None):
        self.x_grid = np.asarray(x_grid)
        self.z_grid = np.asarray(z_grid)
        self.freq_hz = freq_hz
        self.wavelength = 3e8 / self.freq_hz
        self.precision = 1e-10
        if field is not None:
            self.field = np.asarray(field)
        else:
            self.field = np.zeros((self.x_grid.size, self.z_grid.size), dtype=complex)

    def value(self, x, z):
        """Get field value at point (x, z)."""
        return self.field[abs(self.x_grid - x).argmin(), abs(self.z_grid - z).argmin()]

    def horizontal(self, z):
        """Extract horizontal field slice at height z."""
        return self.field[:, abs(self.z_grid - z).argmin()]

    def horizontal_over_terrain(self, z0, terrain_func):
        """Extract horizontal field slice at height z0 above terrain."""
        f = np.zeros(len(self.x_grid), dtype=complex)
        for i in range(len(self.x_grid)):
            terrain_h = float(terrain_func(self.x_grid[i]))
            f[i] = self.field[i, abs(self.z_grid - terrain_h - z0).argmin()]
        return f

    def path_loss(self, gamma=0):
        """Compute path loss in dB.

        Parameters
        ----------
        gamma : float
            Atmospheric absorption coefficient in dB/km.
        """
        res = deepcopy(self)
        wavelength = 3e8 / self.freq_hz
        res.field = (
            -20 * np.log10(np.abs(self.field) + 2e-16)
            + 20 * np.log10(4 * np.pi)
            + 10 * np.tile(np.log10(self.x_grid + 2e-16), (self.z_grid.shape[0], 1)).transpose()
            - 30 * np.log10(wavelength)
            + gamma * np.tile(self.x_grid, (self.z_grid.shape[0], 1)).transpose() * 1e-3
        )
        return res

    def v_func(self):
        """Compute propagation factor V(x,z)."""
        res = deepcopy(self)
        res.field = (
            10 * np.log10(np.abs(self.field) + 2e-16)
            + 10 * np.tile(np.log10(self.x_grid + 2e-16), (self.z_grid.shape[0], 1)).transpose()
        )
        return res


class RWPRandomField:
    """Collection of field samples for Monte Carlo simulation."""

    def __init__(self):
        self.samples: List[RWPField] = []

    def add_sample(self, sample: RWPField):
        self.samples.append(sample)

    def path_loss(self, gamma=0):
        res = RWPRandomField()
        for sample in self.samples:
            res.add_sample(sample.path_loss(gamma))
        return res

    def mean(self) -> Optional[RWPField]:
        if len(self.samples) == 0:
            return None
        res = deepcopy(self.samples[0])
        if len(self.samples) == 1:
            return res
        for sample in self.samples[1:]:
            res.field += sample.field
        res.field /= len(self.samples)
        return res

    def sd(self) -> Optional[RWPField]:
        if len(self.samples) == 0:
            return None
        mean = self.mean()
        res = deepcopy(self.samples[0])
        res.field *= 0.0
        for sample in self.samples:
            res.field += (mean.field - sample.field) ** 2
        res.field = np.sqrt(res.field / len(self.samples))
        return res
