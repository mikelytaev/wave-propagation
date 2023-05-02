import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import cycle
from rwp.field import Field, Field3d
from rwp.environment import *
import logging

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class FieldVisualiser:
    bw_lines = (
        {'color': 'black', 'dashes': (None, None)},
        {'color': 'black', 'dashes': [5, 5]},
        {'color': 'black', 'dashes': [5, 3, 1, 3]},
        {'color': 'black', 'dashes': [1, 3]}
    )

    color_lines = (
        {'color': 'red'},
        {'color': 'blue'},
        {'color': 'green'},
        {'color': 'black'}
    )

    def __init__(self, field: Field, env: Troposphere, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='', x_mult=1.0, bw=False):
        trans_func = np.vectorize(trans_func)
        self.field = trans_func(field.field).real
        self.x_grid, self.z_grid = field.x_grid, field.z_grid
        self.precision = field.precision
        self.max = np.max(self.field)
        self.min = max(trans_func(self.precision) + self.max, np.min(self.field))
        self.label = label
        self.x_grid = self.x_grid * x_mult
        self.x_mult = x_mult
        if bw:
            self.lines_iter = cycle(self.bw_lines)
        else:
            self.lines_iter = cycle(self.color_lines)
        self.env = env

    def plot2d(self, min, max, show_terrain=False):
        norm = Normalize(min, max)
        extent = [self.x_grid[0], self.x_grid[-1], self.z_grid[0], self.z_grid[-1]]
        plt.figure(figsize=(6, 3.2))
        plt.imshow(self.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        plt.colorbar(fraction=0.046, pad=0.04)
        if show_terrain:
            terrain_grid = np.array([self.env.terrain.elevation(v) for v in self.x_grid / self.x_mult])
            plt.plot(self.x_grid, terrain_grid, 'k')
            plt.fill_between(self.x_grid, terrain_grid*0, terrain_grid, color='brown')
        return plt

    def plot_hor(self, z0, *others):
        plt.figure(figsize=(6, 3.2))
        for a in (self,) + others:
            plt.plot(a.x_grid, a.field[:, abs(a.z_grid - z0).argmin()], label=a.label, **next(self.lines_iter))
            if len(self.z_grid) != len(a.z_grid) or not np.all(self.z_grid == a.z_grid):
                logging.warning("z grid is not equal. It may cause some unexpected differences in the results.")
        plt.legend()
        plt.xlim([self.x_grid[0], self.x_grid[-1]])
        return plt

    def plot_hor_over_terrain(self, z0, *others):
        plt.figure(figsize=(6, 3.2))
        for a in (self,) + others:
            field = np.zeros(len(a.x_grid))
            for i in range(0, len(a.x_grid)):
                field[i] = a.field[i, abs(a.z_grid - self.env.terrain.elevation(a.x_grid[i] / self.x_mult) - z0).argmin()]
            if len(self.z_grid) != len(a.z_grid) or not np.all(self.z_grid == a.z_grid):
                logging.warning("z grid is not equal. It may cause some unexpected differences in the results.")
            plt.plot(a.x_grid, field, label=a.label, **next(self.lines_iter))
        plt.legend()
        plt.xlim([self.x_grid[0], self.x_grid[-1]])
        return plt

    def plot_ver(self, x0, ax=plt, *others):
        x0 *= self.x_mult
        ax.plot(self.field[abs(self.x_grid - x0).argmin(), :], self.z_grid, label=self.label)
        for a in others:
            ax.plot(a.field[abs(a.x_grid - x0).argmin(), :], a.z_grid, label=a.label)
            if len(self.x_grid) != len(a.x_grid) or not np.all(self.x_grid == a.x_grid):
                logging.warning("x grid is not equal. It may cause some unexpected differences in the results.")
        ax.legend()
        return ax

    def plot_ver_measurements(self, x0, height, z_min, z_max, measurements, mea_label='measurements', fit=0):
        x0 *= self.x_mult
        g = interp1d(x=measurements[0, :], y=measurements[1, :], fill_value='extrapolate')
        z_ind = np.nonzero(np.logical_and(self.z_grid >= z_min + height, self.z_grid <= z_max + height))
        plt.plot(self.z_grid[z_ind] - height, self.field[abs(self.x_grid - x0).argmin(), z_ind][0] * (1 - fit) +
                 g(self.z_grid[z_ind] - height) * fit, label=self.label)
        plt.plot(measurements[0, :], measurements[1, :], 'ro', label=mea_label)
        plt.legend()
        plt.xlim([z_min, z_max])
        plt.grid(True)
        return plt


class FieldVisualiser3D:
    bw_lines = (
        {'color': 'black', 'dashes': (None, None)},
        {'color': 'black', 'dashes': [5, 5]},
        {'color': 'black', 'dashes': [5, 3, 1, 3]},
        {'color': 'black', 'dashes': [1, 3]}
    )

    color_lines = (
        {'color': 'red'},
        {'color': 'blue'},
        {'color': 'green'},
        {'color': 'black'}
    )

    def __init__(self, field: Field3d, trans_func=lambda v: abs(v), label='', x_mult=1.0, bw=False):
        self.trans_func = np.vectorize(trans_func)
        self.field = field.field
        self.trans_field = self.trans_func(self.field).real
        self.x_grid, self.y_grid, self.z_grid = field.x_grid, field.y_grid, field.z_grid
        self.label = label
        if bw:
            self.lines_iter = cycle(self.bw_lines)
        else:
            self.lines_iter = cycle(self.color_lines)

    def _plot2d(self, f, min_val, max_val, x_min, x_max, y_min, y_max):
        norm = Normalize(min_val, max_val)
        extent = [x_min, x_max, y_min, y_max]
        plt.figure(figsize=(6, 3.2))
        plt.imshow(self.trans_func(f).real, extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        plt.colorbar(fraction=0.046, pad=0.04)
        return plt

    def plot_xy(self, *, z0, min_val, max_val):
        return self._plot2d(self.field[:, :, abs(self.z_grid - z0).argmin()].T, min_val, max_val,
                            self.x_grid[0], self.x_grid[-1], self.y_grid[0], self.y_grid[-1])

    def plot_yz(self, *, x0, min_val, max_val):
        return self._plot2d(self.field[abs(self.x_grid - x0).argmin(), :, :].T[::-1, :], min_val, max_val,
                            self.y_grid[0], self.y_grid[-1], self.z_grid[0], self.z_grid[-1])

    def plot_xz(self, *, y0, min_val, max_val):
        return self._plot2d(self.field[:, abs(self.y_grid - y0).argmin(), :].T[::-1, :], min_val, max_val,
                            self.x_grid[0], self.x_grid[-1], self.z_grid[0], self.z_grid[-1])

    def plot_x(self, y0, z0, others=[]):
        plt.figure(figsize=(6, 3.2))
        for a in [self] + others:
            plt.plot(a.x_grid, a.trans_field[:, abs(a.y_grid - y0).argmin(), abs(a.z_grid - z0).argmin()], label=a.label, **next(self.lines_iter))
        plt.legend()
        plt.xlim([self.x_grid[0], self.x_grid[-1]])
        return plt
