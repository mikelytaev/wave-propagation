from rwp.WPDefs import *
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from itertools import cycle
import matplotlib

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class FieldVisualiser:
    bw_lines = (
        {'color': 'black', 'dashes': (None,None)},
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

    def __init__(self, field: Field, trans_func=lambda v: abs(v), label='', x_mult=1.0, bw=False):
        trans_func = np.vectorize(trans_func)
        self.field = trans_func(field.field).real
        self.x_grid, self.z_grid = field.x_grid, field.z_grid
        self.precision = field.precision
        self.max = np.max(self.field)
        self.min = max(trans_func(self.precision) + self.max, np.min(self.field))
        self.label = label
        self.x_grid *= x_mult
        self.x_mult = x_mult
        if bw:
            self.lines_iter = cycle(self.bw_lines)
        else:
            self.lines_iter = cycle(self.color_lines)

    def plot2d(self, min, max):
        norm = Normalize(min, max)
        extent = [self.x_grid[0], self.x_grid[-1], self.z_grid[0], self.z_grid[-1]]
        plt.figure(figsize=(6, 3.2))
        plt.imshow(self.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        plt.colorbar(fraction=0.046, pad=0.04)
        return plt

    def plot_hor(self, z0, *others):
        plt.figure(figsize=(6, 3.2))
        for a in (self,) + others:
            plt.plot(a.x_grid, a.field[:, abs(a.z_grid - z0).argmin()], label=a.label, **next(self.lines_iter))
        plt.legend()
        plt.xlim([self.x_grid[0], self.x_grid[-1]])
        return plt

    def plot_ver(self, x0, ax=plt, *others):
        x0 *= self.x_mult
        ax.plot(self.field[abs(self.x_grid - x0).argmin(), :], self.z_grid, label=self.label)
        for a in others:
            ax.plot(a.field[abs(a.x_grid - x0).argmin(), :], a.z_grid, label=a.label)
        ax.legend()
        return ax
