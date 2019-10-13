from propagators.field import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import cycle


class FieldVisualiser2d:
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

    def __init__(self, field: Field2d, label='', black_white=False, trans_func=lambda v: v):
        self.trans_func = np.vectorize(trans_func)
        self.field = field
        self.label = label
        if black_white:
            self.lines_iter = cycle(self.bw_lines)
        else:
            self.lines_iter = cycle(self.color_lines)

    def plot2d(self, min_val, max_val):
        norm = Normalize(min_val, max_val)
        extent = [self.field.x_grid[0], self.field.x_grid[-1], self.field.z_grid[0], self.field.z_grid[-1]]
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.trans_func(self.field.field).real.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        ax.colorbar(fraction=0.046, pad=0.04)
        return fig

    def plot_hor(self, z0, *others):
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(1, 1, 1)
        for a in (self,) + others:
            ax.plot(a.x_grid, a.field[:, abs(a.z_grid - z0).argmin()], label=a.label, **next(self.lines_iter))
        ax.legend()
        ax.xlim([self.x_grid[0], self.x_grid[-1]])
        return fig
