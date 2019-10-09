from propagators.field import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class FieldVisualiser2d:

    def __init__(self, field: Field2d, trans_func=lambda v: v):
        self.trans_func = np.vectorize(trans_func)
        self.field = field

    def plot2d(self, min_val, max_val):
        norm = Normalize(min_val, max_val)
        extent = [self.field.x_grid[0], self.field.x_grid[-1], self.field.z_grid[0], self.field.z_grid[-1]]
        plt.figure(figsize=(6, 3.2))
        plt.imshow(self.trans_func(self.field.field).T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.show()

    def plot_x(self, z0, *others):
        plt.figure(figsize=(6, 3.2))
        for a in (self,) + others:
            plt.plot(a.x_grid, a.field[:, abs(a.z_grid - z0).argmin()], label=a.label, **next(self.lines_iter))
        plt.legend()
        plt.xlim([self.x_grid[0], self.x_grid[-1]])
        return plt