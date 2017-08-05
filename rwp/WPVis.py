from rwp.WPDefs import *
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'

class FieldVisualiser:

    def __init__(self, field: Field, trans_func=lambda v: abs(v), label=''):
        trans_func = np.vectorize(trans_func)
        self.field = trans_func(field.field).real
        self.x_grid, self.z_grid = field.x_grid, field.z_grid
        self.precision = field.precision
        self.max = np.max(self.field)
        self.min = max(trans_func(self.precision) + self.max, np.min(self.field))
        self.label = label

    def plot2d(self, min, max):
        norm = Normalize(min, max)
        extent = [self.x_grid[0], self.x_grid[-1], self.z_grid[0], self.z_grid[-1]]
        plt.imshow(self.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        plt.colorbar()
        return plt

    def plot_hor(self, z0, *others):
        plt.plot(self.x_grid, self.field[:, abs(self.z_grid - z0).argmin()], label=self.label)
        for a in others:
            plt.plot(a.x_grid, a.field[:, abs(a.z_grid - z0).argmin()], label=a.label)
        plt.legend()
        z0_i = abs(self.z_grid - z0).argmin()
        #plt.axes([self.x_grid[0], self.x_grid[-1], max(min(self.field[:, z0_i]), self.min), max(self.field[:, z0_i])])
        return plt
