__author__ = 'Mikhail'

from rwp.WPDefs import *
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize


class FieldVisualiser:

    def __init__(self, field: Field, trans_func=lambda v: abs(v)):
        self.field = field
        self.trans_func = np.vectorize(trans_func)

    def plot2d(self, min, max):
        norm = Normalize(min, max)
        extent = [self.field.x_grid[0], self.field.x_grid[-1], self.field.z_grid[0], self.field.z_grid[-1]]
        plt.imshow(self.trans_func(self.field.field.T[::-1,:]).real, extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        plt.colorbar()
        return plt

    def plot_hor(self, z0):
        z0_i = abs(self.field.z_grid - z0).argmin()
        plt.plot(self.field.x_grid, self.trans_func(self.field.field[:,z0_i]))
        return plt