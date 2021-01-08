from propagators.vis import *
from uwa.field import *
import cmath as cm


class AcousticPressureFieldVisualiser2d(FieldVisualiser2d):
    """
    Parameters
    ----------
    field : AcousticPressureField
    label : bool
        name
    black_white : bool
     generate black and white figures
     Default is false
    x_units: 'km' or 'm'
    lang: "en" or "ru"
        axis language
        Default is en
    f_units: 'db' or 'abs'
        Field units
        Default is dB (20log10|u|)
    """

    def __init__(self, field: AcousticPressureField, label='', black_white=False, x_units='km', lang='en', f_units='db'):
        self.f_units = f_units
        if self.f_units.lower() == 'db':
            trans_func = lambda v: 20 * cm.log10(1e-16 + abs(v))
        else:
            trans_func = lambda v: abs(v)
        super().__init__(field, label, black_white, trans_func)
        self.x_units = x_units
        self.lang = lang
        if x_units.lower() == 'km':
            self.x_mult = 1e-3
        else:
            self.x_mult = 1

    def plot2d(self, min_val, max_val):
        norm = Normalize(min_val, max_val)
        extent = [self.field.x_grid[0]*self.x_mult, self.field.x_grid[-1]*self.x_mult, self.field.z_grid[-1], self.field.z_grid[0]]
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(self.trans_func(self.field.field).real.T[:, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        fig.colorbar(im, fraction=0.046, pad=0.04)
        if self.lang == 'en':
            if self.x_units.lower() == 'km':
                ax.set_xlabel('Range (km)')
            elif self.x_units.lower() == 'm':
                ax.set_xlabel('Range (m)')
            ax.set_ylabel('Depth (m)')
        fig.tight_layout()
        return fig

    def plot_hor(self, z0, *others):
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(1, 1, 1)
        for a in (self,) + others:
            ax.plot(a.field.x_grid*self.x_mult, self.trans_func(a.field.field[:, abs(a.field.z_grid - z0).argmin()]).real, label=a.label, **next(self.lines_iter))

        if self.lang == 'en':
            if self.x_units.lower() == 'km':
                ax.set_xlabel('Range (km)')
            elif self.x_units.lower() == 'm':
                ax.set_xlabel('Range (m)')
            if self.f_units.lower() == 'db':
                ax.set_ylabel('20log|u| (dB)')
            elif self.f_units.lower() == 'abs':
                ax.set_ylabel('|u| (dB)')
        fig.tight_layout()
        ax.legend()
        ax.grid()
        ax.set_xlim([self.field.x_grid[0]*self.x_mult, self.field.x_grid[-1]*self.x_mult])
        return fig