import numpy as np
import cmath as cm
from scipy.interpolate import interp1d

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'

EARTH_RADIUS = 6371000


class Field:
    def __init__(self, x_grid, z_grid, precision=1e-6):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.field = np.zeros((x_grid.size, z_grid.size), dtype=complex)
        self.precision = precision


class ImpedanceBC:
    """
    Impedance boundary (alpha1*u(z)+alpha2*u'(z))_{z=0}=0
    """
    def __init__(self, alpha1, alpha2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def __eq__(self, other):
        return (self.alpha1 == other.alpha1) and (self.alpha2 == other.alpha2)

    def __hash__(self):
        return hash(self.alpha1 + 2 * self.alpha2)


class DirichletBC(ImpedanceBC):

    def __init__(self):
        ImpedanceBC.__init__(self, 1, 0)


class NeumannBC(ImpedanceBC):

    def __init__(self):
        ImpedanceBC.__init__(self, 0, 1)


class EarthSurfaceBC(ImpedanceBC):

    def __init__(self, wavelength, conductivity, permittivity, polarz='H'):
        k0 = 2 * cm.pi / wavelength
        self.conductivity = conductivity
        self.permittivity = permittivity
        if polarz.upper() == 'H':
            ImpedanceBC.__init__(self, 1, 1j * k0 * (permittivity + 1j * 60 * conductivity * wavelength) ** (1 / 2))
        else:
            ImpedanceBC.__init__(self, 1, 1j * k0 * (permittivity + 1j * 60 * conductivity * wavelength) ** (-1 / 2))


class TransparentBS:
    pass


class TransparentConstBS(TransparentBS):
    """
    Constant refractive index in outer domain
    """
    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(55726572865)


class TransparentLinearBS(TransparentBS):
    """
    Linear refractive index in outer domain
    """
    def __init__(self, mu_N):
        self.mu_n2 = mu_N

    def __eq__(self, other):
        return self.mu_n2 == other.mu_n2

    def __hash__(self):
        return hash(self.mu_n2)


class EarthAtmosphereBC(TransparentLinearBS):

    def __init__(self):
        TransparentLinearBS.__init__(self, 2 / EARTH_RADIUS)


class Terrain:

    def __init__(self, edge_range, edge_height):
        self.edge_range = edge_range
        self.edge_height = edge_height
        self.terrain_func = interp1d(edge_range, edge_height, fill_value="extrapolate")

    def __call__(self, *args, **kwargs):
        return self.terrain_func(args[0])


class EMEnvironment:
    lower_boundary = ImpedanceBC(1, 0)
    upper_boundary = TransparentConstBS()
    z_min = 0.0
    z_max = 100.0

    def __init__(self):
        self.N_profile = None
        self.terrain = None

    def n2_profile(self, x: float, z: np.ndarray):
        if self.N_profile is None:
            return z * 0
        return self.N_profile(x, z)

    def impediment(self, x: float, z_grid: np.ndarray):
        if self.terrain is not None:
            return np.nonzero(z_grid <= self.terrain(x))
        return []


class gauss_source:

    def __init__(self, k0, height, beam_width, eval_angle, polarz):
        self.height = height
        self.k0 = k0
        self.beam_width = beam_width
        self.eval_angle = eval_angle
        self.polarz = polarz
        self._ww = cm.sqrt(2 * cm.log(2)) / (k0 * cm.sin(beam_width * cm.pi / 180 / 2))

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
