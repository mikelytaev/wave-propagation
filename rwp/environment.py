import cmath as cm

import numpy as np
from scipy.interpolate import interp1d


EARTH_RADIUS = 6371000

# магические константы из петула. TODO: разобраться откуда это вообще

a = [1.4114535e-2, 3.8586749, 79.027635, -0.65750351, 201.97103, 857.94335, 915.31026, 0.8756665, 5.5990969e-3,
     215.87521, 0.17381269, 2.4625032e-2, -4.9560275e-2, 2.2953743e-4, 0.000038814567, 1.2434792e-4, 51852.543,
     4.13105e-5]

b = [-5.2122497e-8, -2.1179295e-5, -2.2083308e-5, 5.5620223e-5, -2.5539582e-3, -8.9983662e-5, -9.4530022e-6,
     4.7236085e-5, 8.7798277e-5, -7.6649237e-5, 1.2655183e-4, 1.8254018e-4, 2.9876572e-5, -8.1212741e-7, 8.467523e-2,
     2.824598e-4, 3.883854e-2, 2.03589e-7]

c = [5.8547829e-11, 9.1253873e-4, -3.5486605e-4, 6.6113198e-4, 1.2197967e-2, 5.5275278e-2, -4.0348211e-3, 2.6051966e-8,
     6.2451017e-8, -2.6151055e-3, -1.6790756e-9, -2.664754e-8, -3.0561848e-10, 1.8045461e-9, 9.878241e-6, 8.680839e-7,
     389.58894, -3.1739e-12]

d = [-7.6717423e-16, 6.5727504e-10, 2.7067836e-9, 3.0140816-10, 3.7853169e-5, 8.8247139e-8, 4.892281e-8, -9.235936e-13,
     -7.1317207e-12,1.2565999e-8,1.1037608e-14,7.6508732e-12,1.1131828e-15,-1.960677e-12,-9.736703e-5,-6.755389e-8,
     6.832108e-5,4.52331e-17]

e = [2.9856318e-21,1.5309921e-8,8.210184e-9,1.4876952e-9,-1.728776e-6,0.0,7.4342897e-7,1.4560078e-17,4.2515914e-16,
     1.9484482e-7,-2.9223433e-20,-7.4193268e-16,0.0,1.2569594e-15,7.990284e-8,7.2701689e-11,0.0,0.0]

f = [0.0,-1.9647664e-15,-1.0007669e-14,0.0,0.0,0.0,0.0,-1.1129348e-22,-1.240806e-20,0.0,0.0,0.0,0.0,-4.46811e-19,
     3.269059e-7,2.8728975e-12,0.0,0.0]


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

    def __call__(self, wavelength, polarz):
        return self.alpha1, self.alpha2


class DirichletBC(ImpedanceBC):

    def __init__(self):
        ImpedanceBC.__init__(self, 1, 0)


class NeumannBC(ImpedanceBC):

    def __init__(self):
        ImpedanceBC.__init__(self, 0, 1)


class PECSurfaceBC(ImpedanceBC):
    def __init__(self):
        pass

    def __call__(self, wavelength, polarz):
        if polarz == 'H':
            return 1, 0
        else:
            return 0, 1


class EarthSurfaceBC(ImpedanceBC):

    def __init__(self, permittivity, conductivity):
        self.permittivity = permittivity
        self.conductivity = conductivity

    def __call__(self, wavelength, polarz):
        k0 = 2 * cm.pi / wavelength
        if polarz.upper() == 'H':
            self.alpha1 = 1
            self.alpha2 = 1j * k0 * (self.permittivity + 1j * 60 * self.conductivity * wavelength) ** (1 / 2)
        else:
             self.alpha1 = 1
             self.alpha2 = 1j * k0 * (self.permittivity + 1j * 60 * self.conductivity * wavelength) ** (-1 / 2)

        return self.alpha1, self.alpha2


class SeaSurfaceBC(EarthSurfaceBC):

    def __init__(self):
        pass

    def __call__(self, wavelength, polarz):
        freqMHz = 2 * cm.pi * 3e8 / wavelength * 1e-6
        epsilon = 70.0
        sigma = 5.0  # S / m
        mi = 1
        mi1 = mi + 1

        if freqMHz > 2253.5895:
            epsilon = 1.0 / (a[mi] + b[mi] * freqMHz + c[mi] * freqMHz ** 2 + d[mi] * freqMHz ** 3 + e[mi] * freqMHz ** 4)

        if freqMHz > 1106.207:
            sigma = a[mi1] + c[mi1] * freqMHz + e[mi1] * freqMHz ** 2
            sigma = sigma / (1.0 + b[mi1] * freqMHz + d[mi1] * freqMHz ** 2 + f[mi1] * freqMHz ** 3)

        self.permittivity = epsilon
        self.conductivity = sigma

        return EarthSurfaceBC.__call__(self, wavelength, polarz)


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
        self.mu_n2m1 = 2 * mu_N * 1e-6

    def __eq__(self, other):
        return self.mu_n2m1 == other.mu_n2m1

    def __hash__(self):
        return hash(self.mu_n2m1)


class EarthAtmosphereBC(TransparentLinearBS):

    def __init__(self, Earth_radius=EARTH_RADIUS):
        TransparentLinearBS.__init__(self, 1 / Earth_radius * 1e6)


class Terrain:

    def __init__(self):
        self.edge_range = []
        self.edge_height = []

    def __call__(self, *args, **kwargs):
        return 0.0


class LinearTerrain(Terrain):

    def __init__(self, edge_range, edge_height):
        self.edge_range = edge_range
        self.edge_height = edge_height
        self.terrain_func = interp1d(edge_range, edge_height, fill_value="extrapolate")

    def __call__(self, x):
        return self.terrain_func(x)


class KnifeEdges(Terrain):

    def __init__(self, edge_range, edge_height):
        self.edge_range = edge_range
        self.edge_height = edge_height

    def __call__(self, x):
            return 0


class EMEnvironment:

    def __init__(self):
        self.N_profile = None
        self.terrain = Terrain()
        self.lower_boundary = ImpedanceBC(1, 0)
        self.upper_boundary = TransparentConstBS()
        self.z_min = 0.0
        self.z_max = 100.0

    def n2m1_profile(self, x: float, z: np.ndarray):
        if self.N_profile is None:
            return z * 0
        return 2 * self.N_profile(x, z) * 1e-6

    def n2_profile(self, x: float, z: np.ndarray):
        return self.n2m1_profile(x, z) + 1.0

    def impediment(self, x: float, z_grid: np.ndarray):
        if self.terrain is not None:
            return np.nonzero(z_grid < self.terrain(x))
        return []


class EarthAtmosphereEnvironment(EMEnvironment):

    def __init__(self, *, boundary_condition: ImpedanceBC, height=300, Earth_radius=EARTH_RADIUS, M_profile=None):
        EMEnvironment.__init__(self)
        self.z_min = 0.0
        self.z_max = height
        self.lower_boundary = boundary_condition
        self.Earth_radius = Earth_radius
        self.M_profile = M_profile
        if self.M_profile is not None:
            self.Earth_radius = 1E6 / (self.M_profile(0, self.z_max + 1) - self.M_profile(0, self.z_max))
        self.upper_boundary = EarthAtmosphereBC(self.Earth_radius)

    def n2m1_profile(self, x: float, z: np.ndarray):
        if self.M_profile is None:
            return 2 * z / self.Earth_radius
        return 2 * self.M_profile(x, z) * 1e-6
