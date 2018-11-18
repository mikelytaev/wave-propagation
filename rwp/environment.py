import cmath as cm
import abc
from enum import Enum

import numpy as np
from scipy.interpolate import interp1d
from operator import itemgetter


EARTH_RADIUS = 6371000

VACUUM_PERMITTIVITY = 8.854187817e-12
LIGHT_SPEED = 3e8

# магические константы из петула. TODO: разобраться откуда это вообще

MAGIC_A = [1.4114535e-2, 3.8586749, 79.027635, -0.65750351, 201.97103, 857.94335, 915.31026, 0.8756665, 5.5990969e-3,
           215.87521, 0.17381269, 2.4625032e-2, -4.9560275e-2, 2.2953743e-4, 0.000038814567, 1.2434792e-4, 51852.543,
           4.13105e-5]

MAGIC_B = [-5.2122497e-8, -2.1179295e-5, -2.2083308e-5, 5.5620223e-5, -2.5539582e-3, -8.9983662e-5, -9.4530022e-6,
           4.7236085e-5, 8.7798277e-5, -7.6649237e-5, 1.2655183e-4, 1.8254018e-4, 2.9876572e-5, -8.1212741e-7, 8.467523e-2,
           2.824598e-4, 3.883854e-2, 2.03589e-7]

MAGIC_C = [5.8547829e-11, 9.1253873e-4, -3.5486605e-4, 6.6113198e-4, 1.2197967e-2, 5.5275278e-2, -4.0348211e-3, 2.6051966e-8,
           6.2451017e-8, -2.6151055e-3, -1.6790756e-9, -2.664754e-8, -3.0561848e-10, 1.8045461e-9, 9.878241e-6, 8.680839e-7,
           389.58894, -3.1739e-12]

MAGIC_D = [-7.6717423e-16, 6.5727504e-10, 2.7067836e-9, 3.0140816 - 10, 3.7853169e-5, 8.8247139e-8, 4.892281e-8, -9.235936e-13,
           -7.1317207e-12, 1.2565999e-8, 1.1037608e-14, 7.6508732e-12, 1.1131828e-15, -1.960677e-12, -9.736703e-5, -6.755389e-8,
           6.832108e-5, 4.52331e-17]

MAGIC_E = [2.9856318e-21, 1.5309921e-8, 8.210184e-9, 1.4876952e-9, -1.728776e-6, 0.0, 7.4342897e-7, 1.4560078e-17, 4.2515914e-16,
           1.9484482e-7, -2.9223433e-20, -7.4193268e-16, 0.0, 1.2569594e-15, 7.990284e-8, 7.2701689e-11, 0.0, 0.0]

MAGIC_F = [0.0, -1.9647664e-15, -1.0007669e-14, 0.0, 0.0, 0.0, 0.0, -1.1129348e-22, -1.240806e-20, 0.0, 0.0, 0.0, 0.0, -4.46811e-19,
           3.269059e-7, 2.8728975e-12, 0.0, 0.0]


class Polarization(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class BoundaryCondition:
    pass


class TransparentBS(BoundaryCondition):

    def n2m1(self, freq_hz: int):
        return None


class TransparentConstBS(TransparentBS):
    """
    Constant refractive index in outer domain
    """

    def __eq__(self, other):
        return self.n2m1(1e8) == other.n2m1(1e8) and \
               self.n2m1(1e9) == other.n2m1(1e9) and \
               self.n2m1(1e10) == other.n2m1(1e10)

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


class EarthSurfaceBS(TransparentConstBS):

    def n2m1(self, freq_hz: int):
        return self.eta(freq_hz) - 1

    def eta(self, freq_hz: int):
        omega = 2 * cm.pi * freq_hz
        return self.permittivity(freq_hz) - 1j * self.conductivity(freq_hz) / (omega * VACUUM_PERMITTIVITY)

    def permittivity(self, freq_hz: int):
        """
        :return: relative permittivity, epsilon
        """
        pass

    def conductivity(self, freq_hz: int):
        """
        :return: conductivity, sigma (S/m)
        """
        pass


class CustomEpsSigmaBC(EarthSurfaceBS):

    def __init__(self, eps_r, sigma):
        self.eps_r = eps_r
        self.sigma = sigma

    def permittivity(self, freq_hz: int):
        return self.eps_r

    def conductivity(self, freq_hz: int):
        return self.sigma


class SaltWaterBC(EarthSurfaceBS):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 70.0
        mi = 1 - 1
        if freqMHz > 2253.5895:
            epsilon = 1.0 / (
                    MAGIC_A[mi] + MAGIC_B[mi] * freqMHz + MAGIC_C[mi] * freqMHz ** 2 + MAGIC_D[mi] * freqMHz ** 3 + MAGIC_E[mi] * freqMHz ** 4)

        return epsilon

    def conductivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        sigma = 5.0
        mi = 1 - 1
        mi1 = mi + 1

        if freqMHz > 1106.207:
            sigma = MAGIC_A[mi1] + MAGIC_C[mi1] * freqMHz + MAGIC_E[mi1] * freqMHz ** 2
            sigma = sigma / (1.0 + MAGIC_B[mi1] * freqMHz + MAGIC_D[mi1] * freqMHz ** 2 + MAGIC_F[mi1] * freqMHz ** 3)

        return sigma


class FreshWaterBC(EarthSurfaceBS):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 80.0
        mi = 3 - 1

        if freqMHz > 6165.776:
            epsilon = MAGIC_A(mi) + MAGIC_C(mi) * freqMHz + MAGIC_E[mi] * freqMHz ** 2
            epsilon = epsilon / (1.0 + MAGIC_B[mi] * freqMHz + MAGIC_D[mi] * freqMHz ** 2 + MAGIC_F[mi] * freqMHz ** 3)
        return epsilon

    def conductivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        mi = 3 - 1
        mi1 = mi + 1

        if freqMHz > 5776.157:
            ki = 2
        else:
            mi1 = mi1 + 1
            ki = -1

        sigma = MAGIC_A[mi1] + MAGIC_C[mi1] * freqMHz + MAGIC_E[mi1] * freqMHz ** 2
        sigma = (sigma / (1.0 + MAGIC_B[mi1] * freqMHz + MAGIC_D[mi1] * freqMHz ** 2)) ** ki

        return sigma


class WetGroundBC(EarthSurfaceBS):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 30.0
        mi = 6 - 1
        if freqMHz >= 4228.11:
            mi = 7 - 1

        if freqMHz > 1312.054:
            epsilon = MAGIC_A[mi] + MAGIC_C[mi] * freqMHz + MAGIC_E[mi] * freqMHz
            epsilon = cm.sqrt(epsilon / (1.0 + MAGIC_B[mi] * freqMHz + MAGIC_D[mi] * freqMHz))

        return epsilon

    def conductivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        if freqMHz > 15454.4:
            mi1 = 8 - 1
            gi = 3.3253339e-28
        else:
            mi1 = 9 - 1
            gi = 1.3854354e-25

        sigma = MAGIC_A[mi1] + MAGIC_B[mi1] * freqMHz + MAGIC_C[mi1] * freqMHz ** 2 + MAGIC_D[mi1] * freqMHz ** 3 + MAGIC_E[mi1] * freqMHz ** 4
        sigma = sigma + MAGIC_F[mi1] * freqMHz ** 5 + gi * freqMHz ** 6

        return sigma


class MediumDryGroundBC(EarthSurfaceBS):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 15.0

        if freqMHz > 4841.945:
            mi = 10 - 1
            epsilon = MAGIC_A[mi] + MAGIC_C[mi] * freqMHz + MAGIC_E[mi] * freqMHz ** 2
            epsilon = cm.sqrt(epsilon / (1.0 + MAGIC_B[mi] * freqMHz + MAGIC_D[mi] * freqMHz ** 2))

        return epsilon

    def conductivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        mi1 = 12 - 1
        if freqMHz > 4946.751:
            mi1 = 11 - 1

        sigma = (MAGIC_A[mi1] + MAGIC_B[mi1] * freqMHz + MAGIC_C[mi1] * freqMHz ** 2 + MAGIC_D[mi1] * freqMHz ** 3 + MAGIC_E[mi1] * freqMHz ** 4) ** 2

        return sigma


class VeryDryGroundBC(EarthSurfaceBS):

    def permittivity(self, freq_hz: int):
        epsilon = 3.0
        return epsilon

    def conductivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        if freqMHz < 590.8924:
            sigma = 1.0e-4
        else:
            if freqMHz > 7131.933:
                mi1 = 13 - 1
                sigma = (MAGIC_A[mi1] + MAGIC_B[mi1] * freqMHz + MAGIC_C[mi1] * freqMHz ** 2 + MAGIC_D[mi1] * freqMHz ** 3) ** 2
            else:
                mi1 = 14 - 1
                gi = 9.4623158e-23
                hi = -1.1787443e-26
                si = 7.9254217e-31
                ti = -2.2088286e-35
                sigma = MAGIC_A[mi1] + MAGIC_B[mi1] * freqMHz + MAGIC_C[mi1] * freqMHz ** 2 + MAGIC_D[mi1] * freqMHz ** 3
                sigma = sigma + MAGIC_E[mi1] * freqMHz ** 4 + MAGIC_F[mi1] * freqMHz ** 5 + gi * freqMHz ** 6
                sigma = sigma + hi * freqMHz**7 + si * freqMHz**8 + ti * freqMHz**9

        return sigma


class ImpedanceBC(BoundaryCondition):
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

    def __call__(self, wavelength: int, polarz: Polarization):
        if polarz == Polarization.HORIZONTAL:
            return 1, 0
        else:
            return 0, 1


# class EarthSurfaceBC(ImpedanceBC):
#
#     def __init__(self, permittivity, conductivity):
#         self.permittivity = permittivity
#         self.conductivity = conductivity
#
#     def __call__(self, wavelength, polarz):
#         k0 = 2 * cm.pi / wavelength
#         if polarz.upper() == 'H':
#             self.alpha1 = 1
#             self.alpha2 = 1j * k0 * (self.permittivity + 1j * 60 * self.conductivity * wavelength) ** (1 / 2)
#         else:
#              self.alpha1 = 1
#              self.alpha2 = 1j * k0 * (self.permittivity + 1j * 60 * self.conductivity * wavelength) ** (-1 / 2)
#
#         return self.alpha1, self.alpha2
#
#
# class SeaSurfaceBC(EarthSurfaceBC):
#
#     def __init__(self):
#         pass
#
#     def __call__(self, wavelength, polarz):
#         freqMHz = 2 * cm.pi * 3e8 / wavelength * 1e-6
#         epsilon = 70.0
#         sigma = 5.0  # S / m
#         mi = 1
#         mi1 = mi + 1
#
#         if freqMHz > 2253.5895:
#             epsilon = 1.0 / (a[mi] + b[mi] * freqMHz + c[mi] * freqMHz ** 2 + d[mi] * freqMHz ** 3 + e[mi] * freqMHz ** 4)
#
#         if freqMHz > 1106.207:
#             sigma = a[mi1] + c[mi1] * freqMHz + e[mi1] * freqMHz ** 2
#             sigma = sigma / (1.0 + b[mi1] * freqMHz + d[mi1] * freqMHz ** 2 + f[mi1] * freqMHz ** 3)
#
#         self.permittivity = epsilon
#         self.conductivity = sigma
#
#         return EarthSurfaceBC.__call__(self, wavelength, polarz)


class EarthAtmosphereBC(TransparentLinearBS):

    def __init__(self, Earth_radius=EARTH_RADIUS):
        TransparentLinearBS.__init__(self, 1 / Earth_radius * 1e6)


class Terrain:

    def __init__(self, func=lambda x: 0.0):
        self.terrain_func = func

    def __call__(self, x):
        return self.terrain_func(x)


class LinearTerrain(Terrain):

    def __init__(self, edge_range, edge_height):
        edge_range, edge_height = zip(*sorted(zip(edge_range, edge_height), key=itemgetter(0)))
        self.terrain_func = interp1d(edge_range, edge_height, kind='cubic', fill_value="extrapolate")


class InterpTerrain(Terrain):

    def __init__(self, edge_range, edge_height, kind='linear'):
        edge_range, edge_height = zip(*sorted(zip(edge_range, edge_height), key=itemgetter(0)))
        self.terrain_func = interp1d(edge_range, edge_height, kind=kind, fill_value="extrapolate")


class KnifeEdges(Terrain):

    def __init__(self, edge_range, edge_height):
        self.edge_range = edge_range
        self.edge_height = edge_height

    def __call__(self, x):
            return 0


class Shape:
    pass


class Point(Shape):

    def __init__(self, x, z):
        self.x = x
        self.z = z

    def intersect(self, x, z: np.ndarray):
        return np.array([])


class Box(Shape):

    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def left(self):
        return min(self.p1.x, self.p2.x)

    def right(self):
        return min(self.p1.x, self.p2.x)

    def top(self):
        return max(self.p1.y, self.p2.y)

    def bottom(self):
        return min(self.p1.y, self.p2.y)

    def intersect(self, x, z: np.ndarray):
        if self.left() <= x <= self.right():
            return np.logical_and(z >= self.bottom(), z <= self.top())


class Edge:

    def __init__(self, x, z1, z2):
        self.x = x
        self.z1 = z1
        self.z2 = z2


class Vegetation:

    def __init__(self, *, x1, x2, height, permittivity, conductivity):
        self.x1 = x1
        self.x2 = x2
        self.height = height
        self.permittivity = permittivity
        self.conductivity = conductivity

    def eta(self, freq_hz):
        omega = 2 * cm.pi * freq_hz
        return self.permittivity + 1j * self.conductivity / (omega * VACUUM_PERMITTIVITY)


class EMEnvironment:

    def __init__(self):
        self.N_profile = None
        self.terrain = Terrain()
        self.vegetation = []
        self.lower_boundary = TransparentConstBS()
        self.upper_boundary = TransparentConstBS()
        self.z_min = 0.0
        self.z_max = 100.0
        self.edges = []

    def vegetation_profile(self, x: float, z: np.ndarray, freq_hz):
        z = np.array(z)
        res = np.zeros(z.shape)*0j
        for v in self.vegetation:
            if v.x1 <= x <= v.x2:
                res[np.nonzero(np.logical_and(self.terrain(x) <= z, z <= self.terrain(x) + v.height))] = v.eta(freq_hz) - 1
                break
        return res

    def n2m1_profile(self, x: float, z: np.ndarray, freq_hz):
        if self.N_profile is None:
            return z * 0 + self.vegetation_profile(x, z, freq_hz)
        return 2 * self.N_profile(x, z) * 1e-6 + self.vegetation_profile(x, z, freq_hz)

    def n2_profile(self, x: float, z: np.ndarray):
        return self.n2m1_profile(x, z) + 1.0

    #def impediment(self, x: float, z_grid: np.ndarray):
        # if self.terrain is not None:
        #     return np.nonzero(z_grid < self.terrain(x))
        # return []


class knife_edge3d:

    def __init__(self, x1, y1, x2, y2, height):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.height = height


class EMEnvironment3d:

    def __init__(self, *, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.z_left_boundary = PECSurfaceBC()
        self.knife_edges = []


class EarthAtmosphereEnvironment(EMEnvironment):

    def __init__(self, *, boundary_condition: BoundaryCondition, height=300, Earth_radius=EARTH_RADIUS, M_profile=None):
        EMEnvironment.__init__(self)
        self.z_min = 0.0
        self.z_max = height
        self.lower_boundary = boundary_condition
        self.Earth_radius = Earth_radius
        self.M_profile = M_profile
        if self.M_profile is not None:
            self.Earth_radius = 1E6 / (self.M_profile(0, self.z_max + 1) - self.M_profile(0, self.z_max))
        self.upper_boundary = EarthAtmosphereBC(self.Earth_radius)

    def n2m1_profile(self, x: float, z: np.ndarray, freq_hz):
        if self.M_profile is None:
            return 2 * z / self.Earth_radius + self.vegetation_profile(x, z, freq_hz)
        return 2 * self.M_profile(x, z) * 1e-6 + self.vegetation_profile(x, z, freq_hz)
