import cmath as cm
import math as fm
from enum import Enum

import numpy as np
from scipy.interpolate import interp1d
from operator import itemgetter

EARTH_RADIUS = 6371000

VACUUM_PERMITTIVITY = 8.854187817e-12
LIGHT_SPEED = 3e8

# магические константы из петула. TODO: разобраться откуда это вообще

_MAGIC_A = [1.4114535e-2, 3.8586749, 79.027635, -0.65750351, 201.97103, 857.94335, 915.31026, 0.8756665, 5.5990969e-3,
            215.87521, 0.17381269, 2.4625032e-2, -4.9560275e-2, 2.2953743e-4, 0.000038814567, 1.2434792e-4, 51852.543,
            4.13105e-5]

_MAGIC_B = [-5.2122497e-8, -2.1179295e-5, -2.2083308e-5, 5.5620223e-5, -2.5539582e-3, -8.9983662e-5, -9.4530022e-6,
            4.7236085e-5, 8.7798277e-5, -7.6649237e-5, 1.2655183e-4, 1.8254018e-4, 2.9876572e-5, -8.1212741e-7, 8.467523e-2,
            2.824598e-4, 3.883854e-2, 2.03589e-7]

_MAGIC_C = [5.8547829e-11, 9.1253873e-4, -3.5486605e-4, 6.6113198e-4, 1.2197967e-2, 5.5275278e-2, -4.0348211e-3, 2.6051966e-8,
            6.2451017e-8, -2.6151055e-3, -1.6790756e-9, -2.664754e-8, -3.0561848e-10, 1.8045461e-9, 9.878241e-6, 8.680839e-7,
            389.58894, -3.1739e-12]

_MAGIC_D = [-7.6717423e-16, 6.5727504e-10, 2.7067836e-9, 3.0140816 - 10, 3.7853169e-5, 8.8247139e-8, 4.892281e-8, -9.235936e-13,
            -7.1317207e-12, 1.2565999e-8, 1.1037608e-14, 7.6508732e-12, 1.1131828e-15, -1.960677e-12, -9.736703e-5, -6.755389e-8,
            6.832108e-5, 4.52331e-17]

_MAGIC_E = [2.9856318e-21, 1.5309921e-8, 8.210184e-9, 1.4876952e-9, -1.728776e-6, 0.0, 7.4342897e-7, 1.4560078e-17, 4.2515914e-16,
            1.9484482e-7, -2.9223433e-20, -7.4193268e-16, 0.0, 1.2569594e-15, 7.990284e-8, 7.2701689e-11, 0.0, 0.0]

_MAGIC_F = [0.0, -1.9647664e-15, -1.0007669e-14, 0.0, 0.0, 0.0, 0.0, -1.1129348e-22, -1.240806e-20, 0.0, 0.0, 0.0, 0.0, -4.46811e-19,
            3.269059e-7, 2.8728975e-12, 0.0, 0.0]


class Polarization(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class Material:

    def permittivity(self, freq_hz: int):
        pass

    def conductivity_sm_m(self, freq_hz: int):
        pass

    def complex_permittivity(self, freq_hz):
        eps = self.permittivity(freq_hz)
        sigma = self.conductivity_sm_m(freq_hz)
        wavelength = 3e8 / freq_hz
        return eps + 1j * 60 * sigma * wavelength


class CustomMaterial(Material):

    def __init__(self, eps, sigma):
        self.eps = eps
        self.sigma = sigma

    def permittivity(self, freq_hz: int):
        return self.eps

    def conductivity_sm_m(self, freq_hz: int):
        return self.sigma


class PerfectlyElectricConducting(Material):

    def permittivity(self, freq_hz: int):
        return complex("Inf")

    def conductivity_sm_m(self, freq_hz: int):
        return complex("Inf")


class Air(Material):

    def permittivity(self, freq_hz: int):
        return 1

    def conductivity_sm_m(self, freq_hz: int):
        return 0


class SaltWater(Material):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 70.0
        mi = 1 - 1
        if freqMHz > 2253.5895:
            epsilon = 1.0 / (
                    _MAGIC_A[mi] + _MAGIC_B[mi] * freqMHz + _MAGIC_C[mi] * freqMHz ** 2 + _MAGIC_D[mi] * freqMHz ** 3 + _MAGIC_E[mi] * freqMHz ** 4)

        return epsilon

    def conductivity_sm_m(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        sigma = 5.0
        mi = 1 - 1
        mi1 = mi + 1

        if freqMHz > 1106.207:
            sigma = _MAGIC_A[mi1] + _MAGIC_C[mi1] * freqMHz + _MAGIC_E[mi1] * freqMHz ** 2
            sigma = sigma / (1.0 + _MAGIC_B[mi1] * freqMHz + _MAGIC_D[mi1] * freqMHz ** 2 + _MAGIC_F[mi1] * freqMHz ** 3)

        return sigma


class FreshWater(Material):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 80.0
        mi = 3 - 1

        if freqMHz > 6165.776:
            epsilon = _MAGIC_A[mi] + _MAGIC_C[mi] * freqMHz + _MAGIC_E[mi] * freqMHz ** 2
            epsilon = epsilon / (1.0 + _MAGIC_B[mi] * freqMHz + _MAGIC_D[mi] * freqMHz ** 2 + _MAGIC_F[mi] * freqMHz ** 3)
        return epsilon

    def conductivity_sm_m(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        mi = 3 - 1
        mi1 = mi + 1

        if freqMHz > 5776.157:
            ki = 2
        else:
            mi1 = mi1 + 1
            ki = -1

        sigma = _MAGIC_A[mi1] + _MAGIC_C[mi1] * freqMHz + _MAGIC_E[mi1] * freqMHz ** 2
        sigma = (sigma / (1.0 + _MAGIC_B[mi1] * freqMHz + _MAGIC_D[mi1] * freqMHz ** 2)) ** ki

        return sigma


class WetGround(Material):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 30.0
        mi = 6 - 1
        if freqMHz >= 4228.11:
            mi = 7 - 1

        if freqMHz > 1312.054:
            epsilon = _MAGIC_A[mi] + _MAGIC_C[mi] * freqMHz + _MAGIC_E[mi] * freqMHz**2
            epsilon = cm.sqrt(epsilon / (1.0 + _MAGIC_B[mi] * freqMHz + _MAGIC_D[mi] * freqMHz**2))

        return epsilon

    def conductivity_sm_m(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        if freqMHz > 15454.4:
            mi1 = 8 - 1
            gi = 3.3253339e-28
        else:
            mi1 = 9 - 1
            gi = 1.3854354e-25

        sigma = _MAGIC_A[mi1] + _MAGIC_B[mi1] * freqMHz + _MAGIC_C[mi1] * freqMHz ** 2 + _MAGIC_D[mi1] * freqMHz ** 3 + _MAGIC_E[mi1] * freqMHz ** 4
        sigma = sigma + _MAGIC_F[mi1] * freqMHz ** 5 + gi * freqMHz ** 6

        return sigma


class MediumDryGround(Material):

    def permittivity(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        epsilon = 15.0

        if freqMHz > 4841.945:
            mi = 10 - 1
            epsilon = _MAGIC_A[mi] + _MAGIC_C[mi] * freqMHz + _MAGIC_E[mi] * freqMHz ** 2
            epsilon = cm.sqrt(epsilon / (1.0 + _MAGIC_B[mi] * freqMHz + _MAGIC_D[mi] * freqMHz ** 2))

        return epsilon

    def conductivity_sm_m(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        mi1 = 12 - 1
        if freqMHz > 4946.751:
            mi1 = 11 - 1

        sigma = (_MAGIC_A[mi1] + _MAGIC_B[mi1] * freqMHz + _MAGIC_C[mi1] * freqMHz ** 2 + _MAGIC_D[mi1] * freqMHz ** 3 + _MAGIC_E[mi1] * freqMHz ** 4) ** 2

        return sigma


class VeryDryGround(Material):

    def permittivity(self, freq_hz: int):
        epsilon = 3.0
        return epsilon

    def conductivity_sm_m(self, freq_hz: int):
        freqMHz = freq_hz * 1e-6
        if freqMHz < 590.8924:
            sigma = 1.0e-4
        else:
            if freqMHz > 7131.933:
                mi1 = 13 - 1
                sigma = (_MAGIC_A[mi1] + _MAGIC_B[mi1] * freqMHz + _MAGIC_C[mi1] * freqMHz ** 2 + _MAGIC_D[mi1] * freqMHz ** 3) ** 2
            else:
                mi1 = 14 - 1
                gi = 9.4623158e-23
                hi = -1.1787443e-26
                si = 7.9254217e-31
                ti = -2.2088286e-35
                sigma = _MAGIC_A[mi1] + _MAGIC_B[mi1] * freqMHz + _MAGIC_C[mi1] * freqMHz ** 2 + _MAGIC_D[mi1] * freqMHz ** 3
                sigma = sigma + _MAGIC_E[mi1] * freqMHz ** 4 + _MAGIC_F[mi1] * freqMHz ** 5 + gi * freqMHz ** 6
                sigma = sigma + hi * freqMHz**7 + si * freqMHz**8 + ti * freqMHz**9

        return sigma


class Terrain:

    def __init__(self, elevation=None, ground_material=PerfectlyElectricConducting()):
        """
        :param elevation: function range (m) -> elevation (m)
        :param ground_material: Material or function range (m) -> Material
        """
        if elevation is None:
            self.is_homogeneous = True
            self.elevation = lambda x: 0.0
        else:
            self.is_homogeneous = False
            self.elevation = elevation

        if isinstance(ground_material, Material):
            self.ground_material = lambda x: ground_material
            self.is_range_dependent_ground_material = False
        elif callable(ground_material):
            self.ground_material = ground_material
            self.is_range_dependent_ground_material = True


class InterpTerrain(Terrain):

    def __init__(self, edge_range, edge_height, *, kind='linear', ground_material):
        edge_range, edge_height = zip(*sorted(zip(edge_range, edge_height), key=itemgetter(0)))
        terrain_func = interp1d(edge_range, edge_height, kind=kind, fill_value="extrapolate")
        super().__init__(elevation=terrain_func, ground_material=ground_material)


class KnifeEdge:

    def __init__(self, range, height):
        self.range = range
        self.height = height


class Impediment:

    def __init__(self, *, left_m, right_m, height_m, material: Material):
        self.x1 = left_m
        self.x2 = right_m
        self.height = height_m
        self.material = material


class Troposphere:

    def __init__(self, flat=False):
        self.M_profile = None
        self.terrain = Terrain()
        self.vegetation = []
        self.knife_edges = []
        self.is_flat = flat
        self.rms_m = None
        if flat:
            self.Earth_radius = float("Inf")
        else:
            self.Earth_radius = EARTH_RADIUS

    def vegetation_profile(self, x: float, z: np.ndarray, freq_hz):
        if hasattr(z, "__len__"):
            res = np.zeros(z.shape)*0j
            for v in self.vegetation:
                if v.x1 <= x <= v.x2:
                    res[np.nonzero(np.logical_and(self.terrain.elevation(x) <= z, z <= self.terrain.elevation(x) + v.height))] = v.material.complex_permittivity(freq_hz) - 1
                    break
            return res
        else:
            for v in self.vegetation:
                if v.x1 <= x <= v.x2:
                    if self.terrain.elevation(x) <= z <= self.terrain.elevation(x) + v.height:
                        return v.material.complex_permittivity(freq_hz) - 1
                    break
            return 0

    def n2m1_profile(self, x: float, z: np.ndarray, freq_hz):
        if self.M_profile is None:
            return 2 * z / self.Earth_radius + self.vegetation_profile(x, z, freq_hz)
        return 2 * self.M_profile(x, z) * 1e-6 + self.vegetation_profile(x, z, freq_hz)

    def is_homogeneous(self):
        return (self.Earth_radius is None or self.Earth_radius == float("Inf")) \
               and len(self.vegetation) == 0 and \
               self.M_profile is None


class KnifeEdge3d:

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
        #self.z_left_boundary = PECSurfaceBC()
        self.knife_edges = []


class StreetCanyon3D:

    def __init__(self, *, domain_width, domain_height, street_width, building_height, x_max):
        self.domain_width = domain_width
        self.domain_height = domain_height
        self.street_width = street_width
        self.building_height = building_height
        self.x_max = x_max


class Manhattan3D:

    def __init__(self, *, domain_width, domain_height, x_max):
        self.domain_width = domain_width
        self.domain_height = domain_height
        self.x_max = x_max

        self.centers = []
        self.sizes = []

    def add_block(self, *, center: '(x, y)', size: '(x_size, y_size, height)'):
        self.centers += [center]
        self.sizes += [size]

    def intersection_mask_x(self, x_m, y_grid_m, z_grid_m):
        res = np.zeros((len(y_grid_m), len(z_grid_m)), dtype=bool)
        y_mg, z_mg = np.meshgrid(y_grid_m, z_grid_m, indexing='ij')
        for center, size in zip(self.centers, self.sizes):
            x_left = center[0] - size[0] / 2
            x_right = center[0] + size[0] / 2
            if x_left <= x_m <= x_right:
                y_left = center[1] - size[1] / 2
                y_right = center[1] + size[1] / 2
                height = size[2]
                res = np.logical_or(res, np.logical_and(z_mg <= height, np.logical_and(y_left <= y_mg, y_mg <= y_right)))
        return res

    def facets(self, x_grid_m, y_grid_m, z_grid_m, forward=True):
        res = []
        sign = 1 if forward else -1
        for center, size in zip(self.centers, self.sizes):
            x_pos = center[0] + sign * size[0] / 2
            x_index = np.abs(x_grid_m - x_pos).argmin()
            mask = self.intersection_mask_x(x_pos - sign * 1e-10, y_grid_m, z_grid_m)
            res += [(x_index, mask)]
        return res


def pyramid(x, angle, height, r):
    length = height / fm.tan(angle * cm.pi / 180)
    if r <= x <= r + length:
        return (x - r) * fm.tan(angle * cm.pi / 180)
    elif r + length < x <= r + 2*length:
        return (r + 2*length - x) * fm.tan(angle * cm.pi / 180)
    else:
        return 0


def pyramid2(x, angle, height, center):
    length = height / fm.tan(angle * cm.pi / 180)
    r = center - length
    if center - length <= x <= center:
        return (x - r) * fm.tan(angle * cm.pi / 180)
    elif center < x <= center + length:
        return (r + 2*length - x) * fm.tan(angle * cm.pi / 180)
    else:
        return 0


def gauss_hill_func(height_m, length_m, x0_m):
    w = length_m / 2
    return lambda x: height_m / 2 * (1 + fm.cos(fm.pi * (x - x0_m) / w)) \
        if -w <= x - x0_m <= w else 0


def evaporation_duct(height, z_grid_m, m_0=320, z_0=1.5e4):
    z_grid_m = z_grid_m + 0.001
    return m_0 + 0.125*(z_grid_m - height*np.log10(z_grid_m / z_0))