from scipy.interpolate import interp1d
from dataclasses import dataclass
import numpy as np


class Bathymetry:

    def __init__(self, ranges_m=None, depths_m=None, func=None, max_depth=None):
        if func is not None:
            self._func = func
        else:
            if len(ranges_m) == 1:
                self._func = lambda x: depths_m[0]
            else:
                self._func = interp1d(ranges_m, depths_m, kind='linear', bounds_error=False, fill_value=(depths_m[0], depths_m[-1]))
                self._ranges_m = ranges_m
                self._depth_m = depths_m
        self._max_depth = max_depth or max(depths_m)

    def __call__(self, ranges_m):
        return self._func(ranges_m)

    def ranges(self):
        return self._ranges_m

    def depths(self):
        return self._depth_m

    def max_depth(self):
        return self._max_depth


@dataclass
class UnderwaterEnvironment:
    bottom_sound_speed_m_s: float = 1500
    sound_speed_profile_m_s: "function" = lambda x, z: 1500 + z*0
    bottom_profile: "function" = Bathymetry(ranges_m=[0], depths_m=[300])
    bottom_density_g_cm: float = 1
    bottom_attenuation_dm_lambda: float = 0.0

def munk_profile(z_grid_m, ref_sound_speed=1500, ref_depth=1300, eps_=0.00737):
    z_ = 2 * (z_grid_m - ref_depth) / ref_depth
    return ref_sound_speed * (1 + eps_ * (z_ - 1 + np.exp(-z_)))


def sound_speed_mps(z_m, temperature_deg_cent, salinity_ppt):
    z = z_m
    t = temperature_deg_cent
    s = salinity_ppt
    return 1449.2 + 4.6*t - 0.055*t**2 + 0.00029*t**3 + (1.34-0.01*t)*(s-35) + 0.016*t
