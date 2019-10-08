from scipy.interpolate import interp1d


class Bathymetry:

    def __init__(self, ranges_m, depths_m):
        self._func = interp1d(ranges_m, depths_m, kind='linear', fill_value=(ranges_m[0], ranges_m[-1]))
        self.max_depth = max(depths_m)
        self._ranges_m = ranges_m
        self._depth_m = depths_m

    def __call__(self, ranges_m):
        return self._func(ranges_m)

    def ranges(self):
        return self._ranges_m

    def depths(self):
        return self._depth_m


class UnderwaterEnvironment:

    def __init__(self):
        self.c0 = 1500
        self.sound_speed_profile_m_s = lambda x, z: self.c0
        self.density_profile_g_cm = lambda x, z: 1
        self.bottom_profile = Bathymetry(0, 300)
        self.bottom_sound_speed_m_s = self.c0
        self.bottom_density_g_cm = 1
