

class UnderwaterEnvironment:

    def __init__(self):
        self.c0 = 1500
        self.sound_speed_profile_m_s = lambda x, z: self.c0
        self.density_profile_g_cm = lambda x, z: 1
        self.bottom_profile = lambda x: 300
        self.bottom_sound_speed_m_s = self.c0
        self.bottom_density_g_cm = 1
