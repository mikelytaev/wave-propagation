import numpy as np


def munk_profile(z_grid_m, ref_sound_speed=1500, ref_depth=1300, eps_=0.00737):
    z_ = 2 * (z_grid_m - ref_depth) / ref_depth
    return ref_sound_speed * (1 + eps_ * (z_ - 1 + np.exp(-z_)))