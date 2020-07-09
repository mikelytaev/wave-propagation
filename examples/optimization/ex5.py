from propagators._utils import *


dx_res, dz_res, pade_order_res = optimal_params_m(max_angle_deg=3,
                                                  max_distance_wl=100e3,
                                                  threshold=1e-3,
                                                  pade_order=(7, 8),
                                                  z_order=4)

print(dx_res)
print(dz_res)
print(pade_order_res)