from rwp.sspade import *

thetas = [3, 10, 20, 30, 45, 60, 70]
for theta in thetas:
    o2 = optimal_params_m(theta, 1, 1e-3, dx_wl=1, dz_wl=None, pade_order=(7, 8), z_order=2)
    o4 = optimal_params_m(theta, 1, 1e-3, dx_wl=1, dz_wl=None, pade_order=(7, 8), z_order=4)
    o5 = optimal_params_m(theta, 1, 1e-3, dx_wl=1, dz_wl=None, pade_order=(7, 8), z_order=5)
    print('{}, {}, {}, {}'.format(theta, o2[1], o4[1], o5[1]))