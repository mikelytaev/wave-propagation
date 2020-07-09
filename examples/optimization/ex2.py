from propagators._utils import *

angles = [3, 10, 20, 45, 80, 85]
pade_orders = [(1, 1), (1, 2), (3, 4), (7, 8)]
x_max = 20000
z_max = 200
vals = np.empty((len(angles), len(pade_orders)))

for a_i, angle in enumerate(angles):
    for p_i, pade_order in enumerate(pade_orders):
        (dx_res, dz_res, pade_order_res) = optimal_params_m(max_angle_deg=angle, max_distance_wl=x_max, threshold=1e-3, pade_order=pade_order, z_order=4)
        if dx_res is None or dz_res is None or pade_order_res is None:
            val = np.inf
        else:
            val = x_max / dx_res * z_max / dz_res * pade_order[1]
        print("angle=", angle, "pade_order=", pade_order, "dx=", dx_res, "dz=", dz_res, "val=", val)
        vals[a_i, p_i] = val

print(vals)

vals_p = np.empty((len(angles), 1))
for a_i, angle in enumerate(angles):
    pade_order = (7, 8)
    (dx_res, dz_res, pade_order_res) = optimal_params_m(max_angle_deg=angle, max_distance_wl=x_max, threshold=1e-3, pade_order=pade_order, z_order=5)
    if dx_res is None or dz_res is None or pade_order_res is None:
        val = np.inf
    else:
        val = x_max / dx_res * z_max / dz_res * pade_order[1]
    print("angle=", angle, "pade_order=", pade_order, "dx=", dx_res, "dz=", dz_res, "val=", val)
    vals_p[a_i, 0] = val

print(vals_p)