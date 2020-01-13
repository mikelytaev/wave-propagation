from propagators._utils import *


def optimal_params_m(max_angle_deg, max_distance_wl, threshold, dx=None, dz=None, pade_order=None, z_order=4):
    k0 = 2*cm.pi
    res = (None, None, None)
    cur_min = 1e100

    if pade_order:
        pade_orders = [pade_order]
    else:
        pade_orders = [(7, 8), (6, 7), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (1, 1)]

    if dx:
        dxs = [dx]
    else:
        dxs = np.concatenate((np.linspace(0.001, 0.01, 10),
                              np.linspace(0.02, 0.1, 9),
                              np.linspace(0.2, 1, 9),
                              np.linspace(2, 10, 9),
                              np.linspace(20, 100, 9),
                              np.linspace(200, 1000, 9),
                              np.linspace(2000, 10000, 9)))

    if dz:
        dzs = [dz]
    else:
        dzs = np.concatenate((np.array([0.001, 0.005]), np.array([0.01, 0.05]), np.linspace(0.1, 4, 40)))

    dxs.sort()
    dzs.sort()
    for pade_order in pade_orders:
        for dx in dxs:
            if z_order <= 4:
                coefs = pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0, dx=dx, spe=False)
            for dz in dzs:
                if z_order > 4:
                    coefs = pade_propagator_coefs(pade_order=pade_order, diff2=lambda s: mpmath.acosh(1 + (k0 * dz) ** 2 * s / 2) ** 2 / (k0 * dz) ** 2, k0=k0, dx=dx, spe=False)

                errors = []
                for al in np.linspace(0, max_angle_deg, 5):
                    kz = k0 * cm.sin(al * cm.pi / 180)
                    if z_order <= 4:
                        d_kx = discrete_k_x(k0, dx, coefs, dz, kz, order=z_order)
                    else:
                        d_kx = discrete_k_x(k0, dx, coefs, dz, kz, order=2)
                    errors += [abs(cm.sqrt(k0 ** 2 - kz ** 2) - d_kx)]

                val = pade_order[1] / (dx * dz)
                error = max(errors)

                # if error >= threshold * dx / max_distance_wl:
                #     break

                if error < threshold * dx / max_distance_wl and val < cur_min:
                    res = (dx, dz, pade_order)
                    cur_min = val

    return res

angles = [3, 10, 20, 45, 80, 85]
pade_orders = [(1,1), (1,2), (3,4), (7,8)]
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
    pade_order = (7,8)
    (dx_res, dz_res, pade_order_res) = optimal_params(max_angle=angle, threshold=1e-3, pade_order=pade_order, z_order=5)
    if dx_res is None or dz_res is None or pade_order_res is None:
        val = np.inf
    else:
        val = x_max / dx_res * z_max / dz_res * pade_order[1]
    print("angle=", angle, "pade_order=", pade_order, "dx=", dx_res, "dz=", dz_res, "val=", val)
    vals_p[a_i, 0] = val

print(vals_p)