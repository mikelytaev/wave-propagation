from propagators._utils import *
from scipy.interpolate import approximate_taylor_polynomial
from scipy.interpolate import pade


def pade_propagator_coefs_m(*, pade_order, diff2, k0, dx, spe=False, alpha=0):
    if spe:
        def sqrt_1plus(x):
            return 1 + x / 2
    elif alpha == 0:
        def sqrt_1plus(x):
            return np.sqrt(1 + x)
    else:
        raise Exception('alpha not supported')

    def propagator_func(s):
        return np.exp(1j * k0 * dx * (sqrt_1plus(diff2(s)) - 1))

    taylor_coefs = approximate_taylor_polynomial(propagator_func, 0, pade_order[0] + pade_order[1] + 5, 0.01)
    p, q = pade(taylor_coefs, pade_order[0], pade_order[1])
    pade_coefs = list(zip_longest([-1 / complex(v) for v in np.roots(p)],
                                       [-1 / complex(v) for v in np.roots(q)],
                                       fillvalue=0.0j))
    return pade_coefs


coefs = pade_propagator_coefs(pade_order=(2, 2), diff2=lambda x: x, k0=2*cm.pi, dx=1)
coefs_m = pade_propagator_coefs_m(pade_order=(2, 2), diff2=lambda x: x, k0=2*cm.pi, dx=1)
print(coefs)
print(coefs_m)
# dx_res, dz_res, pade_order_res = optimal_params_m(max_angle_deg=3,
#                                                   max_distance_wl=100e3,
#                                                   threshold=1e-3,
#                                                   pade_order=(7, 8),
#                                                   z_order=4)
#
# print(dx_res)
# print(dz_res)
# print(pade_order_res)
