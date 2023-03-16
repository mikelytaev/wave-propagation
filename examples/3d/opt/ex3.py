from utils import *


### Table 4 ######


def f1(theta_max_degrees, pade_order=(6, 7), prec=1e-2, x_max_m=1000):
    print(f"theta_max={theta_max_degrees}; x_max={x_max_m}; pade order={pade_order}")

    dx_pade_s, dz_pade_s = get_optimal(
        freq_hz=300e6,
        x_max_m=x_max_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        shift=True,
        z_order=4,
    )

    print(f'4th order: dx={dx_pade_s}; dz={dz_pade_s}')
    print(f"V={1/(dx_pade_s*(dz_pade_s**2)**2)*pade_order[1]}")
    print()


# f1(3, x_max_m=10000)
# f1(5)
# f1(10)
# f1(20)
f1(theta_max_degrees=20, pade_order=(1, 1))
f1(theta_max_degrees=20, pade_order=(2, 3))
f1(theta_max_degrees=20, pade_order=(5, 6))
f1(theta_max_degrees=20, pade_order=(6, 7))
# f1(45)
# f1(60, x_max_m=100)
# f1(70, x_max_m=100)
# f1(80, x_max_m=10)
# f1(85, x_max_m=10)