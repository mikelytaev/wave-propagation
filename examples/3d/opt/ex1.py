from utils import *


### Table 1 ######

pade_order = (6, 7)


def f1(theta_max_degrees, prec=1e-2, x_max_m=1000):
    print(f"theta_max={theta_max_degrees}; x_max={x_max_m}")
    dx_pade, dz_pade = get_optimal(
        freq_hz=300e6,
        x_max_m=x_max_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
    )

    print(f'No shift: dx={dx_pade}; dz={dz_pade}')

    dx_pade_s, dz_pade_s = get_optimal(
        freq_hz=300e6,
        x_max_m=x_max_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        shift=True,
        dxs_m=np.linspace(dx_pade, 3*dx_pade, 10) if dx_pade > 1e-10 else None
    )

    print(f'Shift:    dx={dx_pade_s}; dz={dz_pade_s}')
    print(f"Gain={dx_pade_s*(dz_pade_s**2)**2/(dx_pade*(dz_pade**2)**2)}")
    print()


f1(3, x_max_m=10000)
f1(5)
f1(10)
f1(20)
f1(30)
f1(45)
f1(60, x_max_m=100)
f1(70, x_max_m=100)
f1(80, x_max_m=10)
f1(85, x_max_m=10)