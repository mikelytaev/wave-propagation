from utils import *


### Table 2, 3 ######

pade_order = (6, 7)


def f1(theta_max_degrees, prec=1e-2, x_max_m=1000):
    print(f"theta_max={theta_max_degrees}; x_max={x_max_m}")
    dx_adi_pade_4th, dz_adi_pade_4th = get_optimal(
        freq_hz=300e6,
        x_max_m=x_max_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        shift=True,
        adi=True
    )

    print(f'ADI 4th: dx={dx_adi_pade_4th}; dz={dz_adi_pade_4th}')

    dx_adi_pade_2nd, dz_adi_pade_2nd = get_optimal(
        freq_hz=300e6,
        x_max_m=x_max_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        shift=True,
        adi=True,
        z_order=2
    )

    print(f'ADI 2nd: dx={dx_adi_pade_2nd}; dz={dz_adi_pade_2nd}')
    print(f"Gain={dx_adi_pade_4th * (dz_adi_pade_4th ** 2) / (dx_adi_pade_2nd * (dz_adi_pade_2nd ** 2))}")

    dx_cn_pade_4th, dz_cn_pade_4th = get_optimal(
        freq_hz=300e6,
        x_max_m=x_max_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        shift=True,
        adi=False
    )

    print(f'CN 4th: dx={dx_cn_pade_4th}; dz={dz_cn_pade_4th}')

    dx_cn_pade_2nd, dz_cn_pade_2nd = get_optimal(
        freq_hz=300e6,
        x_max_m=x_max_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        shift=True,
        adi=False,
        z_order=2
    )

    print(f'CN 2nd: dx={dx_cn_pade_2nd}; dz={dz_cn_pade_2nd}')
    print(f"Gain={dx_cn_pade_4th * (dz_cn_pade_4th ** 2)**2 / (dx_cn_pade_2nd * (dz_cn_pade_2nd ** 2)**2)}")
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