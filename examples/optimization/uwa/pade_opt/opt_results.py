from uwa.propagators import *
from uwa.vis import AcousticPressureFieldVisualiser2d
from examples.optimization.uwa.pade_opt.utils import get_optimal
import propagators._utils as utils

#logging.basicConfig(level=logging.DEBUG)


prec = 1e-3

freq_hz = 500



def func(*, max_range_m, theta_max_degrees, pade_order=(7, 8), z_order=4, c_bounds=(1500, 1500)):
    print(f"max range = {max_range_m} m; theta_max = {theta_max_degrees}")
    print(f"pade order = {pade_order}; z order = {z_order}")
    dr_m, dz_m, c0, xi_bounds = get_optimal(
        freq_hz=freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        z_order=z_order,
        c_bounds=c_bounds,
        c0=1500,
        return_meta=True
    )
    print(f"c0 = {c0}; xi = {xi_bounds}; Pade: dx = {dr_m} m; dz = {dz_m} m")
    print(f"p/(dxdz) = {pade_order[1]/dr_m/dz_m}")

    dr_m_s, dz_m_s, c0, xi_bounds = get_optimal(
        freq_hz=freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        z_order=z_order,
        c_bounds=c_bounds,
        return_meta=True
    )
    print(f"c0 = {c0}; xi = {xi_bounds}; Pade (shifted): dx = {dr_m_s} m; dz = {dz_m_s} m")
    print(f"p/(dxdz) = {pade_order[1] / dr_m_s / dz_m_s}")
    gain = (dr_m_s * dz_m_s) / (dr_m * dz_m)
    print(f"gain = {gain}")


# func(max_range_m=1000, theta_max_degrees=30)
# func(max_range_m=2000, theta_max_degrees=30)
# func(max_range_m=5000, theta_max_degrees=30)
# func(max_range_m=10000, theta_max_degrees=30)
# func(max_range_m=50000, theta_max_degrees=30)
#
#
# func(max_range_m=5000, theta_max_degrees=5)
# func(max_range_m=5000, theta_max_degrees=10)
# func(max_range_m=5000, theta_max_degrees=20)
# func(max_range_m=5000, theta_max_degrees=30)
# func(max_range_m=5000, theta_max_degrees=45)
# func(max_range_m=5000, theta_max_degrees=60)
# func(max_range_m=5000, theta_max_degrees=70)
# func(max_range_m=5000, theta_max_degrees=80)
# func(max_range_m=5000, theta_max_degrees=85)


func(max_range_m=1000, theta_max_degrees=30, pade_order=(1, 1), z_order=2)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(2, 3), z_order=2)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(3, 4), z_order=2)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(6, 7), z_order=2)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(7, 8), z_order=2)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(8, 8), z_order=2)

func(max_range_m=1000, theta_max_degrees=30, pade_order=(1, 1), z_order=4)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(2, 3), z_order=4)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(3, 4), z_order=4)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(6, 7), z_order=4)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(7, 8), z_order=4)
func(max_range_m=1000, theta_max_degrees=30, pade_order=(8, 8), z_order=4)


# func(max_range_m=1000, theta_max_degrees=30, pade_order=(7, 8), c_bounds=[1500, 1550])
# func(max_range_m=2000, theta_max_degrees=30, pade_order=(7, 8), c_bounds=[1500, 1550])
# func(max_range_m=5000, theta_max_degrees=30, pade_order=(7, 8), c_bounds=[1500, 1550])
# func(max_range_m=10000, theta_max_degrees=30, pade_order=(7, 8), c_bounds=[1500, 1550])
# func(max_range_m=50000, theta_max_degrees=30, pade_order=(7, 8), c_bounds=[1500, 1550])