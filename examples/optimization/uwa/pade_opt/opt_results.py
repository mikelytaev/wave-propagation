from uwa.propagators import *
from uwa.vis import AcousticPressureFieldVisualiser2d
from examples.optimization.uwa.pade_opt.utils import get_optimal
import propagators._utils as utils

#logging.basicConfig(level=logging.DEBUG)


prec = 1e-3


pade_order = (7, 8)

freq_hz = 500

c_min = 1500
c_max = 1500


def func(max_range_m, theta_max_degrees):
    print(f"max range = {max_range_m} m; theta_max = {theta_max_degrees}")
    dr_wl, dz_wl, c0, xi_bounds = get_optimal(
        freq_hz=freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        c_bounds=[c_min, c_max],
        c0=1500,
        return_meta=True
    )
    print(f"c0 = {c0}; xi = {xi_bounds}; Pade: dx = {dr_wl} m; dz = {dz_wl} m")

    dr_wl_s, dz_wl_s, c0, xi_bounds = get_optimal(
        freq_hz=freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        c_bounds=[c_min, c_max],
        return_meta=True
    )
    print(f"c0 = {c0}; xi = {xi_bounds}; Pade (shifted): dx = {dr_wl_s} m; dz = {dz_wl_s} m")
    gain = (dr_wl_s * dz_wl_s) / (dr_wl * dz_wl)
    print(f"gain = {gain}")


func(max_range_m=1000, theta_max_degrees=30)
func(max_range_m=2000, theta_max_degrees=30)
func(max_range_m=5000, theta_max_degrees=30)
func(max_range_m=10000, theta_max_degrees=30)
func(max_range_m=50000, theta_max_degrees=30)


func(max_range_m=5000, theta_max_degrees=5)
func(max_range_m=5000, theta_max_degrees=10)
func(max_range_m=5000, theta_max_degrees=20)
func(max_range_m=5000, theta_max_degrees=30)
func(max_range_m=5000, theta_max_degrees=45)
func(max_range_m=5000, theta_max_degrees=60)
func(max_range_m=5000, theta_max_degrees=70)
func(max_range_m=5000, theta_max_degrees=80)
func(max_range_m=5000, theta_max_degrees=85)