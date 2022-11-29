from uwa.propagators import *
from uwa.vis import AcousticPressureFieldVisualiser2d
from examples.optimization.uwa.pade_opt.utils import get_optimal
import propagators._utils as utils

#logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=1000, depth=50, beam_width=1, eval_angle=-30)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: 1500 + z/2*0
env.bottom_profile = Bathymetry(ranges_m=[0, 5000], depths_m=[200, 200])
env.bottom_sound_speed_m_s = 1700
#env.bottom_density_g_cm = 1.0
env.bottom_attenuation_dm_lambda = 0.0

max_range = 3000
prec = 1e-1

wavelength = 1500 / src.freq_hz
max_range_wl = max_range / wavelength

pade_order = (8, 8)

k0 = 2*fm.pi
theta_max_degrees = 33
k_z_max = k0*fm.sin(theta_max_degrees*fm.pi/180)
xi_bounds = [-k_z_max**2/k0**2-0.23*0, 0]

dr_wl_s, dz_wl_s = get_optimal(max_range_wl, prec, xi_bounds[0], k_z_max, pade_order=pade_order, shift_pade=True)
#dr_wl_s *= 2
print(f"Shifted: dx = {dr_wl_s}; dz = {dz_wl_s}")

pade_coefs, c0 = utils.pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=2 * cm.pi, dx=dr_wl_s,
                                             a0=(xi_bounds[0]+xi_bounds[1])/2)

sspe_shifted_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl_s,
    dz_wl=dz_wl_s,
    exp_pade_order=pade_order,
    exp_pade_coefs=pade_coefs,
    exp_pade_a0_coef=c0,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_shifted_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range,
    max_depth_m=300,
    comp_params=sspe_shifted_comp_params
)
sspe_shifted_field = sspe_shifted_propagator.calculate()
sspe_shifted_field.field *= 5.50 #normalization
sspe_shifted_vis = AcousticPressureFieldVisualiser2d(field=sspe_shifted_field, label='WPF')
sspe_shifted_vis.plot2d(-60, -5).show()

#####

dr_wl, dz_wl = get_optimal(max_range_wl, prec, xi_bounds[0], k_z_max, pade_order=pade_order, shift_pade=False)
#dr_wl, dz_wl = dr_wl_s, dz_wl_s
print(f"Pade: dx = {dr_wl}; dz = {dz_wl}")

sspe_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl,
    dz_wl=dz_wl,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range,
    max_depth_m=300,
    comp_params=sspe_comp_params
)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-60, -5).show()

######

sspe_f_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl_s,
    dz_wl=dz_wl_s,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_f_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range,
    max_depth_m=300,
    comp_params=sspe_f_comp_params
)
sspe_f_field = sspe_f_propagator.calculate()
sspe_f_field.field *= 5.50 #normalization
sspe_f_vis = AcousticPressureFieldVisualiser2d(field=sspe_f_field, label='WPF')
sspe_f_vis.plot2d(-60, -5).show()