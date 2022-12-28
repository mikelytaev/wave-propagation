from uwa.propagators import *
from uwa.vis import AcousticPressureFieldVisualiser2d
from examples.optimization.uwa.pade_opt.utils import get_optimal
import propagators._utils as utils

#logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=1000, depth=100, beam_width=1, eval_angle=-30)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: 1500 + z*0
env.bottom_profile = Bathymetry(ranges_m=[0, 5000], depths_m=[300, 300])
env.bottom_sound_speed_m_s = 1500
#env.bottom_density_g_cm = 1.0
env.bottom_attenuation_dm_lambda = 0.0

max_range_m = 3000
max_depth_m = 300
prec = 1e-2

wavelength = 1500 / src.freq_hz
max_range_wl = max_range_m / wavelength

pade_order = (7, 8)

k0 = 2*fm.pi
theta_max_degrees = 31
k_z_max = k0*fm.sin(theta_max_degrees*fm.pi/180)
dr_s, dz_s, c0s, _ = get_optimal(
        freq_hz=src.freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        z_order=4,
        c_bounds=[1500, 1500],
        return_meta=True
    )
dr_s *= 1.2
print(f"Shifted: dx = {dr_s}; dz = {dz_s}")

wl0s = c0s / src.freq_hz

sspe_shifted_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_s / wl0s,
    dz_wl=dz_s / wl0s,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_shifted_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range_m,
    max_depth_m=max_depth_m,
    comp_params=sspe_shifted_comp_params,
    c0=1590,
    lower_bc=RobinBC(q1=0, q2=1, q3=0)
)
sspe_shifted_field = sspe_shifted_propagator.calculate()
sspe_shifted_field.field *= 5.50 #normalization
sspe_shifted_vis = AcousticPressureFieldVisualiser2d(field=sspe_shifted_field, label='WPF')
sspe_shifted_vis.plot2d(-50, -5).show()

#####

xi_bounds = [-k_z_max**2/k0**2, 0]
dr, dz, c0ns, _ = get_optimal(
        freq_hz=src.freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        z_order=4,
        c_bounds=[1500, 1500],
        c0=1500,
        return_meta=True
    )
print(f"Pade: dx = {dr}; dz = {dz}")

wl0 = c0ns / src.freq_hz

sspe_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr / wl0,
    dz_wl=dz / wl0,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range_m,
    max_depth_m=max_depth_m,
    comp_params=sspe_comp_params,
    c0=1500,
    lower_bc=RobinBC(q1=0, q2=1, q3=0)
)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-50, -5).show()

######

sspe_f_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_s / wl0s,
    dz_wl=dz_s / wl0s,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_f_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range_m,
    max_depth_m=max_depth_m,
    comp_params=sspe_f_comp_params,
    c0=1500,
    lower_bc=RobinBC(q1=0, q2=1, q3=0)
)
sspe_f_field = sspe_f_propagator.calculate()
sspe_f_field.field *= 5.50 #normalization
sspe_f_vis = AcousticPressureFieldVisualiser2d(field=sspe_f_field, label='WPF')
sspe_f_vis.plot2d(-50, -5).show()