from uwa.propagators import *
from uwa.vis import AcousticPressureFieldVisualiser2d
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from examples.optimization.uwa.pade_opt.utils import get_optimal
import propagators._utils as utils

logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=1000, depth=100, beam_width=1, eval_angle=-30)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: 1500 + z
env.bottom_profile = Bathymetry(ranges_m=[0, 5000], depths_m=[500, 500])
env.bottom_sound_speed_m_s = 2000
#env.bottom_density_g_cm = 1.0
env.bottom_attenuation_dm_lambda = 0.0

max_range_m = 3000
max_depth_m = 600
prec = 1e-2

wavelength = 1500 / src.freq_hz
max_range_wl = max_range_m / wavelength

pade_order = (7, 8)

theta_max_degrees = 31
dr_s, dz_s, c0s, _ = get_optimal(
        freq_hz=src.freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        z_order=4,
        c_bounds=[1500, 2000],
        return_meta=True
    )
wl0s = c0s / src.freq_hz
print(f"Shifted: dx = {dr_s}; dz = {dz_s}")

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
    c0=c0s
)
sspe_shifted_field = sspe_shifted_propagator.calculate()
sspe_shifted_field.field *= 5.50 #normalization
sspe_shifted_vis = AcousticPressureFieldVisualiser2d(field=sspe_shifted_field, label='WPF')
sspe_shifted_vis.plot2d(-90, -5).show()


sspe_shifted_etalon_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_s / wl0s / 3,
    dz_wl=dz_s / wl0s / 3,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False,
    x_output_filter=3,
    z_output_filter=3
)

sspe_shifted_etalon_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range_m,
    max_depth_m=max_depth_m,
    comp_params=sspe_shifted_etalon_comp_params,
    c0=c0s
)
sspe_shifted_etalon_field = sspe_shifted_etalon_propagator.calculate()
sspe_shifted_etalon_field.field *= 5.50 #normalization



##

dr_sf, dz_sf, c0sf, _ = get_optimal(
        freq_hz=src.freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        z_order=4,
        c_bounds=[1500, 1500],
        return_meta=True
    )
wl0sf = c0sf / src.freq_hz
print(f"Shifted: dx = {dr_sf}; dz = {dz_sf}")

sspe_shifted_comp_params_f = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_sf / wl0sf,
    dz_wl=dz_sf / wl0sf,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_shifted_propagator_f = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range_m,
    max_depth_m=max_depth_m,
    comp_params=sspe_shifted_comp_params_f,
    c0=c0sf
)
sspe_shifted_field_f = sspe_shifted_propagator_f.calculate()
sspe_shifted_field_f.field *= 5.50 #normalization
sspe_shifted_vis_f = AcousticPressureFieldVisualiser2d(field=sspe_shifted_field_f, label='WPF')
sspe_shifted_vis_f.plot2d(-90, -5).show()


f, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 3.2), constrained_layout=True)
extent = [sspe_shifted_vis.field.x_grid[0]*1e-3, sspe_shifted_vis.field.x_grid[-1]*1e-3, sspe_shifted_vis.field.z_grid[-1], sspe_shifted_vis.field.z_grid[0]]
norm = Normalize(-90, -5)
im = ax[0].imshow(20*np.log10(abs(sspe_shifted_vis.field.field.T)), extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[0].grid()
ax[0].set_title(" ")
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Depth (m)')

im = ax[1].imshow(20*np.log10(abs(sspe_shifted_vis_f.field.field.T)), extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1].grid()
ax[1].set_title(" ")
ax[1].set_xlabel('Range (km)')
ax[1].set_ylabel('Depth (m)')

f.colorbar(im, ax=ax[:], shrink=0.6, location='bottom')
#plt.show()
plt.savefig("ex2_shifted_2d.eps")

#####
dr, dz, c0ns, _ = get_optimal(
        freq_hz=src.freq_hz,
        x_max_m=max_range_m,
        prec=prec,
        theta_max_degrees=theta_max_degrees,
        pade_order=pade_order,
        z_order=4,
        c_bounds=[1500, 2000],
        c0=1500,
        return_meta=True
    )
print(f"Pade: dx = {dr}; dz = {dz}")

wl0 = c0ns / src.freq_hz
print(f"Pade: dx = {dr}; dz = {dz}")

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
    c0=1500
)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-90, -5).show()


sspe_etalon_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr / wl0 / 2,
    dz_wl=dz / wl0 / 2,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False,
    x_output_filter=2,
    z_output_filter=2
)

sspe_etalon_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range_m,
    max_depth_m=max_depth_m,
    comp_params=sspe_etalon_comp_params,
    c0=1500
)
sspe_etalon_field = sspe_etalon_propagator.calculate()
sspe_etalon_field.field *= 5.50 #normalization



f, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 3.2), constrained_layout=True)
extent = [sspe_shifted_vis.field.x_grid[0]*1e-3, sspe_shifted_vis.field.x_grid[-1]*1e-3, sspe_shifted_vis.field.z_grid[-1], sspe_shifted_vis.field.z_grid[0]]
norm = Normalize(-30, 0)
im = ax[0].imshow(10*np.log10(0.4*abs(sspe_shifted_field.field.T[:-1:,:] - sspe_shifted_etalon_field.field.T[::, ::])), extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[0].grid()
ax[0].set_title("c0=1806 m/s")
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Depth (m)')

im = ax[1].imshow(10*np.log10(40*abs(sspe_field.field.T[:,:-1:] - sspe_etalon_field.field.T[::, ::])), extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[1].grid()
ax[1].set_title("c0=1500 m/s")
ax[1].set_xlabel('Range (km)')
ax[1].set_ylabel('Depth (m)')

f.colorbar(im, ax=ax[:], shrink=0.6, location='bottom')
#plt.show()
plt.savefig("ex2_shifted_2d_error.eps")