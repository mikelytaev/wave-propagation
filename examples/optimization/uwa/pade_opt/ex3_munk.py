from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.propagators import *
from uwa.ram import *
from uwa.utils import *
from examples.optimization.uwa.pade_opt.utils import get_optimal
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from uwa.vis import AcousticPressureFieldVisualiser2d


logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=50, depth=100, beam_width=10, eval_angle=0)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: munk_profile(z)
env.bottom_profile = Bathymetry(ranges_m=[0], depths_m=[5000])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1.5
env.bottom_attenuation_dm_lambda = 0.5

c_min = np.min(munk_profile(np.linspace(0, 5000, 5000)))
c_max = np.max(munk_profile(np.linspace(0, 5000, 5000)))

print(f"c_min = {c_min}; c_max = {c_max}")

max_range = 150000

pade_order = (7, 8)
dr_s, dz_s, c0s, _ = get_optimal(
        freq_hz=src.freq_hz,
        x_max_m=max_range,
        prec=1e-2,
        theta_max_degrees=15,
        pade_order=pade_order,
        z_order=4,
        c_bounds=[c_min, c_max],
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
sspe_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range,
    comp_params=sspe_shifted_comp_params,
    c0=c0s
)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.88
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-70, 0).show()


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
sspe_etalon_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range,
    comp_params=sspe_shifted_etalon_comp_params,
    c0=c0s
)
sspe_etalon_field = sspe_etalon_propagator.calculate()
sspe_etalon_field.field *= 5.88



f, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 3.2), constrained_layout=True)
extent = [sspe_vis.field.x_grid[0]*1e-3, sspe_vis.field.x_grid[-1]*1e-3, sspe_vis.field.z_grid[-1], sspe_vis.field.z_grid[0]]

norm = Normalize(-90, -5)
im = ax[0].imshow(20*np.log10(abs(sspe_vis.field.field.T)), extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[0].grid()
ax[0].set_title(" ")
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Depth (m)')
f.colorbar(im, ax=ax[0], shrink=0.6, location='bottom')

norm = Normalize(-50, -20)
im = ax[1].imshow(10*np.log10(0.4*abs(sspe_field.field.T[:-1:,:] - sspe_etalon_field.field.T[::, ::])), extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[1].grid()
ax[1].set_title(" ")
ax[1].set_xlabel('Range (km)')
ax[1].set_ylabel('Depth (m)')

f.colorbar(im, ax=ax[1], shrink=0.8, location='bottom')
#plt.show()
plt.savefig("ex3_munk.eps")

plt.figure(figsize=(2.5, 4.3))
z_grid = np.linspace(0, 5000, 5001)
plt.plot(munk_profile(z_grid[::-1]), z_grid[::-1])
plt.xlim([c_min, 1550])
plt.ylim([5000, 0])
plt.grid(True)
plt.xlabel("Sound speed (m/s)")
plt.ylabel("Depth (m)")
plt.tight_layout()
#plt.show()
plt.savefig("ex3_munk_profile.eps")