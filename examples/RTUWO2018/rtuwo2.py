from rwp.SSPade import *
from rwp.WPVis import *
from rwp.crank_nicolson import *

logging.basicConfig(level=logging.DEBUG)

profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
env = EarthAtmosphereEnvironment(boundary_condition=VeryDryGroundBC(), height=2000, M_profile=lambda x, z: profile1d(z))

h = 110
w = 10000
x1 = 30000

env.terrain = Terrain(lambda x: h/2*(1 + fm.sin(fm.pi * (x - x1) / (2*w))) if -w <= (x-x1) <= 3*w else 0)

ant = GaussSource(freq_hz=3000e6, height=30, beam_width=5, eval_angle=0, polarz='H')
max_range = 100000
pade12_task = SSPadePropagationTask(src=ant, env=env, two_way=False, max_range_m=max_range, pade_order=(7, 8),
                                    dx_wl=100, n_dx_out=4, dz_wl=1, n_dz_out=1)
pade12_field = pade12_task.calculate()

claerbout_task = CrankNicolsonPropagationTask(src=ant, env=env, type='claerbout', max_range_m=max_range,
                                              dx_wl=25, n_dx_out=16, dz_wl=1, n_dz_out=1)
claerbout_field = claerbout_task.calculate()

greene_task = CrankNicolsonPropagationTask(src=ant, env=env, type='greene', max_range_m=max_range,
                                           dx_wl=25, n_dx_out=16, dz_wl=1, n_dz_out=1)
greene_field = greene_task.calculate()

matplotlib.rcParams.update({'font.size': 10})

pade12_vis = FieldVisualiser(pade12_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[7/8] + NLBC', x_mult=1E-3)
claerbout_vis = FieldVisualiser(claerbout_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                                label='Claerbout approx.', x_mult=1E-3)
greene_vis = FieldVisualiser(greene_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                                label='Greene approx.', x_mult=1E-3)

plt = pade12_vis.plot_hor(150, claerbout_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()
#plt.savefig("elevated_hor.eps")

plt = pade12_vis.plot2d(min=-70, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = claerbout_vis.plot2d(min=-70, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
pade12_vis.plot_ver(40 * 1E3, ax1, claerbout_vis, greene_vis)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')

pade12_vis.plot_ver(80 * 1E3, ax2, claerbout_vis, greene_vis)
ax2.set_ylabel('Height (m)')
ax2.set_xlabel('10log|u| (dB)')
f.tight_layout()
f.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot([a / 6371000 * 1E6 for a in pade12_field.z_grid], pade12_field.z_grid)
ax1.legend()
ax1.set_xlabel('M-units')
ax1.set_ylabel('Height (m)')

ax2.plot([profile1d(a) for a in pade12_field.z_grid], pade12_field.z_grid)
ax2.legend()
ax2.set_xlabel('M-units')
ax2.set_ylabel('Height (m)')
f.tight_layout()
f.show()