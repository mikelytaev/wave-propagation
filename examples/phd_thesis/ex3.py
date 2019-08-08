# Приподнятый волновод и неоднородность ландшафта SSPade vs SSF

from rwp.sspade import *
from rwp.vis import *
from rwp.crank_nicolson import *

from propagators.sspade import TerrainMethod

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
h = 110
w = 10000
x1 = 30000
environment.terrain = Terrain(lambda x: h/2*(1 + fm.sin(fm.pi * (x - x1) / (2*w))) if -w <= (x-x1) <= 3*w else 0)
environment.ground_material = VeryDryGround()
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)

antenna = GaussAntenna(freq_hz=3000e6, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 100e3
comp_params = HelmholtzPropagatorComputationalParams(terrain_method=TerrainMethod.pass_through)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=comp_params)
pade_field = pade_task.calculate()

claerbout_task = CrankNicolsonPropagationTask(src=antenna, env=environment, type='claerbout', max_range_m=max_range,
                                              dx_wl=25, n_dx_out=16, dz_wl=1, n_dz_out=1)
claerbout_field = claerbout_task.calculate()

pade_vis = FieldVisualiser(pade_field, env=environment, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade-[7/8] + NLBC', x_mult=1E-3)
claerbout_vis = FieldVisualiser(claerbout_field, env=environment, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                                label='Claerbout approx.', x_mult=1E-3)

plt = claerbout_vis.plot2d(min=-70, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("ex3_claerbout.eps")

plt = pade_vis.plot2d(min=-70, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("ex3_pade.eps")

plt = pade_vis.plot_hor_over_terrain(30, claerbout_vis)
plt.xlabel('Расстояние, км')
plt.ylabel('10log|u| (дБ)')
plt.tight_layout()
plt.grid(True)
plt.show()
plt.savefig("ex3_pade_vs_claerbout_h30m.eps")

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot([a / 6371000 * 1E6 for a in pade_field.z_grid], pade_field.z_grid)
ax1.legend()
ax1.set_xlabel('M-ед.')
ax1.set_ylabel('Высота, м')

ax2.plot([profile1d(a) for a in pade_field.z_grid], pade_field.z_grid)
ax2.legend()
ax2.set_xlabel('M-ед.')
ax2.set_ylabel('Высота, м')
f.tight_layout()
ax1.grid()
ax2.grid()
f.show()