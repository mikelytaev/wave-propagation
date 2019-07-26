from rwp.sspade import *
from rwp.vis import *
from rwp.petool import PETOOLPropagationTask
from rwp.crank_nicolson import *
from propagators.sspade import TerrainMethod

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.ground_material = VeryDryGround()
env.z_max = 300

h = 110
w = 10000
x1 = 30000

env.terrain = Terrain(lambda x: h/2*(1 + fm.sin(fm.pi * (x - x1) / (2*w))) if -w <= (x-x1) <= 3*w else 0)
ant = GaussAntenna(freq_hz=3000e6, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 100000

computational_params = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8), z_order=2, dx_wl=400, dz_wl=1,
                                                              terrain_method=TerrainMethod.pass_through,
                                                              storage=PickleStorage())
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range, comp_params=computational_params)
pade_field = pade_task.calculate()

petool_task = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=max_range, dx_wl=400, n_dx_out=1, dz_wl=3)
petool_field = petool_task.calculate()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade-[7/8] + NLBC', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)

plt = pade_vis.plot_hor_over_terrain(30, petool_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = petool_vis.plot2d(min=-70, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis.plot2d(min=-70, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
