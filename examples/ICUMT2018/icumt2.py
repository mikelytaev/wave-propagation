from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from rwp.crank_nicolson import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 100
env.ground_material = PerfectlyElectricConducting()
env.knife_edges = [KnifeEdge(100, 10), KnifeEdge(175, 50), KnifeEdge(250, 20)]
antenna = GaussAntenna(freq_hz=30000e6, height=10, beam_width=20, eval_angle=0, polarz='H')
max_range = 300
comp_params_ow = HelmholtzPropagatorComputationalParams(two_way=False, exp_pade_order=(7, 8), dx_wl=1, x_output_filter=10,
                                                        dz_wl=0.25, z_output_filter=80)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=env, max_range_m=max_range, comp_params=comp_params_ow)
pade_field = pade_task.calculate()

pade_vis = FieldVisualiser(pade_field.path_loss(gamma=0), label='30 GHz')

plt = pade_vis.plot2d(min=30, max=200)
plt.title('Path loss (dB)')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

comp_params_tw = HelmholtzPropagatorComputationalParams(two_way=True, two_way_iter_num=2, exp_pade_order=(7, 8),
                                                     dx_wl=1, x_output_filter=10, dz_wl=0.25, z_output_filter=80)
pade_task_tw = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=env, max_range_m=max_range, comp_params=comp_params_tw)
pade_field_tw = pade_task_tw.calculate()

pade_vis_tw = FieldVisualiser(pade_field_tw.path_loss(gamma=0), label='30 GHz')

plt = pade_vis_tw.plot2d(min=30, max=200)
plt.title('Path loss (dB)')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()