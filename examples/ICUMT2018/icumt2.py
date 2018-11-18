from rwp.SSPade import *
from rwp.WPVis import *
from rwp.environment import *
from rwp.petool import PETOOLPropagationTask
from rwp.crank_nicolson import *

logging.basicConfig(level=logging.DEBUG)
env = EMEnvironment()
env.z_max = 100
env.lower_boundary = PECSurfaceBC()
#env = EarthAtmosphereEnvironment(boundary_condition=PECSurfaceBC(), height=100)
env.terrain = KnifeEdges([100, 175, 250], [10, 50, 20])
ant = GaussSource(freq_hz=30000e6, height=10, beam_width=20, eval_angle=0, polarz='H')
max_range = 300
pade12_task = SSPadePropagationTask(src=ant, env=env, two_way=False, max_range_m=max_range, pade_order=(7, 8),
                                    dx_wl=1, n_dx_out=10, dz_wl=0.25, n_dz_out=80)
pade12_field = pade12_task.calculate()

matplotlib.rcParams.update({'font.size': 10})

pade12_vis = FieldVisualiser(pade12_field.path_loss(gamma=0), label='30 GHz')

plt = pade12_vis.plot2d(min=30, max=200)
plt.title('Path loss (dB)')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

pade12_task_tw = SSPadePropagationTask(src=ant, env=env, two_way=True, max_range_m=max_range, pade_order=(7, 8),
                                    dx_wl=1, n_dx_out=10, dz_wl=0.25, n_dz_out=80)
pade12_field_tw = pade12_task_tw.calculate()

matplotlib.rcParams.update({'font.size': 10})

pade12_vis_tw = FieldVisualiser(pade12_field_tw.path_loss(gamma=0), label='30 GHz')

plt = pade12_vis_tw.plot2d(min=30, max=200)
plt.title('Path loss (dB)')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()