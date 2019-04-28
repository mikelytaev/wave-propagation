from rwp.petool import *
from rwp.WPVis import *
import matplotlib.patches as patches

logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 100
#env.knife_edges = [KnifeEdge(range=150, height=30)]
env.terrain = InterpTerrain([0, 70, 70.0001, 80, 80.00001, 160, 160.000001, 170, 170.000001, 210], [0, 0, 15, 15, 0, 0, 50, 50, 0, 0])
max_range = 200

ant_mm = GaussAntenna(freq_hz=30000e6, height=10, beam_width=10, eval_angle=0, polarz='H')

pade_task_mm = PETOOLPropagationTask(antenna=ant_mm, env=env, two_way=False, max_range_m=max_range, dx_wl=30, dz_wl=1, n_dz_out=5, n_dx_out=1)
pade_field_mm = pade_task_mm.calculate()

pade_task_mm_tw = PETOOLPropagationTask(antenna=ant_mm, env=env, two_way=True, max_range_m=max_range, dx_wl=30, dz_wl=1, n_dz_out=5, n_dx_out=1)
pade_field_mm_tw = pade_task_mm_tw.calculate()

pade_vis_mm = FieldVisualiser(pade_field_mm, env=env, label='30 ГГц', x_mult=1)

plt = pade_vis_mm.plot2d(min=20, max=200)
plt.xlabel('Расстояние, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.gca().add_patch(patches.Rectangle((70, 0), 10, 15, linewidth=1, edgecolor='black', facecolor='gray'))
plt.gca().add_patch(patches.Rectangle((160, 0), 10, 50, linewidth=1, edgecolor='black', facecolor='gray'))
plt.show()

pade_vis_mm_tw = FieldVisualiser(pade_field_mm_tw, env=env, label='30 ГГц', x_mult=1)

plt = pade_vis_mm_tw.plot2d(min=20, max=200)
plt.xlabel('Расстояние, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.gca().add_patch(patches.Rectangle((70, 0), 10, 15, linewidth=1, edgecolor='black', facecolor='gray'))
plt.gca().add_patch(patches.Rectangle((160, 0), 10, 50, linewidth=1, edgecolor='black', facecolor='gray'))
plt.show()
