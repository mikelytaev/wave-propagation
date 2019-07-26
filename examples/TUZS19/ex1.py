from rwp.sspade import *
from rwp.vis import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 200

h = 20
w = 1000
x1 = 3000

env.terrain = Terrain(lambda x: h/2*(1 + fm.sin(fm.pi * (x - x1) / (2*w))) if -w <= (x-x1) <= 3*w else 0)
max_range = 10000

ant_mm = GaussAntenna(freq_hz=30000e6, height=10, beam_width=10, eval_angle=0, polarz='H')
ant_uhf = GaussAntenna(freq_hz=300e6, height=10, beam_width=10, eval_angle=0, polarz='H')

comp_params = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8))
pade_task_mm = TroposphericRadioWaveSSPadePropagator(antenna=ant_mm, env=env, max_range_m=max_range, comp_params=comp_params)
pade_field_mm = pade_task_mm.calculate()

pade_task_uhf = TroposphericRadioWaveSSPadePropagator(antenna=ant_uhf, env=env, max_range_m=max_range)
pade_field_uhf = pade_task_uhf.calculate()

pade_vis_mm = FieldVisualiser(pade_field_mm.path_loss(gamma=0.1022), env=env, label='30 ГГц', x_mult=1e-3)
pade_vis_uhf = FieldVisualiser(pade_field_uhf.path_loss(gamma=0), env=env, label='300 МГц', x_mult=1e-3)


plt = pade_vis_mm.plot2d(min=50, max=225)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

plt = pade_vis_uhf.plot2d(min=50, max=225)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

pade_vis_mm_rain = FieldVisualiser(pade_field_mm.path_loss(gamma=0.1022 + 2.4531), label='30 ГГЦ, 12 мм/ч', x_mult=1e-3)
pade_vis_uhf_rain = FieldVisualiser(pade_field_uhf.path_loss(gamma=0), env=env, label='300 МГц, 12 мм/ч ', x_mult=1e-3)

plt = pade_vis_mm.plot_hor_over_terrain(2, pade_vis_uhf, pade_vis_mm_rain, pade_vis_uhf_rain)
plt.xlabel('Расстояние, км')
plt.ylabel('Потери, дБ')
plt.tight_layout()
plt.grid(True)
plt.show()