# Стандартная атмосфера и неоднородность ландшафта SSPade vs SSF

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from rwp.petool import PETOOLPropagationTask

from propagators.sspade import TerrainMethod

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
h = 110
w = 10000
x1 = 30000
environment.terrain = Terrain(lambda x: h/2*(1 + fm.sin(fm.pi * (x - x1) / (2*w))) if -w <= (x-x1) <= 3*w else 0)
environment.ground_material = VeryDryGround()

antenna = GaussAntenna(freq_hz=3000e6, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 100e3
comp_params = HelmholtzPropagatorComputationalParams(terrain_method=TerrainMethod.pass_through)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=comp_params)
pade_field = pade_task.calculate()

petool_task = PETOOLPropagationTask(antenna=antenna, env=environment, two_way=False, max_range_m=max_range, dx_wl=100,
                                    n_dx_out=1, dz_wl=1)
petool_field = petool_task.calculate()

pade_vis = FieldVisualiser(pade_field, env=environment, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade-[7/8] + NLBC', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, env=environment, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)

plt = petool_vis.plot2d(min=-70, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("ex2_petool.eps")

plt = pade_vis.plot2d(min=-70, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("ex2_pade.eps")

plt = pade_vis.plot_hor_over_terrain(30, petool_vis)
plt.xlabel('Расстояние, км')
plt.ylabel('10log|u| (дБ)')
plt.tight_layout()
plt.show()
plt.savefig("ex2_pade_vs_petool_h30m.eps")

# lower, upper = pade_task._prepare_bc()
# abs_lower = [10*fm.log10(np.linalg.norm(a)) for a in lower.coefs]
# plt.plot(abs_lower)
# plt.show()
#
# abs_upper = [10*fm.log10(np.linalg.norm(a)) for a in upper.coefs]
# plt.plot(abs_upper)
# plt.show()