# Падение волн вертикальной поляризации под углом Брюстера

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
#from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 80
environment.ground_material = CustomMaterial(eps=3, sigma=0)

b_angle = brewster_angle(1, environment.ground_material.complex_permittivity(3e9))

antenna = GaussAntenna(freq_hz=3000e6, height=50, beam_width=0.3, eval_angle=90-b_angle, polarz='V')
h1 = antenna.height_m
h2 = 0
a = abs((h1 - h2) / cm.tan(antenna.eval_angle * cm.pi / 180))
max_range = 2 * a + 20 + 100
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, two_way=False, max_range_m=max_range, pade_order=(7, 8))
pade_field = pade_task.calculate()

computed_refl_coef = abs(pade_field.value(2 * a, h1)) / abs(pade_field.value(0, h1))
real_refl_coef = reflection_coef(1, environment.ground_material.complex_permittivity(antenna.freq_hz), b_angle, antenna.polarz)

print('reflection coef real: ' + str((real_refl_coef)))
print('reflection coef comp: ' + str(computed_refl_coef))

# petool_task = PETOOLPropagationTask(src=antenna, env=environment, two_way=False, max_range_m=max_range, dx_wl=4,
#                                     n_dx_out=10, dz_wl=0.25, n_dz_out=4)
# petool_field = petool_task.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade + NLBC', x_mult=1)
# petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
# plt = petool_vis.plot_hor(30, pade_vis)
# plt.xlabel('Расстояние, км')
# plt.ylabel('10log|u| (дБ)')
# plt.tight_layout()
# plt.show()
# #plt.savefig("ex1_pade_vs_petool_h30m.eps")
#
# plt = petool_vis.plot2d(min=-120, max=0)
# plt.xlabel('Расстояние, км')
# plt.ylabel('Высота, м')
# plt.tight_layout()
# plt.show()
#plt.savefig("ex1_petool.eps")

plt = pade_vis.plot2d(min=-30, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.grid(True)
plt.show()
#plt.savefig("ex1_pade.eps")

plt = pade_vis.plot_hor(50)
plt.xlabel('Расстояние, км')
plt.ylabel('10log|u| (дБ)')
plt.tight_layout()
plt.grid(True)
plt.show()
# #plt.savefig("ex1_pade_vs_petool_h30m.eps")