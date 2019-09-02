# Стандартная атмосфера SSPade vs SSF

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from rwp.petool import PETOOLPropagationTask
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
environment = Troposphere()
environment.ground_material = SaltWater()

antenna = GaussAntenna(freq_hz=3000e6, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 150e3
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range)
pade_field = pade_task.calculate()

petool_task = PETOOLPropagationTask(antenna=antenna, env=environment, two_way=False, max_range_m=max_range, dx_wl=100,
                                    n_dx_out=1, dz_wl=1)
petool_field = petool_task.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Предложенный метод', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='Метод Фурье (PETOOL)', x_mult=1E-3)
plt = petool_vis.plot_hor(30, pade_vis)
plt.xlabel('Расстояние, км')
plt.ylabel('10log|u| (дБ)')
plt.tight_layout()
plt.grid(True)
plt.show()
plt.savefig("ex1_pade_vs_petool_h30m.eps")

plt = petool_vis.plot2d(min=-120, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("ex1_petool.eps")

plt = pade_vis.plot2d(min=-120, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("ex1_pade.eps")