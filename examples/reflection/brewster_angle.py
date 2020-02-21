# Incidence of vertically polarized waves at the Brewster angle

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 100
environment.ground_material = CustomMaterial(eps=7, sigma=0)

freq_hz = 3000e6
b_angle = brewster_angle(1, environment.ground_material.complex_permittivity(freq_hz))

antenna = GaussAntenna(freq_hz=freq_hz, height=50, beam_width=0.3, eval_angle=(90-b_angle), polarz='V')
h1 = antenna.height_m
h2 = 0
a = abs((h1 - h2) / cm.tan(abs(antenna.eval_angle) * cm.pi / 180))
max_range = 2 * a + 200
params = HelmholtzPropagatorComputationalParams(two_way=False,
                                                exp_pade_order=(7, 8),
                                                max_propagation_angle=abs(antenna.eval_angle)+5,
                                                z_order=4)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=params)
pade_field = pade_task.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC', x_mult=1)

plt = pade_vis.plot2d(min=-50, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.show()

plt = pade_vis.plot_hor(50)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()
