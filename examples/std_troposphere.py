from rwp.SSPade import *
from rwp.WPVis import *
from rwp.environment import *


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.ground_material = WetGround()
environment.z_max = 300
antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 150e3
propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range)
field = propagator.calculate()

vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade+NLBC', x_mult=1E-3)

plt = vis.plot_hor(30)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = vis.plot2d(min=-120, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
