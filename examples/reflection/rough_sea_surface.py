from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.ground_material = SaltWater()
environment.rms_m = 1
environment.z_max = 300
max_range = 150e3

profile1d = interp1d(x=[0, 50, 1050], y=[330-330, 320-330, 438-330], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)

antenna = GaussAntenna(wavelength=0.03, height=25, beam_width=2, eval_angle=0, polarz='H')

propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range)
field = propagator.calculate()

vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + Transparent BC', x_mult=1E-3)

plt = vis.plot_hor(25)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()


plt = vis.plot2d(min=-70, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
