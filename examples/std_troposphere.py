from rwp.SSPade import *
from rwp.WPVis import *
from rwp.environment import *


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.ground_material = WetGround()
environment.z_max = 300
antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 120000
pade12_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, two_way=False, max_range_m=max_range, pade_order=(7, 8),
                                                    dx_wl=400, n_dx_out=1, dz_wl=1, n_dz_out=1)
pade12_field = pade12_task.calculate()

pade12_vis = FieldVisualiser(pade12_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[7/8] + NLBC', x_mult=1E-3)

plt = pade12_vis.plot_hor(30)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = pade12_vis.plot2d(min=-120, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()