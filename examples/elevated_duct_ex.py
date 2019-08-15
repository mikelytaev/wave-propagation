from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
import matplotlib


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.ground_material = VeryDryGround()
environment.z_max = 300
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 45], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)
antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 150000

params = HelmholtzPropagatorComputationalParams()
pade78_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                    comp_params=params)
pade78_field = pade78_task.calculate()

matplotlib.rcParams.update({'font.size': 10})

pade78_vis = FieldVisualiser(pade78_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[7/8] + NLBC', x_mult=1E-3)
plt = pade78_vis.plot_hor(30)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = pade78_vis.plot2d(min=-70, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()