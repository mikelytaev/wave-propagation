# Incidence of vertically polarized waves at the Brewster angle

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from propagators.wavenumber import *
from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 100
environment.ground_material = WetGround()
environment.knife_edges = [KnifeEdge(range=1000, height=50)]

freq_hz = 3000e6
b_angle = brewster_angle(1, environment.ground_material.complex_permittivity(freq_hz)).real

antenna = GaussAntenna(freq_hz=freq_hz, height=50, beam_width=5, eval_angle=90-b_angle, polarz='V')
max_range = 2000

pade_params = HelmholtzPropagatorComputationalParams(two_way=True,
                                                     exp_pade_order=(7, 8),
                                                     max_propagation_angle=abs(antenna.beam_width) + abs(antenna.eval_angle) + 5,
                                                     z_order=5,
                                                     dx_wl=2,
                                                     dz_wl=0.2,
                                                     inv_z_transform_tau=10**(3 / (max_range / antenna.wavelength / 2))
                                                     )
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=pade_params)
pade_field = pade_task.calculate()

petool_task = PETOOLPropagationTask(antenna=antenna, env=environment, two_way=True, max_range_m=max_range, dx_wl=50, n_dx_out=1, dz_wl=0.2, n_dz_out=5)
petool_field = petool_task.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC', x_mult=1)

plt = pade_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.show()

petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1)
plt = petool_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt = petool_vis.plot_hor(50, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()