from rwp.kediffraction import *
from rwp.antennas import *
from rwp.environment import *
from rwp.WPVis import *
from rwp.SSPade import *

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.ground_material = PerfectlyElectricConducting()
environment.z_max = 300
environment.knife_edges = [KnifeEdge(range=200, height=50)]
max_range_m = 300
antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, eval_angle=0, polarz='H')
propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range_m,
                                                           comp_params=HelmholtzPropagatorComputationalParams(
                                                               exp_pade_order=(7, 8)))
sspade_field = propagator.calculate()

kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=environment, max_range_m=max_range_m,
                                             x_grid_m=sspade_field.x_grid, z_grid_m=sspade_field.z_grid, p_grid_size=2000)
ke_field = kdc.calculate()

sspe_vis = FieldVisualiser(sspade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='sspe')
plt = sspe_vis.plot2d(min=-40, max=0)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

ke_vis = FieldVisualiser(ke_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = ke_vis.plot2d(min=-40, max=0)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

ke_vis.plot_hor(70, sspe_vis)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

f1 = 10 * np.log10(1e-16 + np.abs(sspade_field.horizontal(50)))
f2 = 10 * np.log10(1e-16 + np.abs(ke_field.horizontal(50)))
np.linalg.norm(f1 - f2) / np.linalg.norm(f1)
