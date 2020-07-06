from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *
from rwp.petool import *

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.ground_material = VeryDryGround()
environment.z_max = 300
max_range = 1000
#environment.knife_edges = [KnifeEdge(range=800, height=125)]

#profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 45], fill_value="extrapolate")
profile1d = interp1d(x=[0, 5, 70, 100, 300], y=[0, 0, -30, 0, 0], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)
antenna = GaussAntenna(freq_hz=100e6, height=100, beam_width=15, eval_angle=30, polarz='V')

propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                   comp_params=HelmholtzPropagatorComputationalParams(
                                                       max_propagation_angle=50
                                                   ))
#environment.ground_material = PerfectlyElectricConducting()
propagator_local_bc = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                            comp_params=HelmholtzPropagatorComputationalParams(
                                                                max_propagation_angle=50,
                                                                terrain_method=TerrainMethod.staircase
                                                            ))
field = propagator.calculate()
field_local_bc = propagator_local_bc.calculate()

petool_task = PETOOLPropagationTask(antenna=antenna, env=environment, two_way=False, max_range_m=max_range,
                                    dx_wl=propagator.comp_params.dx_wl, n_dx_out=1, dz_wl=0.2, n_dz_out=5)
petool_field = petool_task.calculate()

vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + Transparent BC', x_mult=1E-3)
vis_local_bc = FieldVisualiser(field_local_bc, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + local bc', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)

plt = vis.plot_hor(125, vis_local_bc, petool_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = vis.plot2d(min=-40, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = vis_local_bc.plot2d(min=-40, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = petool_vis.plot2d(min=-40, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

# coefs = propagator.propagator.lower_bc.coefs
# abs_coefs = np.array([np.linalg.norm(a) for a in coefs])
# plt.plot(np.log10(abs_coefs))
# plt.show()