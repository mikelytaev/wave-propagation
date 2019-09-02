from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
import matplotlib


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.ground_material = PerfectlyElectricConducting()
elevated_duct = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
environment.M_profile = lambda x, z: elevated_duct(z)
environment.knife_edges = [KnifeEdge(range=70e3, height=150)]

antenna = GaussAntenna(freq_hz=3000e6, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 100000
pade_comp_params = HelmholtzPropagatorComputationalParams(two_way=True, max_propagation_angle=10, dx_wl=100)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                  comp_params=pade_comp_params)
pade_field = pade_task.calculate()

computational_params_cn = HelmholtzPropagatorComputationalParams(exp_pade_order=(1, 1), two_way=True, z_order=2,
                                                                 dx_wl=80, x_output_filter=5, dz_wl=1, z_output_filter=1,
                                                                 max_propagation_angle=10)
crank_nicolson_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                            comp_params=computational_params_cn)
crank_nicolson_field = crank_nicolson_task.calculate()

pade78_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Предложенный метод', x_mult=1E-3)
crank_nicolson_vis = FieldVisualiser(crank_nicolson_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                                     label='Метод Кранка-Николсон', x_mult=1E-3)
plt = crank_nicolson_vis.plot_hor(150, pade78_vis)
plt.xlabel('Расстояние, км')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()

plt = crank_nicolson_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade78_vis.plot2d(min=-70, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("ex5_pade.eps")
