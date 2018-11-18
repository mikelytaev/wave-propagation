from rwp.SSPade import *
from rwp.WPVis import *
from rwp.environment import *
import matplotlib


logging.basicConfig(level=logging.DEBUG)
elevated_duct = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
environment = EarthAtmosphereEnvironment(boundary_condition=PECSurfaceBC(), height=300, M_profile=lambda x, z: elevated_duct(z))
environment.terrain = KnifeEdges([70000], [150])
antenna = GaussSource(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 100000
pade78_task = SSPadePropagationTask(src=antenna, env=environment, two_way=True, max_range_m=max_range, pade_order=(7, 8),
                                    dx_wl=400, n_dx_out=1, dz_wl=1, n_dz_out=1)
pade78_field = pade78_task.calculate()

crank_nicolson_task = SSPadePropagationTask(src=antenna, env=environment, two_way=True, max_range_m=max_range, pade_order=(1, 1),
                                            dx_wl=80, n_dx_out=5, dz_wl=1, n_dz_out=1)
crank_nicolson_field = crank_nicolson_task.calculate()

matplotlib.rcParams.update({'font.size': 10})

pade78_vis = FieldVisualiser(pade78_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[7/8] + NLBC', x_mult=1E-3)
crank_nicolson_vis = FieldVisualiser(crank_nicolson_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                                     label='Claerbout-CN', x_mult=1E-3)
plt = crank_nicolson_vis.plot_hor(150, pade78_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = crank_nicolson_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade78_vis.plot2d(min=-70, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()