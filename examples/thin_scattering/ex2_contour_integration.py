from propagators.ts import *
from rwp.field import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
z_max = 150
comp_params = ThinScatteringComputationalParams(max_p_k0=0.90,
                                                p_grid_size=1000,
                                                quadrature_points=1,
                                                alpha=1e-5,
                                                spectral_integration_method=SpectralIntegrationMethod.contour,
                                                h_curve=fm.log(100) / z_max,
                                                x_grid_m=np.linspace(-5, 1000, 500),
                                                #z_grid_m=np.linspace(-10, 10, 500),
                                                z_grid_size=600,
                                                z_min_m=-z_max,
                                                z_max_m=z_max)
bodies = []
#bodies += [Ellipse(x0=200, z0=0, a=0.5, b=4, eps_r=5)]
bodies += [Plate(x0_m=200, z1_m=0, z2_m=50, width_m=4e-5, eps_r=1e7),
           Plate(x0_m=500, z1_m=0, z2_m=50, width_m=4e-5, eps_r=1e7)]
#bodies += [Plate(x0_m=5, z1_m=-7.5, z2_m=5, width_m=0.1, eps_r=50)]
ts = ThinScattering(wavelength=1, bodies=bodies, params=comp_params, save_debug=False)
f = ts.calculate()

field = Field(x_grid=ts.x_computational_grid, z_grid=ts.z_computational_grid, freq_hz=300e6)
field.field[:, :] = f
vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-40, max=0)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()