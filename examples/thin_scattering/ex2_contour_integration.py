from propagators.ts import *
from rwp.field import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
comp_params = ThinScatteringComputationalParams(max_p_k0=1.001,
                                                p_grid_size=1000,
                                                quadrature_points=10,
                                                alpha=1e-5,
                                                spectral_integration_method=SpectralIntegrationMethod.contour,
                                                h_curve=0.1,
                                                x_grid_m=np.linspace(-5, 500, 500),
                                                #z_grid_m=np.linspace(-10, 10, 500),
                                                z_grid_size=600,
                                                z_min_m=-50,
                                                z_max_m=50)
bodies = []
#bodies += [Ellipse(x0=40, z0=0, a=0.5, b=4, eps_r=5)]
#bodies += [Plate(x0_m=-2.5, z1_m=-2.5, z2_m=2.5, width_m=0.5, eps_r=50)]
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