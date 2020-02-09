from propagators.ts import *
from rwp.field import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
comp_params = ThinScatteringComputationalParams(max_p_k0=1.5,
                                                p_grid_size=1000,
                                                quadrature_points=10,
                                                alpha=1e-3,
                                                spectral_integration_method=SpectralIntegrationMethod.fcc,
                                                x_grid_m=np.linspace(-5, 200, 500),
                                                z_grid_m=np.linspace(-10, 10, 500))
bodies = []
bodies += [Ellipse(x0=40, z0=0, a=0.5, b=4, eps_r=5)]
ts = ThinScattering(wavelength=1, bodies=bodies, params=comp_params, save_debug=False)
p_grid_r = np.linspace(-ts.k0*ts.params.max_p_k0, ts.k0*ts.params.max_p_k0, 1000)
p_grid_c = np.linspace(-ts.k0*ts.params.max_p_k0, ts.k0*ts.params.max_p_k0, 1000)
p_grid_r_m, p_grid_c_m = np.meshgrid(p_grid_r, p_grid_c, indexing='ij')
p_grid_m = p_grid_r_m + 1j * p_grid_c_m
# gf = ts.green_function(0, 100, p_grid_m)
# plt.imshow(gf.T.real, norm=Normalize(-0.4, 0.4), extent=[p_grid_r[0] / ts.k0, p_grid_r[-1] / ts.k0, p_grid_c[0] / ts.k0, p_grid_c[-1] / ts.k0])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.show()

h = 0.1
p_grid_h_1 = np.linspace(-ts.k0*ts.params.max_p_k0, -h, 500) + 1j*h
p_grid_h_2 = np.linspace(-h, h, 500) + 1j * np.linspace(h, -h, 500)
p_grid_h_3 = np.linspace(h, ts.k0*ts.params.max_p_k0, 500) - 1j*h
p_grid_h = np.concatenate((p_grid_h_1, p_grid_h_2[1::], p_grid_h_3[1::]))
gf = ts.green_function(0, 100, p_grid_h)
plt.plot(p_grid_h.real / ts.k0, abs(gf))
plt.show()