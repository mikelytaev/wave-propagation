from rwp.antennas import *
from rwp.environment import *
from propagators.ts import *
from rwp.field import *
import math as fm


class KnifeEdgeDiffractionCalculator:

    def __init__(self, src: Source, env: Troposphere, max_range_m, dx_m, max_height_m):
        if not env.is_homogeneous():
            raise Exception("Tropospheric refraction not yet supported")
        if src.polarz.upper() == 'V':
            raise Exception("Vertical polarization not yet supported")

        self.src = src
        width = 4e-5
        eps_r = 1e7
        alpha = 1e-5
        x_grid_size = max_range_m
        bodies = []
        for ke in env.knife_edges:
            bodies += [Plate(x0_m=ke.range, z1_m=-ke.height, z2_m=ke.height, width_m=width, eps_r=eps_r)]

        params = ThinScatteringComputationalParams(max_p_k0=1.001, p_grid_size=4000, dx_m=dx_m, x_min_m=0,
                                                   x_max_m=max_range_m, z_min_m=-max_height_m, z_max_m=max_height_m,
                                                   quadrature_points=1, alpha=alpha, use_mean_value_theorem=False,
                                                   spectral_integration_method=SpectralIntegrationMethod.fractional_ft)

        if isinstance(src, GaussAntenna):
            def fur_q_func(z_spectral_points):
                ww = cm.sqrt(2 * cm.log(2)) / (src.k0 * cm.sin(src.beam_width * cm.pi / 180 / 2))
                q = -4j * src.k0 * np.sin(z_spectral_points * src.height_m) * np.exp(-(z_spectral_points * ww)**2 / 4) / cm.sqrt(2*cm.pi)
                return q
        else:
            def fur_q_func(z_spectral_points):
                return 1 / cm.sqrt(2*cm.pi) * (np.exp(-1j * src.height_m * z_spectral_points) - np.exp(1j * src.height_m * z_spectral_points))
        self.ts = ThinScattering(wavelength=src.wavelength, fur_q_func=fur_q_func, bodies=bodies, params=params, save_debug=True)

    def calculate(self):
        f = self.ts.calculate()
        z_ind = self.ts.z_computational_grid >= 0
        field = Field(x_grid=self.ts.x_computational_grid, z_grid=self.ts.z_computational_grid[z_ind], freq_hz=300e6)
        field.field[:, :] = f[:, z_ind]
        return field
