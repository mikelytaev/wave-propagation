from rwp.antennas import *
from rwp.environment import *
from propagators.ts import *
from rwp.field import *
import math as fm


class KnifeEdgeDiffractionCalculator:

    def __init__(self, src: Source, env: Troposphere, max_range_m, dx_m=1, max_propagation_angle=90, x_grid_m=None,
                 z_grid_m=None, p_grid_size=None):
        if not env.is_homogeneous():
            raise Exception("Tropospheric refraction not yet supported")
        if src.polarz.upper() == 'V' and len(env.knife_edges) > 0:
            raise Exception("Vertical polarization not yet supported")

        max_height_m = env.z_max
        max_p_k0 = fm.sin(max_propagation_angle / 180 * cm.pi)
        p_grid_size = p_grid_size or fm.ceil(max_range_m * 3.5 / src.wavelength * max_p_k0) # empirical
        self.src = src
        width = 4e-5
        eps_r = 1e7
        alpha = 1e-5
        bodies = []
        for ke in env.knife_edges:
            bodies += [Plate(x0_m=ke.range, z1_m=-ke.height, z2_m=ke.height, width_m=width, eps_r=eps_r)]

        params = ThinScatteringComputationalParams(max_p_k0=max_p_k0, p_grid_size=p_grid_size, dx_m=dx_m, x_min_m=0,
                                                   x_max_m=max_range_m, z_min_m=0, z_max_m=max_height_m, z_grid_size=1000,
                                                   quadrature_points=1, alpha=alpha, use_mean_value_theorem=False,
                                                   spectral_integration_method=SpectralIntegrationMethod.fcc, h_curve=fm.log(100) / max_height_m,
                                                   x_grid_m=x_grid_m, z_grid_m=z_grid_m)

        polarz_sign = -1 if self.src.polarz == 'H' else 1
        if isinstance(src, GaussAntenna):
            def fur_q_func(z_spectral_points):
                ww = cm.sqrt(2 * cm.log(2)) / (src.k0 * cm.sin(src.beam_width * cm.pi / 180 / 2))
                if self.src.polarz == 'H':
                    q = -4j * src.k0 * np.sin(z_spectral_points * src.height_m) * np.exp(
                        -(z_spectral_points * ww) ** 2 / 4) / cm.sqrt(2 * cm.pi)
                else:
                    q = -4 * src.k0 * np.cos(z_spectral_points * src.height_m) * np.exp(
                        -(z_spectral_points * ww) ** 2 / 4) / cm.sqrt(2 * cm.pi)
                return q
        else:
            def fur_q_func(z_spectral_points):
                return 1 / cm.sqrt(2*cm.pi) * (np.exp(-1j * src.height_m * z_spectral_points) + polarz_sign * np.exp(1j * src.height_m * z_spectral_points))
        self.ts = ThinScattering(wavelength=src.wavelength, fur_q_func=fur_q_func, bodies=bodies, params=params, save_debug=False)

    def calculate(self):
        f = self.ts.calculate()
        z_ind = self.ts.z_computational_grid >= 0
        field = Field(x_grid=self.ts.x_computational_grid, z_grid=self.ts.z_computational_grid[z_ind], freq_hz=300e6)
        field.field[:, :] = f[:, z_ind]
        return field
