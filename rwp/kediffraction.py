from rwp.antennas import *
from rwp.environment import *
from propagators.ts import *
from rwp.field import *
import math as fm


class KnifeEdgeDiffractionCalculator:

    def __init__(self, src: Source, env: Troposphere, max_range_m, max_height_m):
        if not env.is_homogeneous():
            raise Exception("Tropospheric refraction not yet supported")
        if src.polarz.upper() == 'V':
            raise Exception("Vertical polarization not yet supported")

        self.src = src
        width = 3e-5
        eps_r = 1e7
        alpha = 1e-5
        bodies = []
        for ke in env.knife_edges:
            bodies += [Plate(x0_m=ke.range, z1_m=-ke.height, z2_m=ke.height, width_m=width, eps_r=eps_r)]

        params = ThinScatteringComputationalParams(max_p_k0=0.5, p_grid_size=1000, x_grid_size=1000, x_min_m=0,
                                                   x_max_m=max_range_m, z_min_m=-max_height_m, z_max_m=max_height_m,
                                                   quadrature_points=1, alpha=alpha)

        self.ts = ThinScattering(src=self.src, bodies=bodies, params=params)

    def calculate(self):
        f = self.ts.calculate()
        z_ind = self.ts.z_computational_grid >= 0
        field = Field(x_grid=self.ts.x_computational_grid, z_grid=self.ts.z_computational_grid[z_ind], freq_hz=300e6)
        field.field[:, :] = f[:, z_ind]
        return field
