from rwp.antennas import *
from rwp.environment import *
from propagators.ts import *


class KnifeEdgeDiffractionCalculator:

    def __init__(self, src: Source, env: Troposphere, max_range_m):
        if not env.is_homogeneous():
            raise Exception("Tropospheric refraction not yet supported")
        if src.polarz.upper() == 'V':
            raise Exception("Vertical polarization not yet supported")

        width = 1e-5
        eps_r = 1e7
        alpha = 1e-5
        bodies = []
        for ke in env.knife_edges:
            bodies += Plate(x0_m=ke.range, z1_m=-ke.height, z2_m=ke.height, width_m=width, eps_r=eps_r)

        params = ThinScatteringComputationalParams(max_p_k0=1, p_grid_size=500, x_min_m=0, x_max_m=max_range_m,
                                                   z_min_m=-300, z_max_m=300, quadrature_points=1, alpha=alpha)

        self.ts = ThinScattering(bodies=bodies, params=params)