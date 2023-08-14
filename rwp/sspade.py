from rwp.antennas import *
from rwp.environment import *
from rwp.field import Field
from propagators.sspade import *
from copy import deepcopy
import logging


@dataclass
class RWPSSpadeComputationalParams:
    max_range_m: float = None
    max_height_m: float = None
    k0: float = None
    dx_m: float = None
    dz_m: float = None
    rational_approx_order = (7, 8)
    precision: float = 0.01


def rwp_ss_pade(antenna: Source, env: Troposphere, params: RWPSSpadeComputationalParams) -> Field:
    propagator = TroposphericRadioWaveSSPadePropagator(
        antenna=antenna,
        env=env,
        max_range_m=params.max_range_m,
        comp_params=HelmholtzPropagatorComputationalParams(
            exp_pade_order=params.rational_approx_order,
            max_height_m=params.max_height_m
        )
    )
    return propagator.calculate()


class TroposphericRadioWaveSSPadePropagator:

    def __init__(self, *, antenna: Source, env: Troposphere, max_range_m: float,
                 comp_params: HelmholtzPropagatorComputationalParams = None):
        self.src = deepcopy(antenna)
        self.env = deepcopy(env)
        if comp_params:
            self.comp_params = deepcopy(comp_params)
        else:
            self.comp_params = HelmholtzPropagatorComputationalParams()
        if len(self.env.knife_edges) == 0:
            self.comp_params.two_way = False
        k0 = 2*cm.pi / self.src.wavelength

        ground_material = self.env.terrain.ground_material(0)
        ground_eps_r = ground_material.complex_permittivity(self.src.freq_hz)
        logging.info("ground refractive index: " + str(ground_eps_r))
        if self.comp_params.terrain_method is None:
            if self.env.terrain.is_homogeneous:
                self.comp_params.terrain_method = TerrainMethod.no
            elif abs(ground_eps_r) < 100:
                self.comp_params.terrain_method = TerrainMethod.pass_through
            else:
                self.comp_params.terrain_method = TerrainMethod.staircase

        logging.info("Terrain method: " + self.comp_params.terrain_method.name)

        if self.comp_params.terrain_method == TerrainMethod.pass_through:
            lower_bc = TransparentBC(ground_eps_r)
        elif isinstance(ground_material, PerfectlyElectricConducting):
            if self.src.polarz.upper() == 'H':
                q1, q2 = 1, 0
            else:
                q1, q2 = 0, 1
            lower_bc = RobinBC(q1, q2, 0)
        elif self.comp_params.terrain_method == TerrainMethod.staircase:
            if self.src.polarz.upper() == 'H':
                q1, q2 = 1j * k0 * (ground_eps_r - 1) ** (1 / 2), 1
            else:
                q1, q2 = 1j * k0 * (ground_eps_r - 1) ** (1 / 2) / ground_eps_r, 1
            lower_bc = RobinBC(q1, q2, 0)
        elif self.comp_params.terrain_method == TerrainMethod.no:
            if self.env.rms_m:
                logging.error("rms not yet supported")
                mbf = MillerBrownFactor(8)
                reflection_coefficient = lambda theta, k_z=0: mbf.factor(theta, k0, self.env.rms_m) * reflection_coef(1, ground_eps_r, 90 - theta,
                                                                              self.src.polarz)
            else:
                reflection_coefficient = lambda theta, k_z=0: reflection_coef(1, ground_eps_r, 90 - theta, self.src.polarz)
            lower_bc = AngleDependentBC(reflection_coefficient)

        if self.src.polarz.upper() == 'V':
            rho = lambda x, z: 1 / (self.env.n2m1_profile(x, z, self.src.freq_hz) + 1)
        else:
            rho = lambda x, z: z*0+1

        if self.env.is_flat:
            upper_bc = TransparentBC(self.env.n2m1_profile(0, self.comp_params.max_height_m, self.src.freq_hz) + 1)
        else:
            gamma = self.env.n2m1_profile(0, self.comp_params.max_height_m + 1, self.src.freq_hz) - self.env.n2m1_profile(0, self.comp_params.max_height_m,
                                                                                               self.src.freq_hz)
            beta = self.env.n2m1_profile(0, self.comp_params.max_height_m, self.src.freq_hz) + 1
            upper_bc = TransparentBC(beta, gamma)

        if self.comp_params.terrain_method == TerrainMethod.pass_through:
            def n2m1(x, z, freq_hz):
                if isinstance(z, float):
                    if z < self.env.terrain.elevation(x):
                        return ground_material.complex_permittivity(freq_hz) - 1
                    else:
                        return self.env.n2m1_profile(x, z, freq_hz)
                res = self.env.n2m1_profile(x, z, freq_hz)
                ind = z < self.env.terrain.elevation(x)
                res[ind] = ground_material.complex_permittivity(freq_hz) - 1
                return res
        else:
            def n2m1(x, z, freq_hz):
                return self.env.n2m1_profile(x, z, freq_hz)

        self.comp_params.max_abc_permittivity = abs(ground_eps_r)

        if self.comp_params.max_propagation_angle is None:
            if len(self.env.knife_edges) > 0:
                max_angle = SSPE_MAX_ANGLE
            else:
                max_angle = max(antenna.max_angle(),
                                terrain_max_propagation_angle(terrain=self.env.terrain, distance_m=max_range_m))
            self.comp_params.max_propagation_angle = max_angle

            if self.comp_params.exp_pade_order is None:
                self.comp_params.exp_pade_order = (7, 8)

        self.helm_env = HelmholtzEnvironment(x_max_m=max_range_m,
                                             lower_bc=lower_bc,
                                             upper_bc=upper_bc,
                                             z_min=0,
                                             z_max=comp_params.max_height_m,
                                             n2minus1=n2m1,
                                             use_n2minus1=not self.env.is_homogeneous(),
                                             rho=rho,
                                             use_rho=False,#(not self.env.is_homogeneous()) and self.src.polarz.upper() == 'V',
                                             lower_z=lambda x: self.env.terrain.elevation(x))

        for kn in self.env.knife_edges:
            self.helm_env.knife_edges += [Edge(x=kn.range, z_min=0, z_max=kn.height)]

        if self.comp_params.two_way is None:
            self.comp_params.two_way = len(self.helm_env.knife_edges) > 0

        if self.comp_params.two_way and len(self.helm_env.knife_edges) == 1:
            self.comp_params.two_way_iter_num = 1

        self.propagator = HelmholtzPadeSolver(env=self.helm_env, wavelength=self.src.wavelength, freq_hz=self.src.freq_hz, params=self.comp_params)

    def calculate(self):
        h_field = self.propagator.calculate(lambda z: self.src.aperture(z))
        res = Field(x_grid=h_field.x_grid_m, z_grid=h_field.z_grid_m, freq_hz=self.src.freq_hz)
        res.field = h_field.field
        return res


def terrain_max_propagation_angle(terrain: Terrain, distance_m: float, step_m=10):
    if terrain.is_homogeneous:
        return 0
    res = 0
    step = 10
    for x in np.arange(step, distance_m, step):
        angle = cm.atan((terrain.elevation(x) - terrain.elevation(x - step)) / step) * 180 / cm.pi
        res = max(res, abs(angle))
    return res


# def bessel_ratio(c, d, j, tol):
#     return lentz(lambda n: (-1)**(n+1) * 2.0 * (j + (2.0 + d) / c + n - 1) * (c / 2.0))


