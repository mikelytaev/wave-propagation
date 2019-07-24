from rwp.antennas import *
from rwp.environment import *
from rwp.field import Field
from propagators.sspade import *


class TroposphericRadioWaveSSPadePropagator:

    def __init__(self, *, antenna: Source, env: Troposphere, max_range_m: float,
                 comp_params: HelmholtzPropagatorComputationalParams = None):
        self.src = antenna
        self.env = env
        if comp_params:
            self.comp_params = comp_params
        else:
            self.comp_params = HelmholtzPropagatorComputationalParams()
        if len(self.env.knife_edges) == 0:
            self.comp_params.two_way = False
        k0 = 2*cm.pi / self.src.wavelength

        logging.info("ground refractive index: " + str(self.env.ground_material.complex_permittivity(antenna.freq_hz)))
        if self.comp_params.terrain_method is None:
            if self.env.terrain.is_homogeneous:
                self.comp_params.terrain_method = TerrainMethod.no
            elif abs(self.env.ground_material.complex_permittivity(antenna.freq_hz)) < 100:
                self.comp_params.terrain_method = TerrainMethod.pass_through
            else:
                self.comp_params.terrain_method = TerrainMethod.staircase

        logging.info("Terrain method: " + self.comp_params.terrain_method.name)

        if self.comp_params.terrain_method == TerrainMethod.pass_through:
            lower_bc = TransparentBC(self.env.ground_material.complex_permittivity(self.src.freq_hz))
        else:
            if isinstance(self.env.ground_material, PerfectlyElectricConducting):
                if self.src.polarz.upper() == 'H':
                    q1, q2 = 1, 0
                else:
                    q1, q2 = 0, 1
            else:
                if self.src.polarz.upper() == 'H':
                    q1, q2 = 1j * k0 * self.env.ground_material.complex_permittivity(self.src.freq_hz) ** (1 / 2), 1
                else:
                    q1, q2 = 1j * k0 * self.env.ground_material.complex_permittivity(self.src.freq_hz) ** (-1 / 2), 1

            lower_bc = RobinBC(q1, q2, 0)

        if self.src.polarz.upper() == 'V':
            rho = lambda x, z: 1 / (self.env.n2m1_profile(x, z, self.src.freq_hz) + 1)
        else:
            rho = lambda x, z: z*0+1

        if self.env.is_flat:
            upper_bc = TransparentBC(self.env.n2m1_profile(0, self.env.z_max, self.src.freq_hz) + 1)
        else:
            gamma = self.env.n2m1_profile(0, self.env.z_max + 1, self.src.freq_hz) - self.env.n2m1_profile(0, self.env.z_max,
                                                                                               self.src.freq_hz)
            beta = self.env.n2m1_profile(0, self.env.z_max, self.src.freq_hz) + 1
            upper_bc = TransparentBC(beta, gamma)

        if self.comp_params.terrain_method == TerrainMethod.pass_through:
            def n2m1(x, z, freq_hz):
                if isinstance(z, float):
                    if z < self.env.terrain(x):
                        return self.env.ground_material.complex_permittivity(freq_hz) - 1
                    else:
                        return self.env.n2m1_profile(x, z, freq_hz)
                res = self.env.n2m1_profile(x, z, freq_hz)
                ind = z < self.env.terrain(x)
                res[ind] = self.env.ground_material.complex_permittivity(freq_hz) - 1
                return res
        else:
            def n2m1(x, z, freq_hz):
                return self.env.n2m1_profile(x, z, freq_hz)

        self.comp_params.max_abc_permittivity = abs(self.env.ground_material.complex_permittivity(self.src.freq_hz))

        self.helm_env = HelmholtzEnvironment(x_max_m=max_range_m,
                                             lower_bc=lower_bc,
                                             upper_bc=upper_bc,
                                             z_min=0,
                                             z_max=env.z_max,
                                             n2minus1=n2m1,
                                             use_n2minus1=self.env.is_homogeneous(),
                                             rho=rho,
                                             use_rho=self.env.is_homogeneous() or self.src.polarz.upper() == 'H',
                                             terrain=self.env.terrain)

        for kn in self.env.knife_edges:
            self.helm_env.knife_edges += [Edge(x=kn.range, z_min=0, z_max=kn.height)]

        if self.comp_params.two_way is None:
            self.comp_params.two_way = len(self.helm_env.knife_edges) > 0

        self.propagator = HelmholtzPadeSolver(env=self.helm_env, wavelength=self.src.wavelength, freq_hz=self.src.freq_hz, params=self.comp_params)

    def calculate(self):
        h_field = self.propagator.calculate(lambda z: self.src.aperture(z))
        res = Field(x_grid=h_field.x_grid_m, z_grid=h_field.z_grid_m, freq_hz=self.src.freq_hz)
        res.field = h_field.field
        return res


# def bessel_ratio(c, d, j, tol):
#     return lentz(lambda n: (-1)**(n+1) * 2.0 * (j + (2.0 + d) / c + n - 1) * (c / 2.0))


