from rwp.antennas import *
from rwp.environment import *
from rwp.field import Field
from propagators.sspade import *


class TroposphericRadioWaveSSPadePropagator:

    def __init__(self, *, antenna: Source, env: Troposphere, max_range_m, comp_params=HelmholtzPropagatorComputationalParams()):
        self.src = antenna
        self.env = env
        self.comp_params = comp_params
        if len(self.env.knife_edges) == 0:
            self.comp_params.two_way = False
        k0 = 2*cm.pi / self.src.wavelength

        logging.info("ground refractive index: " + str(self.env.ground_material.complex_permittivity(antenna.freq_hz)))
        if self.comp_params.terrain_method == TerrainMethod.no:
            if abs(self.env.ground_material.complex_permittivity(antenna.freq_hz)) < 100:
                self.comp_params.terrain_method = TerrainMethod.pass_through
            else:
                self.comp_params.terrain_method = TerrainMethod.staircase

        logging.info("Terrain method: " + self.comp_params.terrain_method.name)

        if self.comp_params.terrain_method == TerrainMethod.pass_through:
            lower_bc = TransparentConstBC()
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

        self.helm_env = HelmholtzEnvironment(x_max_m=max_range_m,
                                             lower_bc=lower_bc,
                                             upper_bc=TransparentLinearBC(),
                                             z_min=0,
                                             z_max=env.z_max,
                                             n2minus1=self.env.n2m1_profile(),
                                             use_n2minus1=self.env.is_homogeneous(),
                                             rho=rho,
                                             use_rho=self.env.is_homogeneous() or self.src.polarz.upper() == 'H',
                                             terrain=self.env.terrain)

        for kn in self.env.knife_edges:
            self.helm_env.knife_edges += Edge(x=kn.range, z_min=0, z_max=kn.height)

        self.propagator = HelmholtzPadeSolver(env=self.env, wavelength=self.src.wavelength, freq_hz=self.src.freq_hz, params=self.comp_params)


# def bessel_ratio(c, d, j, tol):
#     return lentz(lambda n: (-1)**(n+1) * 2.0 * (j + (2.0 + d) / c + n - 1) * (c / 2.0))


