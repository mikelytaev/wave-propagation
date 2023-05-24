import unittest

from rwp.sspade import *
from rwp.environment import Troposphere
from rwp.kediffraction import *


__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class TestSSPade(unittest.TestCase):

    def test_flat_terrain__pec__hor_pol(self):
        environment = Troposphere(flat=True)
        environment.ground_material = PerfectlyElectricConducting()
        environment.z_max = 300
        max_range_m = 1000
        antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, elevation_angle=0, polarz='H')
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range_m,
                                                           comp_params=HelmholtzPropagatorComputationalParams(z_order=4))
        sspade_field = propagator.calculate()

        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=environment, max_range_m=max_range_m,
                                             x_grid_m=sspade_field.x_grid, z_grid_m=sspade_field.z_grid)
        ke_field = kdc.calculate()

        f1 = 10 * np.log10(1e-16 + np.abs(sspade_field.horizontal(50)))
        f2 = 10 * np.log10(1e-16 + np.abs(ke_field.horizontal(50)))
        self.assertTrue(np.linalg.norm(f1 - f2) / np.linalg.norm(f1) < 0.01)

    def test_flat_terrain__pec__ver_pol(self):
        environment = Troposphere(flat=True)
        environment.ground_material = PerfectlyElectricConducting()
        environment.z_max = 300
        max_range_m = 1000
        antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, elevation_angle=0, polarz='V')
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range_m,
                                                           comp_params=HelmholtzPropagatorComputationalParams(z_order=4))
        sspade_field = propagator.calculate()

        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=environment, max_range_m=max_range_m,
                                             x_grid_m=sspade_field.x_grid, z_grid_m=sspade_field.z_grid)
        ke_field = kdc.calculate()

        f1 = 10 * np.log10(1e-16 + np.abs(sspade_field.horizontal(50)))
        f2 = 10 * np.log10(1e-16 + np.abs(ke_field.horizontal(50)))
        self.assertTrue(np.linalg.norm(f1 - f2) / np.linalg.norm(f1) < 0.01)

    def test_std_atmo(self):
        environment = Troposphere()
        environment.ground_material = WetGround()
        environment.z_max = 300
        antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, elevation_angle=0, polarz='H')
        max_range = 150e3
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range)
        f = propagator.calculate()

        x1 = 80e3
        x2 = 150e3
        y1 = 10 * cm.log10(abs(f.value(x1, 30)))
        y2 = 10 * cm.log10(abs(f.value(x2, 30)))
        self.assertTrue(abs((y2 - y1) / (x2 - x1) * 1e3 - (-0.80)) < 0.01)

    def test_knife_edge_diffraction(self):
        environment = Troposphere(flat=True)
        environment.ground_material = PerfectlyElectricConducting()
        environment.z_max = 300
        environment.knife_edges = [KnifeEdge(range=200, height=50)]
        max_range_m = 300
        antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, elevation_angle=0, polarz='H')
        params = HelmholtzPropagatorComputationalParams(
            exp_pade_order=(10, 11),
            z_order=5,
            grid_optimizator_abs_threshold=5e-3)
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna,
                                                           env=environment,
                                                           max_range_m=max_range_m,
                                                           comp_params=params
                                                           )
        sspade_field = propagator.calculate()

        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=environment, max_range_m=max_range_m,
                                             x_grid_m=sspade_field.x_grid, z_grid_m=sspade_field.z_grid,
                                             p_grid_size=2000)
        ke_field = kdc.calculate()

        f1 = 10 * np.log10(1e-16 + np.abs(sspade_field.horizontal(50)))
        f2 = 10 * np.log10(1e-16 + np.abs(ke_field.horizontal(50)))
        self.assertTrue(np.linalg.norm(f1 - f2) / np.linalg.norm(f1) < 0.05)

    def test_terrain(self):
        env = Troposphere()
        env.z_max = 300
        max_range = 100000 / 5

        h = 110 / 2
        w = 10000 / 5
        x1 = 30000 / 5
        def elevation_func(x):
            return h / 2 * (1 + fm.sin(fm.pi * (x - x1) / (2 * w))) if -w <= (x - x1) <= 3 * w else 0
        env.terrain = Terrain(
            elevation=elevation_func,
            ground_material=VeryDryGround()
        )
        ant = GaussAntenna(freq_hz=3000e6, height=30, beam_width=2, elevation_angle=0, polarz='H')

        computational_params_pt = HelmholtzPropagatorComputationalParams(terrain_method=TerrainMethod.pass_through,
                                                                         z_order=4)
        pade_task_pt = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range,
                                                             comp_params=computational_params_pt)
        pade_field_pt = pade_task_pt.calculate()

        computational_params_sc = HelmholtzPropagatorComputationalParams(terrain_method=TerrainMethod.staircase,
                                                                         z_order=4)
        pade_task_sc = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range,
                                                             comp_params=computational_params_sc)
        pade_field_sc = pade_task_sc.calculate()

        f1 = 10 * np.log10(1e-16 + np.abs(pade_field_pt.horizontal_over_terrain(30, env.terrain)))
        f2 = 10 * np.log10(1e-16 + np.abs(pade_field_sc.horizontal_over_terrain(30, env.terrain)))
        self.assertTrue(np.linalg.norm(f1 - f2) / np.linalg.norm(f1) < 0.03)


if __name__ == '__main__':
    unittest.main()
