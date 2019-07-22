import unittest

from rwp.SSPade import *
from rwp.environment import Troposphere
from rwp.kediffraction import *


__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class TestSSPade(unittest.TestCase):

    def test_flat_h(self):
        environment = Troposphere(flat=True)
        environment.ground_material = PerfectlyElectricConducting()
        environment.z_max = 300
        max_range_m = 1000
        antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, eval_angle=0, polarz='H')
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range_m,
                                                           comp_params=HelmholtzPropagatorComputationalParams(
                                                               exp_pade_order=(7, 8)))
        sspade_field = propagator.calculate()

        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=environment, max_range_m=max_range_m,
                                             x_grid_m=sspade_field.x_grid, z_grid_m=sspade_field.z_grid)
        ke_field = kdc.calculate()

        f1 = 10 * np.log10(1e-16 + np.abs(sspade_field.horizontal(50)))
        f2 = 10 * np.log10(1e-16 + np.abs(ke_field.horizontal(50)))
        self.assertTrue(np.linalg.norm(f1 - f2) / np.linalg.norm(f1) < 0.01)

    def test_flat_v(self):
        environment = Troposphere(flat=True)
        environment.ground_material = PerfectlyElectricConducting()
        environment.z_max = 300
        max_range_m = 1000
        antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, eval_angle=0, polarz='V')
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range_m,
                                                           comp_params=HelmholtzPropagatorComputationalParams(
                                                               exp_pade_order=(7, 8)))
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
        antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
        max_range = 150e3
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                           comp_params=HelmholtzPropagatorComputationalParams(
                                                               exp_pade_order=(7, 8)))
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
        antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, eval_angle=0, polarz='H')
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range_m,
                                                           comp_params=HelmholtzPropagatorComputationalParams(
                                                               exp_pade_order=(7, 8)))
        sspade_field = propagator.calculate()

        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=environment, max_range_m=max_range_m,
                                             x_grid_m=sspade_field.x_grid, z_grid_m=sspade_field.z_grid,
                                             p_grid_size=2000)
        ke_field = kdc.calculate()

        f1 = 10 * np.log10(1e-16 + np.abs(sspade_field.horizontal(50)))
        f2 = 10 * np.log10(1e-16 + np.abs(ke_field.horizontal(50)))
        self.assertTrue(np.linalg.norm(f1 - f2) / np.linalg.norm(f1) < 0.05)


if __name__ == '__main__':
    TestSSPade.main()
