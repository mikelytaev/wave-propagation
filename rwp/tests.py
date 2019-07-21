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
        environment.ground_material = PerfectlyElectricConducting()
        environment.z_max = 300
        antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=120e3, comp_params=HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8)))
        propagator.calculate()


if __name__ == '__main__':
    TestSSPade.main()
