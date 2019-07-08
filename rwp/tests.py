import unittest

from rwp.SSPade import *
from rwp.environment import Troposphere


__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class TestSSPade(unittest.TestCase):

    def test_std_atmo(self):
        logging.basicConfig(level=logging.DEBUG)
        environment = Troposphere()
        environment.ground_material = PerfectlyElectricConducting()
        environment.z_max = 300
        antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
        propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=120e3, comp_params=HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8)))
        propagator.calculate()


if __name__ == '__main__':
    TestSSPade.main()
