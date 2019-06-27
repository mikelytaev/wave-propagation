import unittest

from propagators.sspade import *
from uwa.source import GaussSource

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class HelmholtzPropagatorTest(unittest.TestCase):

    def test1(self):
        logging.basicConfig(level=logging.DEBUG)
        env = HelmholtzEnvironment(x_max_m=1000,
                                   z_min=0,
                                   z_max=300,
                                   lower_bc=RobinBC(1, 0, 0),
                                   upper_bc=RobinBC(1, 0, 0),
                                   use_n2minus1=False,
                                   use_rho=False)

        src = GaussSource(freq_hz=1, depth=150, beam_width=2, eval_angle=0)
        params = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8), max_src_angle=src.max_angle(), dz_wl=1, dx_wl=1000)
        propagator = HelmholtzPadeSolver(env=env, wavelength=1, freq_hz=300e6, params=params)
        initials_fw = [np.empty(0)] * propagator.n_x
        initials_fw[0] = np.array([src.aperture(2*cm.pi, z) for z in propagator.z_computational_grid])
        propagator.propagate(initials=initials_fw, direction=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    HelmholtzPropagatorTest.main()
