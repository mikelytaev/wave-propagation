import unittest

from rwp.SSPade import *
from rwp.environment import TransparentLinearBS, EMEnvironment

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class TestSSPade(unittest.TestCase):
    tol = 1e-14

    def test_lentz(self):
        self.assertTrue(abs(lentz(lambda n: (n > 1) * 2.0 + (n < 2) * 1.0, self.tol)-fm.sqrt(2)) < self.tol)

    def nlbc_test(self):
        logging.basicConfig(level=logging.DEBUG)
        env = EMEnvironment()
        env.z_max = 300
        env.N_profile = lambda x, z: z / 6371000 * 1e6
        env.upper_boundary = TransparentLinearBS(1 / 6371000 * 1e6)
        pp = PadePropagator(env=env, wave_length=0.03, pade_order=(1, 1), tol=1e-11)
        pp.dx = 10
        pp.n_x = 12001
        pp.dz = 1
        pp.n_z = 301
        pp.calc_nlbc()

if __name__ == '__main__':
    TestSSPade.main()
