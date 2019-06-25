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
        pp = HelmholtzPadeSolver(env=env, wavelength=0.03, pade_order=(1, 1), tol=1e-11)
        pp.dx = 10
        pp.n_x = 12001
        pp.dz = 1
        pp.n_z = 301
        pp.calc_nlbc()

    def tomas_method_test(self):
        n = 10
        b = np.random.rand(n) + 1j*np.random.rand(n)
        a = np.random.rand(n-1) + 1j*np.random.rand(n-1)
        c = np.random.rand(n-1) + 1j*np.random.rand(n-1)
        matrix = np.diag(b, 0) + np.diag(a, -1) + np.diag(c, 1)
        rhs = np.random.rand(n) + 1j*np.random.rand(n)
        res1 = np.linalg.solve(matrix, rhs)
        res2 = tridiag_method(a, b, c, rhs)
        self.assertTrue(np.linalg.norm(res1 - res2)/np.linalg.norm(res1) < 1e-5)


if __name__ == '__main__':
    TestSSPade.main()
