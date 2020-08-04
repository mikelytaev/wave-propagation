import unittest

from propagators.sspade import *
from uwa.source import GaussSource

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class HelmholtzPropagatorTest(unittest.TestCase):

    def test_tomas_method(self):
        n = 10
        b = np.random.rand(n) + 1j*np.random.rand(n)
        a = np.random.rand(n-1) + 1j*np.random.rand(n-1)
        c = np.random.rand(n-1) + 1j*np.random.rand(n-1)
        matrix = np.diag(b, 0) + np.diag(a, -1) + np.diag(c, 1)
        rhs = np.random.rand(n) + 1j*np.random.rand(n)
        res1 = np.linalg.solve(matrix, rhs)
        res2 = np.empty(res1.shape, dtype=complex)
        tridiag_method(a, b, c, rhs, res2)
        self.assertTrue(np.linalg.norm(res1 - res2)/np.linalg.norm(res1) < 1e-5)

    def test_lentz(self):
        tol = 1e-14
        self.assertTrue(abs(lentz(lambda n: (n > 1) * 2.0 + (n < 2) * 1.0, tol) - fm.sqrt(2)) < tol)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    HelmholtzPropagatorTest.main()
