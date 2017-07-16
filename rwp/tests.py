import unittest
from rwp.SSPade import *

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class TestSSPade(unittest.TestCase):
    tol = 1e-14

    def test_lentz(self):
        self.assertTrue(abs(lentz(lambda n: (n > 1) * 2.0 + (n < 2) * 1.0, self.tol)-fm.sqrt(2)) < self.tol)

if __name__ == '__main__':
    TestSSPade.main()
