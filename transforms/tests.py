__author__ = 'Mikhail'

import unittest
from transforms.fcc import *

class TestFCC(unittest.TestCase):

    def test_chebyshev_weights_asymptotics(self):
        val, flag = chebyshev_weights_asymptotics(100, 10)
        self.assertTrue(abs(val-(-0.554883809653741)) < 1e-14*abs(val))

        val, flag = chebyshev_weights_asymptotics(100, 10000)
        self.assertTrue(abs(val - (-5.063654255299453e-05)) < 1e-14 * abs(val))

        val, flag = chebyshev_weights_asymptotics(10000000, 1000)
        self.assertTrue(abs(val - (7.368244829894444e+04)) < 1e-14 * abs(val))

        val, flag = chebyshev_weights_asymptotics(10000000, 10)
        self.assertTrue(abs(val - (7.483142625203647e+34)) < 1e-14 * abs(val))

    def test_chebyshev_weights(self):



if __name__ == '__main__':
    TestFCC.main()
