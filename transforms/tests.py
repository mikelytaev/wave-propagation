__author__ = 'Mikhail'

import unittest
from transforms.fcc import *
from numpy.linalg import norm

# test values from matlab implementation
# http://www.unavarra.es/personal/victor dominguez/clenshawcurtisrule

class TestFCC(unittest.TestCase):
    tol = 1e-14

    def test_chebyshev_weights_asymptotics(self):
        val, flag = chebyshev_weights_asymptotics(100+0j, 10)
        self.assertTrue(abs(val-(-0.554883809653741)) < self.tol * abs(val))

        val, flag = chebyshev_weights_asymptotics(100, 10000)
        self.assertTrue(abs(val - (-5.063654255299453e-05)) < self.tol * abs(val))

        val, flag = chebyshev_weights_asymptotics(10000000, 1000)
        self.assertTrue(abs(val - (7.368244829894444e+04)) < self.tol * abs(val))

        val, flag = chebyshev_weights_asymptotics(10000000, 10)
        self.assertTrue(abs(val - (7.483142625203647e+34)) < self.tol * abs(val))

    def test_chebyshev_weights(self):
        w, rho = chebyshev_weights(1, 4)
        w_true = np.array([1.682941969615793 + 0.000000000000000j,
                           0.000000000000000 + 0.602337357879513j,
                           -0.726407461902261 + 0.000000000000000j,
                           0.000000000000000 - 0.390223474302468j,
                           -0.013969099000570 + 0.000000000000000j])
        rho_true = np.array([1.682941969615793 + 0.000000000000000j,
                             0.000000000000000 + 1.204674715759027j,
                             0.230127045811270 + 0.000000000000000j,
                             0.000000000000000 + 0.424227767154091j,
                             0.000000000000000 + 0.000000000000000j])
        self.assertTrue(norm(w_true-w) < self.tol)
        self.assertTrue(norm(rho_true - rho) < self.tol)

        w, rho = chebyshev_weights(10000000, 4)
        w_true = 1.0e-06*np.array([0.084109558638157 + 0.000000000000000j,
                            0.000000000000000 + 0.181454085647304j,
                            0.084109486056522 + 0.000000000000000j,
                            0.000000000000000 + 0.181454152934907j,
                            0.084109268311566 + 0.000000000000000j])
        rho_true = 1.0e-06*np.array([0.084109558638157 + 0.000000000000000j,
                            0.000000000000000 + 0.362908171294608j,
                            0.252328530751201 + 0.000000000000000j,
                            0.000000000000000 + 0.725816477164422j,
                            0.000000000000000 + 0.000000000000000j])
        self.assertTrue(norm(w_true-w) < self.tol)
        self.assertTrue(norm(rho_true - rho) < self.tol)






if __name__ == '__main__':
    TestFCC.main()
