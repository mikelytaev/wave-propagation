__author__ = 'Mikhail'

import unittest
from transforms.fcc_fourier import *
from numpy.linalg import norm
import sympy as sy
import math as fm

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

    def test_FFCFourier(self):
        fcc = FCCFourier(-1, 7, np.array([1, 10, 100]))
        self.assertTrue(fcc.fw.shape == (3, 8))
        self.assertTrue(abs(fcc.fw[0, 0] - (-0.008954924100922 - 0.004892097332696j)) < self.tol)
        self.assertTrue(abs(fcc.fw[2, 7] - (-0.000550686115736 - 0.009845952674564j)) < self.tol)

        fcc = FCCFourier(3, 7, np.array([1, 10, 100]))
        tf = fcc.forward(np.sin(chebyshev_grid(1, 4, 7) ** 2), 1, 4)
        tf_true = np.array([0.074342089312286 - 0.241773913413446j,
                            0.202737233915876 - 0.022917127345782j,
                            0.007683316767495 - 0.007264724014208j])
        self.assertTrue(norm(tf-tf_true[:, np.newaxis]) < self.tol)

    def test_FCCAdaptiveFourier(self):
        fcca = FCCAdaptiveFourier(2*fm.pi, np.array([1, 10, 100]), rtol=1e-12)
        x = sy.symbols('x')
        test_func = (x**2+x+10)*sy.sin(100*x)
        exact = complex(sy.integrate(test_func * sy.exp(-1j*10*x), (x, 0, 2*fm.pi)))
        int_val = fcca.forward(lambda t: np.array([complex(test_func.subs(x, t)), 2*complex(test_func.subs(x, t))]), 0, 2*fm.pi)
        self.assertTrue(norm(exact - int_val[1, 1]/2) < abs(fcca.rtol * exact))


if __name__ == '__main__':
    TestFCC.main()
