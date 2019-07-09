import unittest
from transforms.frft import *

import matplotlib.pyplot as plt


class FRFTTest(unittest.TestCase):

    def test_1(self):
        size = 2048
        a = 20
        x_grid = get_fcft_grid(size, a)
        f_x = 1/cm.sqrt(2*cm.pi) * np.exp(-x_grid ** 2 / 2)
        ff_x = fcft(f_x, a, a)
        plt.plot(f_x)
        plt.plot(ff_x)
        plt.plot(abs(ff_x-f_x))
        plt.show()

        self.assertTrue(np.linalg.norm(f_x - ff_x) < 1e-10)


if __name__ == '__main__':
    FRFTTest.main()
