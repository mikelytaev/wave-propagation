import unittest
from transforms.frft import *
from transforms.z_transform import *


class FRFTTest(unittest.TestCase):

    def xtest_1(self):
        size = 2048
        l = 20
        a = 1/2
        x_grid = get_fcft_grid(size, l)
        f_x = np.exp(-x_grid ** 2 * a)
        ff_x = fcft(f_x, l, l, l)
        # plt.plot(f_x)
        # plt.plot(ff_x)
        # plt.plot(abs(ff_x-f_x))
        # plt.show()
        ff_x_true = 1 / cm.sqrt(2 * a) * np.exp(-x_grid ** 2 / (4 * a))

        self.assertTrue(np.linalg.norm(ff_x - ff_x_true) < 1e-12)

        fff_x = ifcft(ff_x, l, l, l)
        self.assertTrue(np.linalg.norm(f_x - fff_x) < 1e-12)

    def xtest_2(self):
        size = 2048
        l = 20
        a = 1 / 2
        b = 1 / 3
        c = 1 / 2.5
        x_grid = get_fcft_grid(size, l)
        f_x = np.zeros((3, size))*0j
        f_x[0, :] = np.exp(-x_grid ** 2 * a)
        f_x[0, :] = np.exp(-x_grid ** 2 * b)
        f_x[0, :] = np.exp(-x_grid ** 2 * c)
        ff_x = fcft(f_x, l, l, l)
        ff_x_true = np.zeros((3, size))*0j
        ff_x_true[0, :] = 1 / cm.sqrt(2 * a) * np.exp(-x_grid ** 2 / (4 * a))
        ff_x_true[0, :] = 1 / cm.sqrt(2 * b) * np.exp(-x_grid ** 2 / (4 * b))
        ff_x_true[0, :] = 1 / cm.sqrt(2 * c) * np.exp(-x_grid ** 2 / (4 * c))
        self.assertTrue(np.linalg.norm(ff_x - ff_x_true) < 1e-12)
        fff_x = ifcft(ff_x, l, l, l)
        self.assertTrue(np.linalg.norm(f_x - fff_x) < 1e-12)

    def xtest_3(self):
        size = 2048
        l = 20
        a = 1 / 2
        offset = 1.1
        x_grid = get_fcft_grid(size, l)
        f_x = np.exp(-(x_grid - offset) ** 2 * a)
        ff_x = fcft(f_x, l, l, l)
        ff_x_true = 1 / cm.sqrt(2 * a) * np.exp(-1j * offset * x_grid) * np.exp(-x_grid ** 2 / (4 * a))

        self.assertTrue(np.linalg.norm(ff_x - ff_x_true) < 1e-12)

        fff_x = ifcft(ff_x, l, l, l)
        self.assertTrue(np.linalg.norm(f_x - fff_x) < 1e-12)


class InvZtrTest(unittest.TestCase):

    def test_1(self):
        a = 0.5
        f = lambda z: 1 / (1 - a/z)
        n = 5
        expected = a ** np.arange(0, n)
        res = inv_z_transform(f, n)
        self.assertTrue(np.linalg.norm(res - expected) < 1e-10)


if __name__ == '__main__':
    #FRFTTest.main()
    InvZtrTest.main()
