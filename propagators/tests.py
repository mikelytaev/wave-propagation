import unittest

from propagators.sspade import *
from uwa.source import GaussSource

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


def energy_conservation(field: Field, eps=1e-11) -> bool:
    norms = np.linalg.norm(field.field, axis=1)
    return np.all(np.abs(norms - norms[0]) < eps)


def local_bc(lbc):
    env = HelmholtzEnvironment(x_max_m=1000,
                               z_min=0,
                               z_max=300,
                               lower_bc=lbc,
                               upper_bc=lbc,
                               use_n2minus1=False,
                               use_rho=False)

    src = GaussSource(freq_hz=1, depth=150, beam_width=15, eval_angle=0)
    params = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8), max_src_angle=src.max_angle(), dz_wl=0.5, dx_wl=50)
    propagator = HelmholtzPadeSolver(env=env, wavelength=100, freq_hz=300e6, params=params)
    initials_fw = [np.empty(0)] * propagator.n_x
    initials_fw[0] = np.array([src.aperture(2*cm.pi, z) for z in propagator.z_computational_grid])
    f, r = propagator.propagate(initials=initials_fw, direction=1)

    plt.imshow(10*np.log10(np.abs(f.field.T[::-1, :])), cmap=plt.get_cmap('jet'), norm=Normalize(-50, 10))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    return f


def transparent_const_bc():
    env = HelmholtzEnvironment(x_max_m=1000,
                               z_min=0,
                               z_max=300,
                               lower_bc=TransparentConstBC(),
                               upper_bc=TransparentConstBC(),
                               use_n2minus1=False,
                               use_rho=False)

    src = GaussSource(freq_hz=1, depth=150, beam_width=15, eval_angle=0)
    params = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8), max_src_angle=src.max_angle(), dz_wl=0.5,
                                                    dx_wl=50)
    propagator = HelmholtzPadeSolver(env=env, wavelength=0.1, freq_hz=300e6, params=params)
    initials_fw = [np.empty(0)] * propagator.n_x
    initials_fw[0] = np.array([src.aperture(2 * cm.pi, z) for z in propagator.z_computational_grid])
    f, r = propagator.propagate(initials=initials_fw, direction=1)

    plt.imshow(10 * np.log10(np.abs(f.field.T[::-1, :])), cmap=plt.get_cmap('jet'), norm=Normalize(-50, 10))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    return f


class HelmholtzPropagatorTest(unittest.TestCase):

    # def test_Dirichlet(self):
    #     logging.basicConfig(level=logging.DEBUG)
    #     f = local_bc(RobinBC(1, 0, 0))
    #     self.assertTrue(energy_conservation(f, eps=1e-11))
    #
    # def test_Neumann(self):
    #     logging.basicConfig(level=logging.DEBUG)
    #     f = local_bc(RobinBC(0, 1, 0))
    #     self.assertTrue(energy_conservation(f, eps=1e-11))

    def test_transparent_1(self):
        logging.basicConfig(level=logging.DEBUG)
        f = transparent_const_bc()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    HelmholtzPropagatorTest.main()
