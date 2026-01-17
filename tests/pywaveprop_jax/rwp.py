import unittest

from pywaveprop.experimental.rwp_jax import TroposphereModel, RWPGaussSourceModel
from pywaveprop.rwp.kediffraction import KnifeEdgeDiffractionCalculator


__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class TestSSPade(unittest.TestCase):

    def test_flat_terrain__pec__hor_pol(self):
        environment = TroposphereModel(slope=0)
        max_range_m = 1000
        antenna = RWPGaussSourceModel(
            freq_hz=300E6,
            height_m=50,
            beam_width_deg=15,
            elevation_angle_deg=0,
            polarz='H')

        model = create_rwp_model(antenna=antenna, env=environment, max_range_m=max_range_m,
                                                           comp_params=HelmholtzPropagatorComputationalParams(z_order=4))
        sspade_field = propagator.calculate()

        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=environment, max_range_m=max_range_m,
                                             x_grid_m=sspade_field.x_grid, z_grid_m=sspade_field.z_grid)
        ke_field = kdc.calculate()

        f1 = 10 * np.log10(1e-16 + np.abs(sspade_field.horizontal(50)))
        f2 = 10 * np.log10(1e-16 + np.abs(ke_field.horizontal(50)))
        self.assertTrue(np.linalg.norm(f1 - f2) / np.linalg.norm(f1) < 0.01)


if __name__ == '__main__':
    unittest.main()
