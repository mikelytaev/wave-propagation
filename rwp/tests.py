import unittest

from rwp.sspade import *

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class TestSSPade(unittest.TestCase):

    def test_terrain_max_propagation_angle__flat_terrain__expect_zero_angle(self):
        terrain = Terrain()

        max_angle = terrain_max_propagation_angle(terrain=terrain, distance_m=1000)

        self.assertAlmostEqual(max_angle, 0.0)

    def test_terrain_max_propagation_angle__linear_terrain__expect_zero_angle(self):
        mult = -0.1
        terrain = Terrain(lambda x: mult * x)
        expected_max_angle = abs(fm.degrees(fm.atan(mult)))

        max_angle = terrain_max_propagation_angle(terrain=terrain, distance_m=1000)

        self.assertAlmostEqual(max_angle, expected_max_angle)


if __name__ == '__main__':
    unittest.main()