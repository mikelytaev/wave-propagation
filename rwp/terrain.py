import numpy as np
from geographiclib.geodesic import Geodesic


def geodesic_problem(lat, long, azi, x_grid):
    """
    solves direct geodesic problem for array of points
    :param lat: latitude of starting point (deg)
    :param long: longitude of starting point (deg)
    :param azi: azimuth (deg)
    :param x_grid: ranges from starting point (m)
    :return: list [(lat, long)] of coordinates of points
    """
    line = Geodesic.WGS84.Line(lat, long, azi)
    pozs = [line.Position(v) for v in x_grid]
    return [(pos['lat2'], pos['lon2']) for pos in pozs]


def inv_geodesic_problem(lat1, long1, lat2, long2, n_points):
    geo_dic = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)
    range = geo_dic['s12']
    azi = geo_dic['azi1']
    x_grid = np.linspace(0, range, n_points)
    return geodesic_problem(lat1, long1, azi, x_grid), x_grid