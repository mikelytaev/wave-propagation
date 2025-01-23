import importlib
import logging

import googlemaps
import numpy as np
from geographiclib.geodesic import Geodesic
from geopy.distance import geodesic
from scipy.interpolate import interp1d


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


def get_elevation_func(lat1: float, long1: float, lat2: float, long2: float, n_points: int):
    from podpac.datalib.terraintiles import TerrainTiles
    from podpac import Coordinates
    from podpac import settings

    settings['DEFAULT_CACHE'] = ['disk']
    node = TerrainTiles(tile_format='geotiff', zoom=11)
    coords, x_grid = inv_geodesic_problem(lat1, long1, lat2, long2, n_points)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    c = Coordinates([lats, lons], dims=['lat', 'lon'])
    o = node.eval(c)
    eval = np.array([o.data[i, i] for i in range(0, len(x_grid))])
    eval[np.isnan(eval)] = 0
    eval = np.array([max(a, 0) for a in eval])

    #podpac делает какую-то дичь с логгером
    importlib.reload(logging)

    return interp1d(x=x_grid, y=eval, fill_value="extrapolate")


def get_elevation_gmap(start_point, end_point, api_key="AIzaSyDefFxClGxerq" + "B0Rzkas5OnkzDY9eaMsVc", samples=100):
    """
    Fetches the elevation profile between two geographical points.

    Parameters:
        api_key (str): Google Maps API key.
        start_point (tuple): Starting point as (latitude, longitude).
        end_point (tuple): Ending point as (latitude, longitude).
        samples (int): Number of points to sample along the path (default is 100).

    Returns:
        list: A list of elevations corresponding to the sampled points.
        list: A list of distances along the sampled path in meters.
    """
    # Initialize the Google Maps client
    gmaps = googlemaps.Client(key=api_key)

    # Generate points along the path
    latitudes = np.linspace(start_point[0], end_point[0], samples)
    longitudes = np.linspace(start_point[1], end_point[1], samples)
    path = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]

    # Fetch elevation data
    elevations = []
    for i in range(0, len(path), 512):  # Google API limits requests to 512 locations per call
        chunk = path[i:i+512]
        response = gmaps.elevation(chunk)
        elevations.extend([result['elevation'] for result in response])

    # Calculate distances along the path in meters
    distances = [geodesic(start_point, point).meters for point in path]

    return elevations, distances
