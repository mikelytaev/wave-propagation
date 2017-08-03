from rwp.WPDefs import *
from rwp.SSPade import *
from rwp.WPVis import *
from scipy.interpolate import interp1d
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    env = EMEnvironment()
    env.z_min, env.z_max = 0, 300
    #env.N_profile = lambda x, z: 2 * z / 6371000
    #env.upper_boundary = TransparentLinearBS(2 / 6371000)
    env.lower_boundary = EarthSurfaceBC(wavelength=0.1, conductivity=17.03, permittivity=53.44, polarz='H')
    env.terrain = Terrain([0, 499, 500, 501, 1000], [0, 0, 100, 0, 0])
    max_range = 1000
    pp = PadePropagator(env=env, wavelength=1, pade_order=(7, 8), dx_wl=1, dz_wl=0.2, tol=1e-11)
    nlbc = NLBCManager()
    lower_nlbc, upper_nlbc = nlbc.get_NLBC(pp, max_range)
    src = gauss_source(pp.k0, 100, 45, 0, polarz='H')
    field = pp.propagate(max_range, src, lower_nlbc=lower_nlbc, upper_nlbc=upper_nlbc, n_dz_out=1, n_dx_out=1)
    vis = FieldVisualiser(field, lambda v: 10*fm.log10(1e-16+abs(v)))
    vis.plot_hor(30).show()
    #vis.plot2d(10*fm.log10(pp.tol)+5, 0).show()
    vis.plot2d(-50, 0).show()