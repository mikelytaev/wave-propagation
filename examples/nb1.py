from rwp.WPDefs import *
from rwp.SSPade import *
from rwp.WPVis import *
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    env = EMEnvironment()
    env.z_max = 300
    env.N_profile = lambda x, z: 2 * z / 6371000
    env.upper_boundary = TransparentLinearBS(2 / 6371000)
    max_range = 150000
    env.lower_boundary = Earth_surface(wave_length=0.03, conductivity=17.03, permittivity=53.44, polarz='H')
    pp = PadePropagator(env=env, wave_length=0.03, pade_order=(1, 2), dx_wl=100/0.03, dz_wl=1, tol=1e-11)
    nlbc = NLBCManager()
    nlbc_coefs = nlbc.getNLBC(pp, 0) if nlbc.getNLBC(pp, 0).shape[0] > 0 else pp.calc_nlbc(max_range)
    src = gauss_source(pp.k0, 30, 2, 0)
    field = pp.propagate(max_range, src, nlbc_coefs=nlbc_coefs, n_dz_out=10, n_dx_out=1)
    nlbc.setNLBC(pp, nlbc_coefs)
    vis = FieldVisualiser(field, lambda v: 10*log10(1e-16+abs(v)))
    vis.plot_hor(30).show()
    vis.plot2d(10*fm.log10(pp.tol)+5, 0).show()