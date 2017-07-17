from rwp.WPDefs import *
from rwp.SSPade import *
from rwp.WPVis import *
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    env = EMEnvironment()
    env.z_max = 300
    env.N_profile = lambda x, z: z / 6371000 * 1e6
    env.upper_boundary = TransparentLinearBS(1 / 6371000 * 1e6)
    pp = PadePropagator(env=env, wave_length=0.1, pade_order=(3, 4), dx_wl=1000, dz_wl=1)
    src = gauss_source(pp.k0, 50, 10, 0)
    field = pp.propagate(50000, src)
    vis = FieldVisualiser(field, lambda v: 10*log10(1e-16+abs(v)))
    #vis.plot_hor(50).show()
    vis.plot2d(-20, 0).show()