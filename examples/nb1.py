from rwp.WPDefs import *
from rwp.SSPade import *
from rwp.WPVis import *
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    env = EMEnvironment()
    env.z_max = 100
    pp = PadePropagator(env=env, wave_length=0.1, pade_order=(2, 3), dx_wl=100, dz_wl=1)
    src = gauss_source(pp.k0, 50, 2, 0)
    field = pp.propagate(2000, src)
    vis = FieldVisualiser(field, lambda v: 10*log10(eps+abs(v)))
    #vis.plot_hor(50).show()
    vis.plot2d(-20, 0).show()