from rwp.sspade import *
from rwp.vis import *
from scipy.io import loadmat
from cheb_pade_coefs import *


logging.basicConfig(level=logging.DEBUG)

dx_wl = 6
coefs, a0 = cheb_pade_coefs(dx_wl, (9, 9), fm.sin(89*fm.pi/180)**2, 'aaa')

env = Troposphere(flat=True)
env.z_max = 300
env.knife_edges = [KnifeEdge(range=0.5e3, height=150)]

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=20, eval_angle=0, polarz='H')

max_range_m = 1e3

cheb_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      modify_grid=False,
                                                      two_way=False,
                                                      max_propagation_angle=85,
                                                      z_order=4,
                                                      exp_pade_coefs=coefs,
                                                      exp_pade_a0_coef=a0,
                                                      #exp_pade_order=(8, 9),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.25,
                                                      z_output_filter=8
                                                  ))
cheb_pade_field = cheb_pade_task.calculate()

pade_vis = FieldVisualiser(cheb_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Pade")
plt = pade_vis.plot2d(min=-120, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()