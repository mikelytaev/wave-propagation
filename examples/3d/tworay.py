from rwp.tworay import *
from rwp.field import *
from rwp.vis import *


ant = GaussAntenna(freq_hz=5900e6, height=4, beam_width=14, eval_angle=0, polarz='V')#Source(freq_hz=5900e6, height_m=4, polarz='V')#
env = Troposphere(flat=True)
env.z_max = 20
env.terrain = Terrain(ground_material=CustomMaterial(eps=1, sigma=1000000000))
trm = TwoRayModel(src=ant, env=env)
x_grid_m = np.arange(1, 700)
z_grid_m = np.linspace(0, 20, 200)
#z_grid_m = np.array([1.5])
trm_f = trm.calculate(x_grid_m, z_grid_m)

trm_f = trm.calculate(x_grid_m, z_grid_m)
trm_field = Field(x_grid=x_grid_m, z_grid=z_grid_m, freq_hz=ant.freq_hz)
trm_field.field[:, :] = trm_f

trm_vis = FieldVisualiser(trm_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v))-26.5,
                          label='Two-ray model', x_mult=1)

plt = trm_vis.plot_hor(1.5)
plt.grid(True)
plt.show()

plt = trm_vis.plot2d(-100, 0)
plt.show()