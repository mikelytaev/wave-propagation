from rwp.sspade import *
from rwp.vis import *
from rwp.crank_nicolson import *
from propagators.sspade import TerrainMethod

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.ground_material = CustomMaterial(eps=15, sigma=1e-3)
terr = np.loadtxt('holm_terrain.txt')
env.terrain = InterpTerrain(terr[:, 0], terr[:, 1]-320, kind='linear')
vegetation_x = np.loadtxt('vegetation_x.txt')
env.vegetation = [Impediment(x1=a, x2=b, height=18, material=CustomMaterial(eps=1.004, sigma=180e-6))
                  for a, b in zip(vegetation_x[0::2], vegetation_x[1::2])]
ant = GaussAntenna(freq_hz=1599.5e6, height=360 + 22.9 - 320, beam_width=5, eval_angle=0, polarz='H')
max_range = 11000
comp_params = HelmholtzPropagatorComputationalParams(terrain_method=TerrainMethod.staircase)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range)
pade_field = pade_task.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[7/8] + NLBC', x_mult=1E-3)

plt = pade_vis.plot_hor(150)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()
#plt.savefig("elevated_hor.eps")

plt = pade_vis.plot2d(min=-50, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
#plt.savefig("ex4_pade.eps")

pade_pl_vis = FieldVisualiser(pade_field.path_loss(), label='Pade-[7/8] + NLBC', x_mult=1)

h51 = np.loadtxt('holm1599_5_51m.txt')
plt = pade_pl_vis.plot_ver_measurements(vegetation_x[-1] + 51, env.terrain(vegetation_x[-1] + 51), 0, 30, h51.transpose(), mea_label='Измерения', fit=0.0)
plt.xlabel('Высота, м')
plt.ylabel('Коэффициент потерь (dB)')
plt.tight_layout()
plt.show()
plt.savefig("ex4_pade_vs_measures_51m.eps")

h109 = np.loadtxt('holm1599_5_109m.txt')
plt = pade_pl_vis.plot_ver_measurements(vegetation_x[-1] + 109, env.terrain(vegetation_x[-1] + 109), 0, 30, h109.transpose(), mea_label='Измерения', fit=0.3)
plt.xlabel('Высота, м')
plt.ylabel('Коэффициент потерь (dB)')
plt.tight_layout()
plt.show()

# plt = holm_vis.plot2d(min=-50, max=0)
# plt.title('10log|u|')
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.tight_layout()
# plt.show()