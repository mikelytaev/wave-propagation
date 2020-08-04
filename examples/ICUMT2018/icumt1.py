from rwp.sspade import *
from rwp.vis import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.z_max = 200

h = 50
w = 1000
x1 = 3000

env.terrain = Terrain(
    elevation=lambda x: h/2*(1 + fm.sin(fm.pi * (x - x1) / (2*w))) if -w <= (x-x1) <= 3*w else 0,
    ground_material=PerfectlyElectricConducting()
)
ant60 = GaussAntenna(freq_hz=60000e6, height=10, beam_width=5, eval_angle=0, polarz='H')
ant70 = GaussAntenna(freq_hz=70000e6, height=10, beam_width=5, eval_angle=0, polarz='H')
max_range = 10000

comp_params60 = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8), dx_wl=100, x_output_filter=10, dz_wl=1,
                                                       z_output_filter=20, storage=PickleStorage())
pade_task60 = TroposphericRadioWaveSSPadePropagator(antenna=ant60, env=env, max_range_m=max_range, comp_params=comp_params60)
pade_field60 = pade_task60.calculate()

comp_params70 = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8), dx_wl=100, x_output_filter=10, dz_wl=1,
                                                       z_output_filter=20, storage=PickleStorage())
pade_task70 = TroposphericRadioWaveSSPadePropagator(antenna=ant70, env=env, max_range_m=max_range, comp_params=comp_params70)
pade_field70 = pade_task70.calculate()

pade_vis60 = FieldVisualiser(pade_field60.path_loss(gamma=14.1956), label='60 GHz', x_mult=1e-3)
pade_vis70 = FieldVisualiser(pade_field70.path_loss(gamma=0.5157), label='70 GHz', x_mult=1e-3)

plt = pade_vis60.plot_hor(10, pade_vis70)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = pade_vis60.plot2d(min=100, max=400)
plt.title('Path loss (dB)')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis70.plot2d(min=100, max=400)
plt.title('Path loss (dB)')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

pade12_vis70rain = FieldVisualiser(pade_field70.path_loss(gamma=0.5157 + 6.1869), label='70 GHz, 12 mm/h', x_mult=1e-3)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
pade12_vis70rain.plot_ver(2 * 1e3, ax1, pade_vis70)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('Path loss (dB)')

pade12_vis70rain.plot_ver(10 * 1e3, ax2, pade_vis70)
ax2.set_ylabel('Height (m)')
ax2.set_xlabel('Path loss (dB)')

f.tight_layout()
f.show()