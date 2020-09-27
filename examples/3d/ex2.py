from rwp.urban import *
from rwp.antennas import GaussSource3D
from rwp.vis import FieldVisualiser3D
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

env = Manhattan3D(domain_width=200, domain_height=100, x_max=200)
env.add_block(center=(50, -35), size=(80, 50, 20))
env.add_block(center=(50, 35), size=(80, 50, 15))
env.add_block(center=(170, -35), size=(100, 50, 25))
env.add_block(center=(150, 50), size=(100, 40, 15))
ant = GaussSource3D(freq_hz=900E6, height=5, ver_beamwidth=10, hor_beamwidth=10, polarz='H')

comp_params = FDUrbanPropagatorComputationParameters(dx_wl=0.5, dy_wl=0.5, dz_wl=0.5,
                                                         n_dx_out=2, n_dy_out=2, n_dz_out=2,
                                                         pade_order=(3, 4), abs_layer_scale=0.25)
urban_propagator = FDUrbanPropagator(env=env, comp_params=comp_params, freq_hz=ant.freq_hz)
field = urban_propagator.calculate(ant)

vis = FieldVisualiser3D(field=field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)))

vis.plot_xz(y0=0, min_val=-45, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

vis.plot_xy(z0=5, min_val=-60, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('y, м)')
plt.tight_layout()
plt.show()

vis.plot_yz(x0=90, min_val=-60, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

vis.plot_yz(x0=800, min_val=-45, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

# y_grid = urban_propagator.y_computational_grid
# z_grid = urban_propagator.z_computational_grid
# mask = env.intersection_mask_x(500, y_grid, z_grid)
# extent = [y_grid[0],y_grid[-1], z_grid[0], z_grid[-1]]
# plt.imshow(mask.T, extent=extent)
# plt.show()