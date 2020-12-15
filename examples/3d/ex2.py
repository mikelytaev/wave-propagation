from rwp.urban import *
from rwp.antennas import GaussSource3D
from rwp.vis import FieldVisualiser3D
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

env = Manhattan3D(domain_width=200, domain_height=30, x_max=200)
#env.add_block(center=(100, -35), size=(210, 40, 20))
#env.add_block(center=(100, 35), size=(210, 40, 20))
# env.add_block(center=(170, -35), size=(100, 40, 17.5))
#env.add_block(center=(250, 0), size=(10, 50, 20))
ant = GaussSource3D(freq_hz=1000E6, height=4, ver_beamwidth=14, hor_beamwidth=60, polarz='V')

comp_params = FDUrbanPropagatorComputationParameters(dx_wl=5, dy_wl=1, dz_wl=1,
                                                         n_dx_out=1, n_dy_out=1, n_dz_out=1,
                                                         pade_order=(3, 4), abs_layer_scale=1)
urban_propagator = FDUrbanPropagator(env=env, comp_params=comp_params, freq_hz=ant.freq_hz)
field = urban_propagator.calculate(ant)

vis = FieldVisualiser3D(field=field, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)))

vis.plot_xz(y0=0, min_val=-50, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

vis.plot_xy(z0=1.5, min_val=-50, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('y, м')
plt.tight_layout()
plt.show()

vis.plot_yz(x0=50, min_val=-50, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

vis.plot_yz(x0=250, min_val=-50, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

vis.plot_x(y0=0, z0=4)
plt.xlabel('Расстояние, м')
plt.ylabel('Power (dBm)')
plt.grid(True)
plt.tight_layout()
plt.show()

# y_grid = urban_propagator.y_computational_grid
# z_grid = urban_propagator.z_computational_grid
# mask = env.intersection_mask_x(500, y_grid, z_grid)
# extent = [y_grid[0],y_grid[-1], z_grid[0], z_grid[-1]]
# plt.imshow(mask.T, extent=extent)
# plt.show()