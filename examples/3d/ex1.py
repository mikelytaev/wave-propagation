from rwp.urban import *
from rwp.antennas import GaussSource3D
from rwp.vis import FieldVisualiser3D
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

env = StreetCanyon3D(domain_width=50, domain_height=100, street_width=30, building_height=50, x_max=1000)
ant = GaussSource3D(freq_hz=2000E6, height=50, ver_beamwidth=30, hor_beamwidth=30, polarz='H')

comp_params = FDUrbanPropagatorComputationParameters(dx_wl=5, dy_wl=0.5, dz_wl=0.5,
                                                         n_dx_out=10, n_dy_out=10, n_dz_out=10,
                                                         pade_order=(3, 4), abs_layer_scale=0.25)
urban_propagator = FDUrbanPropagator(env=env, comp_params=comp_params, freq_hz=ant.freq_hz)
field = urban_propagator.calculate(ant)

vis = FieldVisualiser3D(field=field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)))

vis.plot_xz(y0=0, min_val=-45, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

vis.plot_xy(z0=5, min_val=-45, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('y, м)')
plt.tight_layout()
plt.show()

vis.plot_yz(x0=300, min_val=-45, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()

vis.plot_yz(x0=900, min_val=-45, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()