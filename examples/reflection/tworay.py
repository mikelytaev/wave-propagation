# Incidence of vertically polarized waves at the Brewster angle

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from propagators.wavenumber import *
from rwp.petool import PETOOLPropagationTask
from rwp.tworay import *

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 100
environment.ground_material = CustomMaterial(eps=3, sigma=0)

freq_hz = 3000e6
b_angle = brewster_angle(1, environment.ground_material.complex_permittivity(freq_hz))

antenna = GaussAntenna(freq_hz=freq_hz, height=50, beam_width=5, eval_angle=(90-b_angle).real, polarz='V')
h1 = antenna.height_m
h2 = 0
a = abs((h1 - h2) / cm.tan(abs(antenna.eval_angle) * cm.pi / 180))
max_range = 2 * a + 200

trm = TwoRayModel(src=antenna, env=environment)
x_grid_m = np.linspace(1, max_range, 5000)
z_grid_m = np.linspace(1, environment.z_max, 100)
trm_f = trm.calculate(x_grid_m, z_grid_m)

field = Field(x_grid=x_grid_m, z_grid=z_grid_m, freq_hz=antenna.freq_hz)
field.field[:, :] = trm_f
vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Two-ray model')
plt = vis.plot2d(min=-70, max=0)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

plt = vis.plot_hor(50)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()
