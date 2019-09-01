from propagators.wavenumber import *
from rwp.field import *
from rwp.vis import *
import logging

logging.basicConfig(level=logging.DEBUG)
wnparams = WaveNumberIntegratorParams(alpha=1e-4, fcc_tol=1e-9, x_grid_m=np.linspace(1, 10000, 1000), z_grid_m=np.linspace(0, 100, 100))
wavelength = 1
wnp = WaveNumberIntegrator(k0=2*cm.pi / wavelength, q_func=DeltaFunction(x_c=50), params=wnparams)
res = wnp.calculate()

field = Field(x_grid=wnparams.x_grid_m, z_grid=wnparams.z_grid_m, freq_hz=300e6)
field.field[:, :] = res
vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-40, max=0)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()