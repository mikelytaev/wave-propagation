# Падение волн вертикальной поляризации под углом Брюстера

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
#from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 100
environment.ground_material = CustomMaterial(eps=7, sigma=0)

freq_hz = 3000e6
b_angle = brewster_angle(1, environment.ground_material.complex_permittivity(freq_hz))

#b_angle=70
antenna = GaussAntenna(freq_hz=freq_hz, height=50, beam_width=0.3, eval_angle=(90-b_angle), polarz='V')
h1 = antenna.height_m
h2 = 0
a = abs((h1 - h2) / cm.tan(abs(antenna.eval_angle) * cm.pi / 180))
max_range = 2 * a + 20 + 100 + 100
params = HelmholtzPropagatorComputationalParams(two_way=False,
                                                exp_pade_order=(7, 8),
                                                max_propagation_angle=abs(antenna.eval_angle)+5,
                                                z_order=4)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=params)
pade_field = pade_task.calculate()

computed_refl_coef = abs(pade_field.value(2 * a, h1)) / abs(pade_field.value(0, h1))
real_refl_coef = reflection_coef(1, environment.ground_material.complex_permittivity(antenna.freq_hz), b_angle, antenna.polarz)

print('reflection coef real: ' + str((real_refl_coef)))
print('reflection coef comp: ' + str(computed_refl_coef))

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade + NLBC', x_mult=1)

plt = pade_vis.plot2d(min=-50, max=0)
plt.xlabel('Расстояние, км')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.grid(True)
plt.show()

plt = pade_vis.plot_hor(50)
plt.xlabel('Расстояние, км')
plt.ylabel('10log|u| (дБ)')
plt.tight_layout()
plt.grid(True)
plt.show()

Miller_Brown_factor(10, 2*cm.pi / 0.03, 0.5)


tau = 1.001
t = np.linspace(-cm.pi, cm.pi, 1000)
t2 = tau * np.exp(1j * t)
wl = 0.1
k0 = 2 * cm.pi / wl
dx_m = 1.5 * wl
theta = np.arccos(1 + 1 / (1j * k0 * dx_m) * np.log(t2)) * 180 / cm.pi
plt.plot(t, theta.real, t, theta.imag)
refls = np.array([reflection_coef(1, 3, 90-t, "V") for t in theta])
plt.plot(t, refls.real, t, refls.imag)
t3 = np.linspace(-90, 90, 1000)
refls = np.array([reflection_coef(1, 3, t, "V") for t in t3*1j])
plt.plot(t3, refls.real, t3, refls.imag)