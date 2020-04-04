from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from propagators.wavenumber import *

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 100
environment.ground_material = PerfectlyElectricConducting()
profile1d = interp1d(x=[0, 70, 100], y=[330-330, 320-330, 0], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)
freq_hz = 3000e6

antenna = GaussAntenna(freq_hz=freq_hz, height=25, beam_width=2, eval_angle=0, polarz='H')
max_range = 100000

pade_params = HelmholtzPropagatorComputationalParams(two_way=False,
                                                     exp_pade_order=(7, 8),
                                                     max_propagation_angle=abs(antenna.eval_angle)+5,
                                                     z_order=4)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=pade_params)
pade_field = pade_task.calculate()

k0 = 2*cm.pi / antenna.wavelength
wnparams = WaveNumberIntegratorParams(fcc_tol=1e-9,
                                      x_grid_m=np.linspace(1, max_range, 1500),
                                      z_computational_grid_m=np.linspace(0, 100, 201),
                                      z_out_grid_m=np.linspace(0, 100, 201),
                                      alpha=5e-7,
                                      max_p_k0=10,
                                      lower_refl_coef=lambda theta: -1,
                                      het=lambda z: k0**2*profile1d(z)*2e-6)
#wnp = WaveNumberIntegrator(k0=2*cm.pi / antenna.wavelength, initial_func=lambda z: antenna.aperture(z), params=wnparams)
wnp = WaveNumberIntegrator(k0=k0, initial_func=DeltaFunction(x_c=antenna.height_m), params=wnparams)
res = wnp.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC', x_mult=1)

plt = pade_vis.plot2d(min=-30, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.show()

# plt = pade_vis.plot_hor(50)
# plt.xlabel('Range (km)')
# plt.ylabel('10log|u| (dB)')
# plt.tight_layout()
# plt.grid(True)
# plt.show()

wn_field = Field(x_grid=wnparams.x_grid_m, z_grid=wnparams.z_out_grid_m, freq_hz=300e6)
wn_field.field[:, :] = res
wn_vis = FieldVisualiser(wn_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Wavenumber integration')
plt = wn_vis.plot2d(min=-50, max=0)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

plt = wn_vis.plot_hor(50, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()