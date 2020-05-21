from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *
from propagators.wavenumber import *


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.ground_material = SaltWater()
environment.rms_m = 1
environment.z_max = 100
max_range = 50e3

profile1d = interp1d(x=[0, 5, 70, 100, 300], y=[0, 0, -10, 0, 0], fill_value="extrapolate")
#environment.M_profile = lambda x, z: profile1d(z)

antenna = GaussAntenna(freq_hz=10000e6, height=25, beam_width=2, eval_angle=0, polarz='H')

sspe_propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna,
                                                        env=environment,
                                                        max_range_m=max_range,
                                                        comp_params=
                                                        HelmholtzPropagatorComputationalParams(
                                                            exp_pade_order=(7, 8),
                                                            z_order=4,
                                                            #max_propagation_angle=abs(antenna.beam_width)+5,
                                                            #storage=PickleStorage('nlbc_ex4_refl')
                                                        ),
                                                        )
sspe_field = sspe_propagator.calculate()
sspe_vis = FieldVisualiser(sspe_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC')

k0 = 2*cm.pi / antenna.wavelength
alpha = 1e-6

mbf = MillerBrownFactor(7)
def refl_coef(theta):
    #print("theta = " + str(theta))
    return mbf.factor(theta, k0, environment.rms_m) * \
           reflection_coef(1+1j*alpha, environment.ground_material.complex_permittivity(antenna.freq_hz), 90-theta, antenna.polarz)

wnparams = WaveNumberIntegratorParams(fcc_tol=1e-7,
                                      x_grid_m=np.linspace(1, max_range, 1500),
                                      z_computational_grid_m=np.linspace(0, environment.z_max, 301),
                                      z_out_grid_m=np.array([0, 25, 100]),#np.linspace(0, environment.z_max, 301),
                                      alpha=alpha,
                                      alpha_compensate=True,
                                      #min_p_k0=0.99,
                                      max_p_k0=10,
                                      lower_refl_coef=refl_coef,
                                      #het=lambda z: k0**2*profile1d(z)*2e-6
                                      )
wnp = WaveNumberIntegrator(k0=k0, initial_func=DeltaFunction(x_c=antenna.height_m), params=wnparams)
#wnp = WaveNumberIntegrator(k0=k0, initial_func=lambda z: antenna.aperture(z), params=wnparams)
wnp_f = wnp.calculate()

wn_field = Field(x_grid=wnparams.x_grid_m, z_grid=wnparams.z_out_grid_m, freq_hz=antenna.freq_hz)
wn_field.field[:, :] = wnp_f
wn_vis = FieldVisualiser(wn_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v))+22.3, label='Wavenumber integration')
plt = wn_vis.plot2d(min=-30, max=0)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

plt = sspe_vis.plot2d(min=-30, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = sspe_vis.plot_hor(25, wn_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()
