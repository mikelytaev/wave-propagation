from propagators._utils import *
import matplotlib.pyplot as plt

wl = 0.1
k0 = 2*cm.pi / wl
dx = 1
dz = 0.1
coefs = pade_propagator_coefs(pade_order=(1, 1), diff2=lambda x: x, k0=k0, dx=dx, spe=False)
(a, b) = coefs[0]

tau = 1.2
xi = tau * np.exp(1j*np.linspace(0, 2*fm.pi, 1000))
gamma2 = (1 - xi) / (-a + b*xi)
gamma = np.sqrt(gamma2)
mu = np.exp(1j*gamma*dz)

mbf = MillerBrownFactor(8)
rms = 1
roots = np.array(mbf.a_roots) / (2*rms**2)
roots_real = [x.real for x in roots]
roots_imag = [x.imag for x in roots]

X = [x.real for x in gamma2]
Y = [x.imag for x in gamma2]
plt.scatter(X, Y, color='red', s=1)
plt.scatter(roots_real, roots_imag, color='blue', s=2)
plt.grid(True)
plt.show()

# mbf = np.exp(-4*gamma2*k0**2)
# X = [x.real for x in mbf]
# Y = [x.imag for x in mbf]
# plt.scatter(X,Y, color='red', s=1)
# plt.grid(True)
# plt.show()