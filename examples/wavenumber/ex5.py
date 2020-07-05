from propagators._utils import *
import matplotlib.pyplot as plt


wl = 0.03
k0 = 2*cm.pi / wl

thetas = np.linspace(0, 5, 1000)
mbf = MillerBrownFactor(8)

def test_func(theta, k0, rms_m):
    gamma = 2 * k0 * rms_m * cm.sin(theta * cm.pi / 180)
    arg = 0.5 * gamma ** 2
    return (cm.atan(-cm.log10(arg+1e-10)*3) / (cm.pi/2) + 1) / 2

#mbf_app = np.array([test_func(a, k0, 1) for a in thetas])
mbf_app = np.array([mbf.factor(a, k0, 1) for a in thetas])
mbf = np.array([Miller_Brown_factor(a, k0, 1) for a in thetas])
plt.plot(thetas, 10*np.log10(abs(mbf)), label="mbf")
plt.plot(thetas, 10*np.log10(abs(mbf_app)), label="mbf approx.")
plt.legend()
plt.show()

gamma = 2 * k0 * 1 * cm.sin(2 * cm.pi / 180)
arg = -0.5 * gamma ** 2

