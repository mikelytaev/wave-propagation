import mpmath
import numpy as np
import math as fm
import matplotlib.pyplot as plt
from propagators._utils import *
from rwp.environment import *


def pade_coefs(*, pade_order, k0, dx, alpha=0):
    mpmath.mp.dps = 63

    if alpha == 0:
        def sqrt_1plus(x):
            return mpmath.mp.sqrt(1 + x)
    else:
        a_n, b_n = pade_sqrt_coefs(pade_order[1])
        def sqrt_1plus(x):
            return pade_sqrt(x, a_n, b_n, alpha)

    def propagator_func(s):
        return mpmath.mp.exp(1j * k0 * dx * (sqrt_1plus(s) - 1))

    t = mpmath.taylor(propagator_func, 0, pade_order[0] + pade_order[1] + 20)
    p, q = mpmath.pade(t, pade_order[0], pade_order[1])
    num_coefs = np.array([complex(v) for v in p])
    den_coefs = np.array([complex(v) for v in q])
    return num_coefs[::-1], den_coefs[::-1]


def characteristic_poly_roots(num_coefs, den_coefs, xi):
    c_poly = num_coefs - xi * den_coefs
    return np.roots(c_poly)


wl = 0.03
k0 = 2*fm.pi / wl
dx = 900*wl

tau = 1.001
xi = tau * np.exp(1j*np.linspace(-fm.pi, fm.pi, 1000))
num_coefs, den_coefs = pade_coefs(pade_order=(8, 8), k0=k0, dx=dx, alpha=0)
roots = np.array([characteristic_poly_roots(num_coefs, den_coefs, a) for a in xi])#*k0**2

colors = ["red", "black", "yellow", "blue", "pink", "orange", "green", "navy"]

for r_i in range(0, len(roots[0, :])):
    X = [x.real for x in roots[:, r_i]]
    Y = [x.imag for x in roots[:, r_i]]
    plt.scatter(X, Y, color=colors[r_i], s=r_i+1)
plt.grid(True)
plt.show()

mbf = MillerBrownFactor(5)
theta = 0.0
material = SaltWater()
#r = reflection_coef(1, material.complex_permittivity(3e8 / wl), 90-theta, "V")
r = -1
#print(mbf.factor(1, 1, 1, mbf.roots(r)[0]/2))

poles = np.array([np.polyval(num_coefs, -root/2/k0**2) / np.polyval(den_coefs, -root/2/k0**2) for root in mbf.roots(r)])
print(max(abs(poles)))

X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
f = (np.sqrt(X + 1j*Y)).imag
plt.imshow(f)
plt.show()