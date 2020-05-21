import numpy as np
import math as fm
import cmath as cm
import matplotlib.pyplot as plt


wl = 0.03
k0 = 2*fm.pi / wl
dx = 900*wl

eps = 0.001
tau = 1.0 + eps
circle = tau * np.exp(1j*np.linspace(-0.99*fm.pi, 0.99*fm.pi, 1000))
xi = np.concatenate([circle,
                     np.linspace(circle[-1], eps + 1j*circle[-1].imag, 3000),
                     np.linspace(eps + 1j*circle[-1].imag, eps + 1j*circle[0].imag, 3000),
                     np.linspace(eps + 1j*circle[0].imag, circle[0], 3000)
                     ])

k_x_d = (k0 + 1 / (1j*dx) * np.log(xi))# / k0

X = [x.real for x in k_x_d]
Y = [x.imag for x in k_x_d]
plt.scatter(X, Y, color="red", s=1)
plt.grid(True)
plt.show()

thetas = np.arccos(k_x_d)
X = [x.real for x in thetas]
Y = [x.imag for x in thetas]
plt.scatter(X, Y, color="red", s=1)
plt.grid(True)
plt.show()

k_z2 = k0**2 - k_x_d**2
X = [x.real for x in k_z2]
Y = [x.imag for x in k_z2]
plt.scatter(X, Y, color="red", s=1)
plt.grid(True)
plt.show()