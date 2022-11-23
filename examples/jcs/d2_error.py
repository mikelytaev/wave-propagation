import cmath as cm
import numpy as np
import matplotlib.pyplot as plt


def second_difference_disp_rel(k_z: complex, dz: float, z=0):
    return cm.exp(1j*k_z*z) * (cm.exp(-1j*k_z*dz) - 2 + cm.exp(1j*k_z*dz))


def fourth_difference_disp_rel(k_z: complex, dz: float, z=0):
    return cm.exp(1j*k_z*z) * (cm.exp(-1j*k_z*dz) - 2 + cm.exp(1j*k_z*dz))**2


def second_order_error(theta: float, dz: float):
    k0 = 2 * cm.pi
    k_z = k0 * cm.sin(theta * cm.pi / 180)
    d = 1 / dz**2 * (second_difference_disp_rel(k_z, dz))
    return abs(d - (-k_z**2))


def fourth_order_error_theta(theta: float, dz: float):
    k0 = 2 * cm.pi
    k_z = k0 * cm.sin(theta * cm.pi / 180)
    d = 1 / dz**2 * (second_difference_disp_rel(k_z, dz) - 1/12 * fourth_difference_disp_rel(k_z, dz))
    return abs(d - (-k_z**2))


def fourth_order_error_kz(k_z: float, dz: float):
    d = 1 / dz**2 * (second_difference_disp_rel(k_z, dz) - 1/12 * fourth_difference_disp_rel(k_z, dz))
    return abs(d - (-k_z**2))


thetas = np.linspace(0, 45, 1000)
dz = 0.25

errors_2nd = [second_order_error(theta, dz) for theta in thetas]
errors_4th = [fourth_order_error_theta(theta, dz) for theta in thetas]

plt.figure(figsize=(6, 3.2))
plt.plot(thetas, errors_2nd, label="2nd order")
plt.plot(thetas, errors_4th, label="4th order")
plt.xlim([thetas[0], thetas[-1]])
plt.ylim([1e-10, 1e1])
plt.xlabel("Angle (degrees)")
plt.ylabel("Abs. error")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()