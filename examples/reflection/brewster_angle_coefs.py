from propagators._utils import *
from rwp.environment import *
import matplotlib.pyplot as plt


ground_material = CustomMaterial(eps=3, sigma=0)
thetas = np.linspace(0, 90, 500)
freq_hz = 3e9

abs_relf_hor = [abs(reflection_coef(1, ground_material.complex_permittivity(freq_hz), 90-theta, "H")) for theta in thetas]
abs_relf_ver = [abs(reflection_coef(1, ground_material.complex_permittivity(freq_hz), 90-theta, "V")) for theta in thetas]
plt.figure(figsize=(6, 3.2))
plt.plot(thetas, abs_relf_hor, label="H-pol.")
plt.plot(thetas, abs_relf_ver, label="V-pol.")
plt.grid(True)
plt.xlabel("Angle of incidence (Deg)")
plt.xlim([thetas[0], thetas[-1]])
plt.ylim([0, 1])
plt.ylabel("|R|")
plt.legend()
plt.tight_layout()
plt.show()