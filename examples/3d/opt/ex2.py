import math as fm


theta_max_degrees = 80

c0 = fm.sqrt(2/(1+fm.cos(fm.radians(theta_max_degrees))**2))*3e8
c = 3e8


def xi(c0):
    return -(c0/c)**2 * fm.sin(fm.radians(theta_max_degrees))**2 + (c0/c)**2 - 1, (c0/c)**2 - 1


print(xi(c0))
print(xi(c))