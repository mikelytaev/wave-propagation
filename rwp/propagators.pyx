import numpy as np
cimport numpy as np

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'

def Crank_Nikolson_propagator(complex k0dz, complex a, complex b, double[:] het, complex[:] initial,
                              lower_bound=(1.0, 0.0, 0.0), upper_bound=(0.0, 1.0, 0.0)):
    cdef complex lb0 = lower_bound[0]
    cdef complex lb1 = lower_bound[1]
    cdef complex lbv = lower_bound[2]
    cdef complex ub0 = upper_bound[0]
    cdef complex ub1 = upper_bound[1]
    cdef complex ubv = upper_bound[2]
    cdef complex[:] rhs = np.zeros(len(initial), dtype=complex)
    rhs[0] = lbv
    rhs[-1] = ubv
    cdef Py_ssize_t i
    for i in range(1, len(rhs)-1):
        rhs[i] = (initial[i-1] + initial[i+1]) * a / k0dz ** 2 + initial[i] * (1 + a * (-2.0 / k0dz ** 2 + het[i]))

    cdef complex[:] v = np.zeros(len(rhs), dtype=complex)
    cdef complex[:] y = np.zeros(len(rhs), dtype=complex)
    cdef complex ci = b / k0dz ** 2
    cdef complex w = lb0
    y[0] = rhs[0] / w

    v[0] = lb1 / w
    w = 1 + b * (-2.0 / k0dz ** 2 + het[1]) - ci * v[0]
    y[1] = (rhs[1] - ci * y[0]) / w
    for i in range(2, len(y)-1):
        v[i-1] = ci / w
        w = 1 + b * (-2.0 / k0dz ** 2 + het[i]) - ci * v[i-1]
        y[i] = (rhs[i] - ci * y[i-1]) / w
    v[-2] = ci / w
    w = ub1 - ub0 * v[-2]
    y[-1] = (rhs[-1] - ub0 * y[-2]) / w

    for i in range(len(y)-2, -1, -1):
        y[i] = y[i] - v[i] * y[i+1]

    return y

