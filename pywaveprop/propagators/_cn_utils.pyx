import numpy as np
cimport numpy as np

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'

def Crank_Nikolson_propagator(complex k0dz2, complex a, complex b, complex[:] het, complex[:] initial,
                              lower_bound=(1.0, 0.0, 0.0), upper_bound=(0.0, 1.0, 0.0)):
    cdef complex lb0 = lower_bound[0]
    cdef complex lb1 = lower_bound[1]
    cdef complex lbv = lower_bound[2]
    cdef complex ub0 = upper_bound[0]
    cdef complex ub1 = upper_bound[1]
    cdef complex ubv = upper_bound[2]
    cdef complex[:] rhs = np.empty(len(initial), dtype=complex)
    rhs[0] = lbv
    rhs[-1] = ubv
    cdef Py_ssize_t i
    for i in range(1, len(rhs)-1):
        rhs[i] = (initial[i-1] + initial[i+1]) * a / k0dz2 + initial[i] * (1 + a * (-2.0 / k0dz2 + het[i]))

    cdef complex[:] v = np.empty(len(rhs), dtype=complex)
    cdef complex[:] y = np.empty(len(rhs), dtype=complex)
    cdef complex ci = b / k0dz2
    cdef complex w = lb0
    y[0] = rhs[0] / w

    v[0] = lb1 / w
    w = 1 + b * (-2.0 / k0dz2 + het[1]) - ci * v[0]
    y[1] = (rhs[1] - ci * y[0]) / w
    for i in range(2, len(y)-1):
        v[i-1] = ci / w
        w = 1 + b * (-2.0 / k0dz2 + het[i]) - ci * v[i-1]
        y[i] = (rhs[i] - ci * y[i-1]) / w
    v[-2] = ci / w
    w = ub1 - ub0 * v[-2]
    y[-1] = (rhs[-1] - ub0 * y[-2]) / w

    for i in range(len(y)-2, -1, -1):
        y[i] = y[i] - v[i] * y[i+1]

    return y


def Crank_Nikolson_propagator2(complex k0dz, complex b, complex[:] het, complex[:] rhs,
                               lower_bound=(1.0, 0.0, 0.0), upper_bound=(0.0, 1.0, 0.0)):
    cdef complex lb0 = lower_bound[0]
    cdef complex lb1 = lower_bound[1]
    cdef complex lbv = lower_bound[2]
    cdef complex ub0 = upper_bound[0]
    cdef complex ub1 = upper_bound[1]
    cdef complex ubv = upper_bound[2]
    rhs[0] = lbv
    rhs[-1] = ubv

    cdef complex[:] v = np.zeros(len(rhs), dtype=complex)
    cdef complex[:] y = np.zeros(len(rhs), dtype=complex)
    cdef complex ci = b / k0dz ** 2
    cdef complex w = lb0
    y[0] = rhs[0] / w

    v[0] = lb1 / w
    w = 1 + b * (-2.0 / k0dz ** 2 + het[1]) - ci * v[0]
    y[1] = (rhs[1] - ci * y[0]) / w
    cdef Py_ssize_t i
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

def Crank_Nikolson_propagator2_4th_order(complex k0, complex dz, complex b, complex[:] het, complex[:] rhs,
                               lower_bound=(1.0, 0.0, 0.0), upper_bound=(0.0, 1.0, 0.0)):
    cdef complex lb0 = lower_bound[0]
    cdef complex lb1 = lower_bound[1]
    cdef complex lbv = lower_bound[2]
    cdef complex ub0 = upper_bound[0]
    cdef complex ub1 = upper_bound[1]
    cdef complex ubv = upper_bound[2]
    rhs[0] = lbv
    rhs[-1] = ubv

    cdef complex[:] b_i = np.zeros(len(rhs), dtype=complex)
    cdef complex[:] res = np.zeros(len(rhs), dtype=complex)
    cdef complex a_diag = k0**2 / 12 + b / dz**2
    cdef complex b_diag = k0**2 - 2*(k0**2 / 12 + b / dz**2)

    cdef complex w = a_diag / lb0
    b_i[1] = b_diag - w * lb1
    rhs[1] = rhs[1] - w * rhs[0]

    cdef Py_ssize_t i
    for i in range(2, len(rhs)-1):
        w = a_diag / b_i[i-1]
        b_i[i] = b_diag - w * a_diag
        rhs[i] = rhs[i] - w * rhs[i-1]

    w = ub0 / b_i[-2]
    b_i[-1] = ub1 - w * a_diag
    rhs[-1] = rhs[-1] - w * rhs[-2]

    res[-1] = rhs[-1] / b_i[-1]

    for i in range(len(rhs)-2, 0, -1):
        res[i] = (rhs[i] - a_diag * res[i+1]) / b_i[i]

    res[0] = (rhs[0] - lb1 * res[1]) / lb0

    return res

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void tridiag_method(np.ndarray[complex, ndim=1] lower,
                     np.ndarray[complex, ndim=1] diag,
                     np.ndarray[complex, ndim=1] upper,
                     np.ndarray[complex, ndim=1] rhs,
                     np.ndarray[complex, ndim=1] res):
    cdef Py_ssize_t i
    for i in range(1, len(diag)):
        w = lower[i-1] / diag[i-1]
        diag[i] = diag[i] - w * upper[i-1]
        rhs[i] = rhs[i] - w * rhs[i-1]

    res[len(res)-1] = rhs[len(rhs)-1] / diag[len(diag)-1]
    for i in range(len(diag)-2, -1, -1):
        res[i] = (rhs[i] - upper[i] * res[i+1]) / diag[i]


def tridiag_multiply(complex[:] a, complex[:] b, complex[:] c, complex[:] x):
    cdef complex[:] res = np.empty(len(b), dtype=complex)
    res[0] = b[0] * x[0] + c[0] * x[1]
    cdef Py_ssize_t i
    for i in range(1, len(b)-1):
        res[i] = a[i-1] * x[i-1] + b[i] * x[i] + c[i] * x[i+1]

    res[-1] = a[-1] * x[-2] + b[-1] * x[-1]

    return res

def Crank_Nikolson_propagator_4th_order(complex k0dz2, complex a, complex b, complex alpha, complex[:] het, complex[:] initial,
                              lower_bound=(1.0, 0.0, 0.0), upper_bound=(0.0, 1.0, 0.0)):
    cdef complex lb0 = lower_bound[0]
    cdef complex lb1 = lower_bound[1]
    cdef complex lbv = lower_bound[2]
    cdef complex ub0 = upper_bound[0]
    cdef complex ub1 = upper_bound[1]
    cdef complex ubv = upper_bound[2]
    cdef complex k0dz2alpha = alpha * k0dz2
    cdef complex[:] rhs = np.empty(len(initial), dtype=complex)
    rhs[0] = lbv
    rhs[-1] = ubv
    cdef Py_ssize_t i
    for i in range(1, len(rhs)-1):
        rhs[i] = (k0dz2alpha + a + a * k0dz2alpha * het[i-1]) * initial[i-1] + \
                 (k0dz2 * (1 - 2 * alpha) - 2 * a + a * k0dz2 * het[i] - 2 * a * k0dz2alpha * het[i]) * initial[i] + \
                 (k0dz2alpha + a + a * k0dz2alpha * het[i+1]) * initial[i+1]

    cdef complex[:] v_b = np.empty(len(initial), dtype=complex)
    v_b[0] = lb0

    cdef complex t = 0
    for i in range(1, len(v_b)):
        v_b[i] = (k0dz2 * (1 - 2 * alpha) - 2 * b + b * k0dz2 * het[i] - 2 * b * k0dz2alpha * het[i])
        if i == (len(v_b) - 1):
            t = ub0
            v_b[-1] = ub1
        else:
            t = alpha * k0dz2 + b + k0dz2alpha * b * het[i-1]
        w = t / v_b[i-1]
        if i == 0:
            t = lb1
        else:
            t = alpha * k0dz2 + b + k0dz2alpha * b * het[i]
        v_b[i] = v_b[i] - w * t
        rhs[i] = rhs[i] - w * rhs[i-1]

    cdef complex[:] res = np.empty(len(v_b), dtype=complex)
    res[-1] = rhs[-1] / v_b[-1]
    for i in range(len(v_b)-2, -1, -1):
        res[i] = (rhs[i] - (k0dz2alpha + b + k0dz2alpha * b * het[i+1]) * res[i+1]) / v_b[i]

    return res