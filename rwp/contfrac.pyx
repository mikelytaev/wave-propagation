__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'

cpdef complex bessel_ratio(complex c, complex d, int j, double tol):
    tol /= 1e4
    cdef complex num = cont_frac_seq(c, d, j, 2) + 1.0 / cont_frac_seq(c, d, j, 1)
    cdef complex den = cont_frac_seq(c, d, j, 2)
    cdef complex y = cont_frac_seq(c, d, j, 1) * num / den
    cdef int i = 3
    while abs(num / den - 1) > tol:
        num = cont_frac_seq(c, d, j, i) + 1.0 / num
        den = cont_frac_seq(c, d, j, i) + 1.0 / den
        y = y * num / den
        i += 1

    return y

cdef complex cont_frac_seq(complex c, complex d, int j, int n):
    return (-1)**(n+1) * 2.0 * (j + (2.0 + d) / c + n - 1) * (c / 2.0)