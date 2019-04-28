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

cpdef complex bessel_ratio_4th_order(complex a_m1, complex a_1, complex b, complex c, complex d, int m, double tol):
    tol /= 1e6
    cdef complex h_n = -f_m(c, d, a_m1, b, m)
    cdef complex D_n = 0.0
    cdef complex C_n = h_n
    cdef complex delta = 0
    cdef int i = 1
    cdef complex b_n = 0
    cdef complex a_n = 0
    while abs(delta - 1) > tol:
        b_n = -f_m(c, d, a_m1, b, m+i)
        a_n = -f_m(a_1, b, a_m1, b, m+i-1)
        D_n = b_n + a_n * D_n
        C_n = b_n + a_n / C_n
        D_n = 1 / D_n
        delta = D_n * C_n
        h_n = h_n * delta
        i = i + 1

    return h_n

cdef complex f_m(complex a, complex b, complex c, complex d, int m):
    return (a + b * m) / (c + d * m)