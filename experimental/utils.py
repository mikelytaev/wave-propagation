import jax


@jax.jit
def f_m(a: complex, b: complex, c: complex, d: complex, m: int):
    return (a + b * m) / (c + d * m)

@jax.jit
def bessel_ratio_4th_order(a_m1: complex, a_1: complex, b: complex, c: complex, d: complex, m: int, tol: float):
    tol /= 1e6
    h_n = -f_m(c, d, a_m1, b, m)
    D_n = 0.0
    C_n = h_n
    delta = 0
    i = 1
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
