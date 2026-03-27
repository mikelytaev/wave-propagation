import jax
from jax import numpy as jnp


@jax.jit
def f_m(a: complex, b: complex, c: complex, d: complex, m: int):
    return (a + b * m) / (c + d * m)

@jax.jit
def bessel_ratio_4th_order(a_m1: complex, a_1: complex, b: complex, c: complex, d: complex, m: int, tol: float):
    tol = jnp.maximum(tol / 1e6, 1e-14)
    h_n = -f_m(c, d, a_m1, b, m)
    D_n = 0.0
    C_n = h_n
    delta = 0
    i = 1

    def cond_fun(val):
        _, delta, _, _, _ = val
        return abs(delta - 1) > tol

    def body_fun(val):
        i, delta, h_n, C_n, D_n = val
        b_n = -f_m(c, d, a_m1, b, m + i)
        a_n = -f_m(a_1, b, a_m1, b, m + i - 1)
        D_n = b_n + a_n * D_n
        C_n = b_n + a_n / C_n
        D_n = 1 / D_n
        delta = D_n * C_n
        h_n = h_n * delta
        i = i + 1
        return i, delta, h_n, C_n, D_n

    _, _, h_n, _, _ = jax.lax.while_loop(cond_fun, body_fun, (i, delta, h_n, C_n, D_n))

    return h_n


@jax.jit
def sqr_eq(a, b, c):
    c1 = (-b + jnp.sqrt(b**2 - 4 * a * c + 0j)) / (2 * a)
    c2 = (-b - jnp.sqrt(b ** 2 - 4 * a * c + 0j)) / (2 * a)
    return jax.lax.select(abs(c1) > abs(c2), c2, c1)
