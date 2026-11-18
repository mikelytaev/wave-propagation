"""Two-way (Bremmer) terrain-following propagator.

Discrete realization of the forward-backward splitting of
``papers/tap2026/two_way_helmholtz_tf.tex`` in terrain-following coordinates.
The Liouville-transformed equation ``phi_xixi + B phi = 0`` is factorized in
the flux-normalized variables ``u +/- = (Q phi -+ i Q^-1 phi_xi)/2`` with
``Q = B^(1/4)``, ``Lambda = B^(1/2)``; the exact coupled evolution is

    d/dxi (u+, u-) = ((i Lambda + T, R), (R, -i Lambda + T)) (u+, u-),

with the coupling ``R = (Q_xi Q^-1 + Q^-1 Q_xi)/2`` and the skew-adjoint pure
phase ``T = [Q_xi, Q^-1]/2``.  The discrete realization reuses the one-way
terrain-following solver's rational resolvent solves: a forward march of a
split component is the existing Padé chain (``G+``), a backward march runs the
same tridiagonal solves with the Padé coefficient pairs swapped (``G-``), and
the coupling over one step is applied through the discrete ``B^(+/-1/4)``
(flux-factor) chains,

    dxi R y = 1/2 [ (Q_{j+1} - Q_j) Q_mid^-1 y
                    + Q_mid^-1 (Q_{j+1} - Q_j) y ],

evaluated on the transition-layer rows (the flux factors are the identity in
the exterior, so the coupling vanishes there).  The two-sweep Bremmer
iteration starts from the incident forward wave, marches it forward, sweeps
the total forward field backward with ``u-(xi_end) = 0`` to collect the
backward wave, propagates it forward, and repeats ``n_bremmer`` times; the
physical field is ``psi = r^-1/2 Q^-1 (u+ + u-)``.

The commutator term ``T`` (order ``eps^2``, a pure phase) is not included.
The upper boundary is a Dirichlet row placed well above the field, which is
why the verification domain should be tall; the lower boundary is the terrain
(Dirichlet or impedance, as in the one-way solver).
"""
import jax
import jax.numpy as jnp
from jax import tree_util

from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator


class TwoWayTerrainFollowingPropagator(TerrainFollowingHelmholtzPropagator):
    """The one-way terrain-following solver extended with the backward sweep
    ``G-``, the coupling ``R`` of the forward-backward splitting, and the
    Bremmer two-sweep march.  See the module docstring."""

    @classmethod
    def create(cls, *args, **kwargs):
        prop = TerrainFollowingHelmholtzPropagator.create(*args, **kwargs)
        prop.__class__ = cls
        return prop

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        obj = TerrainFollowingHelmholtzPropagator._tree_unflatten.__func__(cls, static, dynamic)
        obj.__class__ = cls
        return obj

    # ------------------------------------------------------------- Q actions
    def _q_action(self, y, cols):
        """``B_cols^(1/4) y`` on the rows ``[0, n_flux)``; identity above.
        The rotated Padé chain is the identity at ``L = 0``, so the action is
        beta-normalized consistently across columns."""
        n = self.n_flux
        cf = tuple(c[:n] for c in cols[:3]) + (cols[3][:n - 1], cols[4])
        u = y[:n]
        for l in range(self.flux_coefs.shape[0]):
            a, b = self.flux_coefs[l]
            u = self._tf_substep(a, b, u, *cf, (0.0, 1.0, u[-1]))
        return y.at[:n].set(u)

    def _q_inv_action(self, y, cols):
        """``B_cols^(-1/4) y`` on the rows ``[0, n_flux)``; identity above.
        The inverse of the composite forward chain applies the swapped solves
        in reverse order."""
        n = self.n_flux
        cf = tuple(c[:n] for c in cols[:3]) + (cols[3][:n - 1], cols[4])
        u = y[:n]
        for l in reversed(range(self.flux_coefs.shape[0])):
            a, b = self.flux_coefs[l]
            u = self._tf_substep(b, a, u, *cf, (0.0, 1.0, u[-1]))
        return y.at[:n].set(u)

    def _coupling(self, y, cols_j, cols_j1, cols_mid):
        """``(d_xi R_{j+1/2}) y`` on the flux rows (zero elsewhere): the coupling
        accumulated by one step, ``1/2[(Q_{j+1} - Q_j) Q_mid^-1 y +
        Q_mid^-1 (Q_{j+1} - Q_j) y]``."""
        qmy = self._q_inv_action(y, cols_mid)
        a = self._q_action(qmy, cols_j1) - self._q_action(qmy, cols_j)
        dq = self._q_action(y, cols_j1) - self._q_action(y, cols_j)
        b = self._q_inv_action(dq, cols_mid)
        return jnp.zeros_like(y).at[:self.n_flux].set(0.5 * (a[:self.n_flux] + b[:self.n_flux]))

    # ------------------------------------------------------------ transport
    def _transport(self, y, cols_mid, k, forward=True, upper_bound=(0.0, 1.0, 0.0)):
        """One marching step of a split component with the mid-column metrics:
        the rational Padé chain of class ``k`` (the split state is gauge-free,
        so the flux brackets are the identity).  ``forward=False`` uses the
        approximant of ``exp(-i dxi sqrt(1+L))`` -- the coefficient pairs are
        negated, which conjugates the transfer function; the exact inverse of
        the forward chain must not be used, since it is unbounded on the
        evanescent spectrum."""
        het, m, rt, rt_half, h_eta0 = cols_mid
        beta_k = self.tables['beta_cls'][k]
        coefs_k = self.tables['coefs_cls'][k]
        het_k = (het + 1) * (self.beta / beta_k) ** 2 - 1
        p = len(self.coefs)
        if forward:
            def body(i, y):
                return self._tf_substep(coefs_k[i, 0], coefs_k[i, 1], y, het_k, m, rt, rt_half,
                                        h_eta0, upper_bound, beta=beta_k)
            y = jax.lax.fori_loop(0, p, body, y)
            return y * jnp.exp(1j * (beta_k - self.beta) * self.dx_m)

        def body(i, y):
            return self._tf_substep(-coefs_k[i, 0], -coefs_k[i, 1], y, het_k, m, rt, rt_half,
                                    h_eta0, upper_bound, beta=beta_k)
        y = jax.lax.fori_loop(0, p, body, y)
        return y * jnp.exp(-1j * (beta_k - self.beta) * self.dx_m)

    def _class_at(self, i):
        cls_p = self.tables['cls_p']
        return cls_p[jnp.clip(jnp.round((i + 0.5) * self.dx_m / self.tables['dxi_p']).astype(jnp.int64),
                              0, cls_p.shape[0] - 1)]

    # --------------------------------------------------------------- sweeps
    @jax.jit
    def _march_forward(self, y0, tm):
        """March the split state forward through all columns, storing every
        column's state.  ``tm`` is the total backward field whose coupling
        feeds the forward wave (zeros for the first pass).  The top row uses
        the discrete transparent (NLBC) condition -- the same kernels as the
        one-way march, applied to this sweep's state."""
        n = self.x_n
        use_nlbc = self.upper_bc == 'nlbc'
        # the full top-row history: row j holds the top-row field of step j
        upper_field = jnp.zeros((n, len(self.coefs)), dtype=jnp.complex128)

        def body(j, val):
            y, res, upper_field = val
            cols_j = self._columns_at(self.x_min_m + j * self.dx_m)
            cols_j1 = self._columns_at(self.x_min_m + (j + 1) * self.dx_m)
            cols_mid = self._columns_at(self.x_min_m + (j + 0.5) * self.dx_m)
            # the coupling of the total backward wave feeds the forward wave
            y = y + self._coupling(tm[j], cols_j, cols_j1, cols_mid)
            k = self._class_at(j)
            if use_nlbc:
                upper_convolution = self._convolution(self.upper_nlbc_coefs, upper_field, j + 1)
                upper_bound = (1, -self.upper_nlbc_coefs[0, j + 1, j + 1],
                               upper_convolution[j + 1]
                               + self.upper_nlbc_coefs[0, j + 1] @ upper_field[j + 1])
            else:
                upper_bound = (0.0, 1.0, 0.0)
            y = self._transport(y + 0j, cols_mid, k, forward=True, upper_bound=upper_bound)
            upper_field = upper_field.at[j + 1].set(y[-1])
            res = res.at[j + 1].set(y)
            return y, res, upper_field

        res = jnp.zeros((n, self.z_n), dtype=jnp.complex128).at[0].set(y0)
        y, res, upper_field = jax.lax.fori_loop(0, n - 1, body, (y0, res, upper_field))
        return res

    @jax.jit
    def _march_backward(self, tp):
        """Sweep the total forward field ``tp`` backward, accumulating the
        coupling into the backward wave; ``u-(xi_end) = 0``.  Returns all
        columns' backward states."""
        n = self.x_n

        def body(k, val):
            w, res = val
            i = n - 2 - k                    # step from column i+1 back to i
            cols_i = self._columns_at(self.x_min_m + i * self.dx_m)
            cols_i1 = self._columns_at(self.x_min_m + (i + 1) * self.dx_m)
            cols_mid = self._columns_at(self.x_min_m + (i + 0.5) * self.dx_m)
            # the coupling of the total forward wave feeds the backward wave
            w = w - self._coupling(tp[i + 1], cols_i, cols_i1, cols_mid)
            w = self._transport(w + 0j, cols_mid, self._class_at(i), forward=False)
            res = res.at[i].set(w)
            return w, res

        res = jnp.zeros((n, self.z_n), dtype=jnp.complex128).at[n - 1].set(0.0)
        w, res = jax.lax.fori_loop(0, n - 1, body, (tp[n - 1] * 0.0, res))
        return res

    @jax.jit
    def _psi_of(self, up, um):
        """Physical field of the split states: ``psi = r^-1/2 Q^-1 (up + um)``,
        per column, on all marching columns."""
        n = self.x_n

        def body(j, psi):
            cols = self._columns_at(self.x_min_m + j * self.dx_m)
            m = cols[1]
            phi = self._q_inv_action((up[j] + um[j]) + 0j, cols)
            return psi.at[j].set(phi / m ** 0.25)

        return jax.lax.fori_loop(0, n, body, jnp.zeros((n, self.z_n), dtype=jnp.complex128))

    # ------------------------------------------------------------- interface
    def compute(self, initial, n_bremmer=2):
        """Two-way march.  ``initial`` is the physical ``psi`` of the incident
        field on the starting trajectory ``xi = x_min_m``; ``n_bremmer`` is the
        number of reflection passes.  Returns the physical ``psi`` on the
        computational output grid (every ``x_grid_scale`` / ``z_grid_scale``
        columns/rows), plus the total forward and backward states and the
        Liouville field on the marching grid."""
        cols0 = self._columns_at(self.x_min_m)
        m0 = cols0[1]
        phi0 = (initial + 0j) * m0 ** 0.25                 # Liouville: phi = r^(1/2) psi
        up0 = self._q_action(phi0, cols0)                  # u+(0) = Q phi (a forward beam)
        tp = self._march_forward(up0, jnp.zeros((self.x_n, self.z_n), dtype=jnp.complex128))
        tm = jnp.zeros_like(tp)
        for _ in range(int(n_bremmer)):
            tm = self._march_backward(tp)                  # backward sweep of the total forward
            tp = self._march_forward(tp[0], tm)            # forward sweep of the total backward
        psi = self._psi_of(tp, tm)
        psi_out = psi[::self.x_grid_scale, ::self.z_grid_scale]
        return psi_out, tp, tm, psi


tree_util.register_pytree_node(
    TwoWayTerrainFollowingPropagator,
    TerrainFollowingHelmholtzPropagator._tree_flatten,
    lambda static, dynamic: TerrainFollowingHelmholtzPropagator._tree_unflatten.__func__(
        TwoWayTerrainFollowingPropagator, static, dynamic))
