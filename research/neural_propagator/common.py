"""Shared setup for the neural range-step study.

Scenario: 300 MHz (lambda = 1 m) over random smooth hills, perfectly conducting
ground (Dirichlet), homogeneous atmosphere.  The reference field comes from the
terrain-following rational propagator (``helmholtz_tf_jax``) on its optimized
grid with the discrete NLBC on a tall domain; the learned step works on a
lower window ``0 <= eta < WINDOW_M`` of the same computational coordinates.

All fields are envelopes relative to ``K0``: ``v = psi * exp(-i K0 xi)``.
"""
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D  # noqa: E402
from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium  # noqa: E402
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FREQ = 3e8
K0 = 2 * math.pi * FREQ / 3e8
X_MAX = 3000.0          # range of one scenario, m
DX_OUT = 25.0           # spacing of the stored columns, m
DETA = 0.25             # vertical step of the learned propagator, m (0.5 m aliases the stretched field)
N_WIN = 1024            # rows of the window
WINDOW_M = N_WIN * DETA
N_EVAL = 800            # rows below 200 m: errors are measured there; above is an absorbing layer
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data_d{int(round(DETA * 100)):03d}")
ABSORB_PER_25M = 0.3    # attenuation exponent at the window top per 25 m of range
TRANSITION = 150.0      # transition-layer height L, m
H_COMP = 420.0          # height of the reference computational domain, m
DX_METRIC = 5.0         # range spacing of the stored metric columns, m
THETA_MAX_DEG = 45.0    # aperture of the reference solver
MAX_SLOPE_DEG = 12.0


@dataclass
class Scenario:
    seed: int
    hills: np.ndarray        # (n, 3): centre, height, width
    beams: np.ndarray        # (n, 5): height, waist, elevation_deg, Re a, Im a

    def terrain(self, x):
        x = np.asarray(x, dtype=float)
        t = np.zeros_like(x)
        for c, h, w in self.hills:
            t += h * np.exp(-0.5 * ((x - c) / w) ** 2)
        return t

    def initial(self, z):
        """Starting field on the vertical line xi = 0 (height above ground)."""
        z = np.asarray(z, dtype=float)
        u = np.zeros(z.shape, dtype=complex)
        for zs, w, el, ar, ai in self.beams:
            kz = K0 * math.sin(math.radians(el))
            g = lambda zz: np.exp(-((zz - zs) / w) ** 2 + 1j * kz * (zz - zs))  # noqa: E731
            u += (ar + 1j * ai) * (g(z) - g(-z))      # odd image: Dirichlet ground
        return u


def sample_scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    while True:
        n = rng.integers(1, 5)
        hills = np.stack([rng.uniform(200.0, X_MAX + 300.0, n), rng.uniform(-40.0, 70.0, n),
                          rng.uniform(100.0, 350.0, n)], axis=1)
        x = np.linspace(0, X_MAX + 500, 8001)
        sc = Scenario(seed, hills, np.zeros((0, 5)))
        slope = np.degrees(np.arctan(np.abs(np.gradient(sc.terrain(x), x)).max()))
        if slope <= MAX_SLOPE_DEG:
            break
    nb = rng.integers(1, 4)
    beams = np.stack([rng.uniform(20.0, 170.0, nb), rng.uniform(3.0, 10.0, nb), rng.uniform(-8.0, 10.0, nb),
                      rng.normal(size=nb), rng.normal(size=nb)], axis=1)
    return Scenario(seed, hills, beams)


def terrain_map(sc: Scenario) -> OrthogonalTerrainMap:
    xb = np.linspace(0.0, X_MAX + 1500.0, 9001)
    return OrthogonalTerrainMap(xb, sc.terrain(xb), x_max_m=X_MAX, transition_m=TRANSITION, smoothing_m=0.0)


def metric_features(tm: OrthogonalTerrainMap, xi: np.ndarray):
    """(len(xi), N_WIN, 2): log h_xi and log h_eta on the window nodes."""
    eta = np.arange(N_WIN) * DETA
    n_lab = min(int(math.ceil(tm.eta_lab / DETA)) + 2, N_WIN)
    mt = tm.metrics(xi, eta[:n_lab])
    f = np.zeros((xi.size, N_WIN, 2))
    f[:, :n_lab, 0] = np.log(mt.h_xi)
    f[:, :n_lab, 1] = np.log(mt.h_eta)
    return f


def reference(sc: Scenario, precision=1e-3, order=None, x_max=X_MAX, verbose=False):
    """Terrain-following rational solution on the stored grid.

    Returns ``v`` (n_cols, N_WIN) complex, the solver and its wall time."""
    tm = terrain_map(sc)
    med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), K0 ** 2, dtype=complex))
    mp = HelmholtzMeshParams2D(x_size_m=x_max, z_size_m=H_COMP, dx_output_m=DX_OUT, dz_output_m=DETA)
    kz_max = K0 * math.sin(math.radians(THETA_MAX_DEG))
    out = sys.stdout
    if not verbose:
        sys.stdout = open(os.devnull, "w")
    try:
        tf = TerrainFollowingHelmholtzPropagator.create(FREQ, tm, med, kz_max, (K0, K0), precision, mp,
                                                        rational_approx_order=order)
    finally:
        if not verbose:
            sys.stdout.close()
            sys.stdout = out
    eta = tf.eta_computational_grid()
    init = jnp.asarray(sc.initial(eta))
    t = time.time()
    u = np.asarray(tf.compute(init).block_until_ready())
    wall = time.time() - t
    xi = np.asarray(tf.x_output_grid())
    v = u[:, :N_WIN] * np.exp(1j * (tf.beta - K0) * xi)[:, None]
    return v, tf, wall
