"""Generate reference scenarios with the terrain-following rational propagator.

Each scenario is saved as ``data/scn_<seed>.npz`` with the field columns ``v``
(every ``DX_OUT`` m, ``N_WIN`` rows), the metric features (every ``DX_METRIC``
m), the terrain parameters and the reference solver's grid and wall time.

Run::

    python gen_data.py <first_seed> <last_seed_exclusive> [precision]
"""
import os
import sys
import time

import numpy as np

from common import DATA_DIR, DX_METRIC, X_MAX, metric_features, reference, sample_scenario, terrain_map

DATA = DATA_DIR


def main():
    s0, s1 = int(sys.argv[1]), int(sys.argv[2])
    precision = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
    os.makedirs(DATA, exist_ok=True)
    for seed in range(s0, s1):
        path = os.path.join(DATA, f"scn_{seed:04d}.npz")
        if os.path.exists(path):
            continue
        t = time.time()
        sc = sample_scenario(seed)
        v, tf, wall = reference(sc, precision=precision)
        xi_m = np.arange(0.0, X_MAX + DX_METRIC / 2, DX_METRIC)
        feats = metric_features(terrain_map(sc), xi_m)
        np.savez_compressed(path, v=v.astype(np.complex64), feats=feats.astype(np.float32), hills=sc.hills,
                            beams=sc.beams, wall=wall, dx=tf.dx_m, deta=tf.dz_m, order=np.array(tf.order),
                            precision=precision)
        print(f"seed {seed}: {time.time() - t:.1f} s (compute {wall:.1f} s, dxi {tf.dx_m:.2f}, "
              f"deta {tf.dz_m:.4f}, order {tf.order})", flush=True)


if __name__ == "__main__":
    main()
