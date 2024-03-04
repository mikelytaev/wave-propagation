from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Measure:
    x_m: float
    height_m: float
    freq_hz: float
    value: Optional[complex] = None


@dataclass
class SearchArea:
    min_x_m: float
    max_x_m: float
    min_z_m: float = None
    max_x_m: float = None


def bartlett_mfp(measures: List[Measure], fields: List[np.ndarray]) -> np.ndarray:
    res = deepcopy(fields[0])
    res *= 0
    normalizer = deepcopy(fields[0])
    normalizer *= 0
    for ind in range(0, len(fields)):
        res += measures[ind].value.conjugate() * fields[ind]
        normalizer += abs(fields[ind]) ** 2

    res = res**2 / normalizer
    return res


def mv_mfp(measures: List[Measure], fields: List[np.ndarray]) -> np.ndarray:
    res = deepcopy(fields[0])
    res *= 0
    d = np.array([m.value for m in measures], dtype=complex)
    k = np.matmul(d.reshape(len(d), 1), d.reshape(1, len(d)).conj())
    k_inv = np.linalg.inv(k)

    normalizer = deepcopy(fields[0])
    normalizer *= 0
    for ind in range(0, len(fields)):
        normalizer += abs(fields[ind]) ** 2

    normalizer = np.sqrt(normalizer)

    for ind_i in range(0, res.shape[0]):
        for ind_j in range(0, res.shape[1]):
            w = np.array([f[ind_i, ind_j] for f in fields]) / normalizer[ind_i, ind_j]
            w = w.reshape(len(w), 1)
            t = np.matmul(np.matmul(w.reshape(1, len(w)).conj(), k_inv), w)
            res[ind_i, ind_j] = (1 / t) / (1 / np.matmul(w.reshape(1, len(w)).conj(), w))

    return res
