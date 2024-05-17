from typing import List


def is_list(x_list, A_tuple) -> bool:
    if not isinstance(x_list, list):
        return False
    for x in x_list:
        if not isinstance(x, A_tuple):
            return False
    return True


def is_list_equal(l: List[complex], tol: float = 1e-10):
    med = sum(l) / len(l)
    ll = [abs(v - med) for v in l]
    return max(ll) < tol
