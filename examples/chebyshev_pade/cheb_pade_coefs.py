import matlab.engine
import numpy as np
from itertools import zip_longest


def cheb_pade_coefs(dx_wl, pade_order, max_spc_val, type):
    """

    :param dx_wl:
    :param pade_order:
    :param max_spc_val:
    :param type: "chebpade" , "ratinterp"
    :return:
    """
    eng = matlab.engine.start_matlab()
    a_mat, b_mat, a0 = eng.ExpChebPadeCoefs(int(pade_order[0]), int(pade_order[1]), float(dx_wl), float(max_spc_val), type, nargout=3)
    eng.quit()
    a = np.array(a_mat).reshape(max(a_mat.size))
    b = np.array(b_mat).reshape(max(b_mat.size))
    return list(zip_longest(a, b, fillvalue=0.0j)), a0
