import numpy as np
from itertools import zip_longest


def cheb_pade_coefs(k0, dx_m, pade_order, spc_val, typee):
    """

    :param dx_wl:
    :param pade_order:
    :param spc_val:
    :param typee: "chebpade" , "ratinterp"
    :return:
    """
    import matlab.engine
    if type(spc_val) is tuple:
        eng = matlab.engine.start_matlab()
        a_mat, b_mat, a0 = eng.ExpChebPadeCoefs2(int(pade_order[0]), int(pade_order[1]), float(k0), float(dx_m),
                                                float(spc_val[0]), float(spc_val[1]), typee, nargout=3)
    else:
        eng = matlab.engine.start_matlab()
        a_mat, b_mat, a0 = eng.ExpChebPadeCoefs2(int(pade_order[0]), int(pade_order[1]), float(dx_wl), float(spc_val), typee, nargout=3)
    eng.quit()
    a = np.array(a_mat).reshape(max(a_mat.size))
    b = np.array(b_mat).reshape(max(b_mat.size))
    return list(zip_longest(a, b, fillvalue=0.0j)), a0
