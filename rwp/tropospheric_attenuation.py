from rwp.environment import Polarization
import math as fm


def log10k(freq_hz, polarz: Polarization):
    freq_ghz = freq_hz * 1e-9
    if polarz == Polarization.HORIZONTAL:
        a = [-5.33980, -0.35351, -0.23789, -0.94158]
        b = [-0.10008, 1.26970, 0.86036, 0.64552]
        c = [1.13098, 0.45400, 0.15354, 0.16817]
        m_k = -0.18961
        c_k = 0.71147
    else:
        a = [-3.80595, -3.44965, -0.39902, 0.50167]
        b = [0.56934, -0.22911, 0.73042, 1.07319]
        c = [0.81061, 0.51059, 0.11899, 0.27195]
        m_k = -0.16398
        c_k = 0.63297

    return sum([a[i] * fm.exp(-((fm.log10(freq_ghz) - b[i]) / c[i]) ** 2) for i in [0, 1, 2, 3]]) +\
             m_k * fm.log10(freq_ghz) + c_k


def alpha(freq_hz, polarz: Polarization):
    freq_ghz = freq_hz * 1e-9
    if polarz == Polarization.HORIZONTAL:
        a = [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721]
        b = [1.82442, 0.77564, 0.63773, -0.96230, -3.29980]
        c = [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990]
        m_a = 0.67849
        c_a = -1.95537
    else:
        a = [-0.07771, 0.56727, -0.20238, -48.2991, 48.5833]
        b = [2.33840, 0.95545, 1.14520, 0.791669, 0.791459]
        c = [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479]
        m_a = -0.053739
        c_a = 0.83433

    return sum([a[i] * fm.exp(-((fm.log10(freq_ghz) - b[i]) / c[i]) ** 2) for i in [0, 1, 2, 3, 4]]) +\
             m_a * fm.log10(freq_ghz) + c_a


def gamma(r, freq_hz, polarz: Polarization):
    return 10 ** log10k(freq_hz, polarz) * r ** alpha(freq_hz, polarz)
