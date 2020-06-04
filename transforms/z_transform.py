from transforms.fcc_fourier.fcc import *


def inv_z_transform(func, n, tau=1.0, tol=1e-12):
    fcca = FCCAdaptiveFourier(2 * fm.pi, -np.arange(0, n), rtol=tol)
    m_size = 1
    coefs = (tau ** np.repeat(np.arange(0, n)[:, np.newaxis], m_size ** 2, axis=1) / (2 * fm.pi) *
             fcca.forward(lambda t: func(tau * cm.exp(1j*t)), 0, 2 * cm.pi)).reshape((n, m_size, m_size))
    return np.squeeze(coefs)