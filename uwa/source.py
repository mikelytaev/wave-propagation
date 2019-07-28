import cmath as cm
import numpy as np


class Source:

    def __init__(self, freq_hz):
        self.freq_hz = freq_hz

    def max_angle(self):
        pass


class GaussSource(Source):

    def __init__(self, *, freq_hz, depth, beam_width, eval_angle):
        self.freq_hz = freq_hz
        self.depth = depth
        self.beam_width = beam_width
        self.eval_angle = eval_angle

    def aperture(self, k0, z):
        ww = cm.sqrt(2 * cm.log(2)) / (k0 * cm.sin(self.beam_width * cm.pi / 180 / 2))
        return 1 / (cm.sqrt(cm.pi) * ww) * np.exp(-1j * k0 * np.sin(self.eval_angle * cm.pi / 180) * z) * \
        np.exp(-((z - self.depth) / ww) ** 2)

    def max_angle(self):
        return self.beam_width + abs(self.eval_angle)
