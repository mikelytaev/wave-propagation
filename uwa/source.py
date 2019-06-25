import cmath as cm
import numpy as np


class Source:

    def max_angle(self):
        pass


class GaussSource(Source):

    def __init__(self, *, freq_hz=None, depth, beam_width, eval_angle):
        self.freq_hz = freq_hz
        self.depth = depth
        self.beam_width = beam_width
        self.eval_angle = eval_angle

    def aperture(self, k0, z):
        w = cm.sqrt(2) / k0
        return np.exp(-(z-self.depth)**2 / w**2)

    def max_angle(self):
        return self.beam_width + abs(self.eval_angle)
