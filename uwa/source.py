import cmath as cm
import numpy as np


class Source:

    def __init__(self, freq_hz, depth_m):
        self.freq_hz = freq_hz
        self.depth_m = depth_m

    def aperture(self, k0, z):
        pass

    def max_angle_deg(self):
        pass


class GaussSource(Source):

    def __init__(self, *, freq_hz, depth_m, beam_width_deg, eval_angle_deg):
        super().__init__(freq_hz, depth_m)
        self.beam_width_deg = beam_width_deg
        self.eval_angle_deg = eval_angle_deg

    def aperture(self, k0, z, n2=1):
        eval_angle_m = cm.asin(cm.sin(self.eval_angle_deg * cm.pi / 180) / (1 / n2))
        ww = cm.sqrt(2 * cm.log(2)) / (k0 * cm.sin(self.beam_width_deg * cm.pi / 180 / 2))
        return 1 / (cm.sqrt(cm.pi) * ww) * np.exp(-1j * k0 * np.sin(eval_angle_m) * z) * \
        np.exp(-((z - self.depth_m) / ww) ** 2)

    def max_angle_deg(self):
        return self.beam_width_deg + abs(self.eval_angle_deg)
