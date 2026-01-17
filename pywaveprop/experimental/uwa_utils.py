from dataclasses import dataclass


@dataclass
class UWAComputationalParams:
    max_range_m: float
    max_depth_m: float = None
    rational_approx_order = None
    dx_m: float = None
    dz_m: float = None
    x_output_points: int = None
    z_output_points: int = None
    precision: float = 0.01

    def __post_init__(self):
        if self.x_output_points is None and self.dx_m is None:
            raise ValueError("x output grid (x_output_points or dx_m) is not specified!")
        if self.x_output_points is not None and self.dx_m is not None:
            raise ValueError("only one x output grid parameter (x_output_points or dx_m) should be specified!")

        if self.z_output_points is None and self.dz_m is None:
            raise ValueError("z output grid (z_output_points or dz_m) is not specified!")
        if self.z_output_points is not None and self.dz_m is not None:
            raise ValueError("only one z output grid parameter (z_output_points or dz_m) should be specified!")
