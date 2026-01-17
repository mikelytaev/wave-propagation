from dataclasses import dataclass


@dataclass
class HelmholtzMeshParams2D:
    x_size_m: float
    z_size_m: float
    dx_output_m: float = None
    x_n_upper_bound: int = None
    dz_output_m: float = None
    z_n_upper_bound: int = None

    def __post_init__(self):
        if self.x_n_upper_bound is None and self.dx_output_m is None:
            raise ValueError("one of x_n_upper_bound or dx_output_m should be specified")
        if self.x_n_upper_bound is not None and self.dx_output_m is not None:
            raise ValueError("Only one of x_n_upper_bound or dx_output_m should be specified, not both")
        if self.z_n_upper_bound is None and self.dz_output_m is None:
            raise ValueError("one of z_n_upper_bound or dz_output_m should be specified")
        if self.z_n_upper_bound is not None and self.dz_output_m is not None:
            raise ValueError("Only one of z_n_upper_bound or dz_output_m should be specified, not both")
