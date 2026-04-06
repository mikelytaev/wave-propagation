"""
Compare JAX propagator against RAM (Range-dependent Acoustic Model).

RAM is the reference split-step Padé solver by M. D. Collins (NRL).
We generate RAM input files, run the Fortran executable, parse results,
and compare transmission loss with the JAX uwa_forward_task.
"""
import os
import struct
import subprocess
import tempfile
import unittest

import numpy as np

RAM_EXE = "/tmp/RAM_src/RAM/ram.exe"


def write_ram_input(filename, freq_hz, src_depth, rcv_depth,
                    max_range_m, dr, max_depth_m, dz,
                    c0, ssp_z, ssp_c,
                    bottom_c, bottom_rho, bottom_attn,
                    bottom_depth=None, np_pade=6):
    """Write a RAM input file for a range-independent environment."""
    bottom_depth = bottom_depth or max(ssp_z)
    ndr = 1
    ndz = 1
    with open(filename, 'w') as f:
        f.write("JAX comparison test\n")
        f.write(f"{freq_hz} {src_depth} {rcv_depth}    freq zs zr\n")
        f.write(f"{max_range_m} {dr} {ndr}    rmax dr ndr\n")
        f.write(f"{max_depth_m} {dz} {ndz} {max_depth_m}    zmax dz ndz zmplt\n")
        f.write(f"{c0} {np_pade} 1 0.0    c0 np ns rs\n")
        # Bathymetry (flat)
        f.write(f"0.0 {bottom_depth}\n")
        f.write(f"{max_range_m} {bottom_depth}\n")
        f.write("-1 -1\n")
        # Sound speed in water
        for z, c in zip(ssp_z, ssp_c):
            f.write(f"{z} {c}\n")
        f.write("-1 -1\n")
        # Sediment sound speed
        f.write(f"0.0 {bottom_c}\n")
        f.write("-1 -1\n")
        # Sediment density
        f.write(f"0.0 {bottom_rho}\n")
        f.write("-1 -1\n")
        # Attenuation
        f.write(f"0.0 {bottom_attn}\n")
        f.write("-1 -1\n")


def parse_tl_line(filename):
    """Parse RAM tl.line output (range, TL pairs)."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # ranges, TL


def parse_tl_grid(filename, nz):
    """Parse RAM tl.grid binary output."""
    with open(filename, 'rb') as f:
        data = f.read()
    # tl.grid is unformatted Fortran sequential: each record has 4-byte length prefix/suffix
    offset = 0
    ranges = []
    tl_grid = []
    while offset < len(data):
        rec_len = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4
        record = data[offset:offset+rec_len]
        offset += rec_len
        offset += 4  # trailing length
        n_floats = rec_len // 4
        values = struct.unpack(f'{n_floats}f', record)
        if n_floats == 1:
            ranges.append(values[0])
        elif n_floats == nz:
            tl_grid.append(list(values))
    return np.array(ranges), np.array(tl_grid)


def run_ram(input_file, work_dir):
    """Run RAM executable and return tl.line data."""
    result = subprocess.run(
        [RAM_EXE],
        stdin=open(input_file),
        cwd=work_dir,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"RAM failed: {result.stderr.decode()}")
    return parse_tl_line(os.path.join(work_dir, "tl.line"))


@unittest.skipUnless(os.path.isfile(RAM_EXE), f"RAM executable not found at {RAM_EXE}")
class TestJAXvsRAM(unittest.TestCase):

    def test_shallow_water_isovelocity(self):
        """Isovelocity shallow water: compare JAX vs RAM via the old code.

        RAM and old code both use a point source; JAX uses a Gaussian beam.
        We first verify old code ≈ RAM, then compare JAX vs old code using
        normalised TL (to factor out the source model difference).
        """
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import jax
        jax.config.update('jax_enable_x64', True)

        freq_hz = 100.0
        src_depth = 50.0
        rcv_depth = 50.0
        max_range_m = 5000.0
        bottom_depth = 300.0
        c_water = 1500.0
        c_bottom = 1700.0
        rho_bottom = 1.5
        attn_bottom = 0.5
        dr = 5.0
        dz = 1.0
        max_depth_m = bottom_depth + 500

        # --- Run RAM ---
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "ram.in")
            write_ram_input(
                inp, freq_hz, src_depth, rcv_depth,
                max_range_m, dr, max_depth_m, dz,
                c0=c_water, ssp_z=[0, bottom_depth], ssp_c=[c_water, c_water],
                bottom_c=c_bottom, bottom_rho=rho_bottom, bottom_attn=attn_bottom,
                bottom_depth=bottom_depth,
            )
            ram_ranges, ram_tl = run_ram(inp, tmpdir)

        # --- Run old code ---
        from pywaveprop.uwa.sspade import uwa_ss_pade, UWASSpadeComputationalParams
        from pywaveprop.uwa.source import GaussSource
        from pywaveprop.uwa.environment import UnderwaterEnvironment, Bathymetry

        old_src = GaussSource(freq_hz=freq_hz, depth_m=src_depth,
                              beam_width_deg=30, elevation_angle_deg=0)
        old_env = UnderwaterEnvironment(
            sound_speed_profile_m_s=lambda x, z: c_water,
            bottom_profile=Bathymetry(ranges_m=[0], depths_m=[bottom_depth]),
            bottom_sound_speed_m_s=c_bottom,
            bottom_density_g_cm=rho_bottom,
            bottom_attenuation_dm_lambda=attn_bottom,
        )
        old_params = UWASSpadeComputationalParams(
            max_range_m=max_range_m, max_depth_m=max_depth_m,
            dx_m=dr, dz_m=dz,
        )
        old_field = uwa_ss_pade(old_src, old_env, old_params)

        old_z_idx = np.argmin(np.abs(old_field.z_grid - rcv_depth))
        old_tl = -20 * np.log10(np.abs(old_field.field[:, old_z_idx]) + 1e-30)
        old_ranges = old_field.x_grid

        # --- Run JAX ---
        import jax.numpy as jnp
        from pywaveprop.uwa_jax import (
            UWAGaussSourceModel, UnderwaterLayerModel,
            UnderwaterEnvironmentModel, uwa_forward_task,
        )
        from pywaveprop.uwa_utils import UWAComputationalParams
        from pywaveprop.helmholtz_jax import ConstWaveSpeedModel

        jax_src = UWAGaussSourceModel(
            freq_hz=freq_hz, depth_m=src_depth,
            beam_width_deg=30, elevation_angle_deg=0, multiplier=1,
        )
        jax_env = UnderwaterEnvironmentModel(layers=[
            UnderwaterLayerModel(height_m=bottom_depth,
                                 sound_speed_profile_m_s=ConstWaveSpeedModel(c0=c_water), density=1.0),
            UnderwaterLayerModel(height_m=800,
                                 sound_speed_profile_m_s=ConstWaveSpeedModel(c0=c_bottom),
                                 density=rho_bottom, attenuation_dm_lambda=attn_bottom),
        ])
        jax_params = UWAComputationalParams(
            max_range_m=max_range_m, max_depth_m=max_depth_m,
            dx_m=dr, dz_m=dz,
        )
        jax_params.rational_approx_order = (7, 8)
        jax_result = uwa_forward_task(jax_src, jax_env, jax_params)

        jax_z_idx = np.argmin(np.abs(np.asarray(jax_result.z_grid) - rcv_depth))
        jax_tl = -20 * np.log10(np.abs(np.asarray(jax_result.field)[:, jax_z_idx]) + 1e-30)
        jax_ranges = np.asarray(jax_result.x_grid)

        # --- Compare JAX vs old code (same source model) ---
        jax_on_old = np.interp(old_ranges, jax_ranges, jax_tl)
        ref_idx = np.argmin(np.abs(old_ranges - 1000))
        old_norm = old_tl - old_tl[ref_idx]
        jax_norm = jax_on_old - jax_on_old[ref_idx]

        start = ref_idx + 5
        end = len(old_ranges) - 5
        diff_jax_old = np.abs(old_norm[start:end] - jax_norm[start:end])
        med_jax_old = float(np.median(diff_jax_old))

        # --- Compare old code vs RAM (reference) ---
        old_on_ram = np.interp(ram_ranges, old_ranges, old_tl)
        ram_ref = np.argmin(np.abs(ram_ranges - 1000))
        ram_norm = ram_tl - ram_tl[ram_ref]
        old_on_ram_norm = old_on_ram - old_on_ram[ram_ref]
        diff_old_ram = np.abs(ram_norm[ram_ref+5:-5] - old_on_ram_norm[ram_ref+5:-5])
        med_old_ram = float(np.median(diff_old_ram))

        # --- Compare JAX vs RAM directly ---
        jax_on_ram = np.interp(ram_ranges, jax_ranges, jax_tl)
        jax_on_ram_norm = jax_on_ram - jax_on_ram[ram_ref]
        diff_jax_ram = np.abs(ram_norm[ram_ref+5:-5] - jax_on_ram_norm[ram_ref+5:-5])
        med_jax_ram = float(np.median(diff_jax_ram))

        print(f"\nRAM comparison (shallow water, 100 Hz, isovelocity):")
        print(f"  Old code vs RAM: median {med_old_ram:.1f} dB")
        print(f"  JAX vs old code: median {med_jax_old:.1f} dB")
        print(f"  JAX vs RAM:      median {med_jax_ram:.1f} dB")

        # JAX and old code use the same source model but different grids
        # and Padé orders, so moderate differences are expected
        self.assertLess(med_jax_old, 6.0,
                        f"JAX vs old code median diff {med_jax_old:.1f} dB too large")
        # Old code vs RAM should be reasonable (different source models)
        self.assertLess(med_old_ram, 6.0,
                        f"Old code vs RAM median diff {med_old_ram:.1f} dB too large")
        # JAX (Gaussian beam) vs RAM (point source) — larger tolerance
        # because the source model difference is not normalised away
        self.assertLess(med_jax_ram, 8.0,
                        f"JAX vs RAM median diff {med_jax_ram:.1f} dB too large")


if __name__ == '__main__':
    unittest.main()
