"""
Tests for the JAX-based split-step Fourier (SSF) propagator.

The SSF solver is validated on tropospheric radio wave propagation problems
against

* the analytical two-ray model (flat ground, homogeneous atmosphere), and
* the JAX rational finite-difference (Padé) propagator, which is the reference
  implementation of the package, on ducting, terrain, finite-permittivity
  ground and vegetation problems.

Every test writes a self-contained HTML report (figures, setup, checked
metrics, explanation) and an ``.npz`` file with the raw results into the
directory given by the ``TEST_REPORTS`` environment variable (default
``test_reports/`` at the repository root; ``TEST_REPORTS=0`` disables it).
Open ``test_reports/index.html`` for the overview.
"""
import cmath
import math as fm
import unittest
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_jax import ConstWaveSpeedModel, PiecewiseLinearTerrainModel
from pywaveprop.helmholtz_ssf_jax import SSFHelmholtzPropagator, impedance_alpha
from pywaveprop.rwp_field import RWPField
from pywaveprop.rwp_jax import (
    EvaporationDuctModel,
    GroundMaterial,
    PiecewiseLinearNProfileModel,
    RWPComputationalParams,
    RWPGaussSourceModel,
    TroposphereModel,
    VegetationLayerModel,
    create_rwp_model,
    create_rwp_ssf_model,
    rwp_forward_task,
)
from tests.pywaveprop_jax.reporting import TestReport


def _run(src, env, params, method):
    p = deepcopy(params)
    p.approx_method = method
    if method != 'ssf':
        p.rational_approx_order = (7, 8)
    return rwp_forward_task(src, env, p)


def _db_stats(ref: RWPField, other: RWPField, z_m, x_skip=0.2, floor_db=-40.0, normalize=False):
    """Absolute dB difference statistics of ``|field|`` along a horizontal cut.

    Points more than ``floor_db`` below the maximum of the reference cut
    (deep interference nulls) are excluded; with ``normalize`` both cuts are
    scaled to their own maximum first.
    """
    iz = int(np.argmin(np.abs(ref.z_grid - z_m)))
    i0 = int(len(ref.x_grid) * x_skip)
    a = np.abs(ref.field[i0:, iz])
    b = np.abs(other.field[i0:, iz])
    if normalize:
        a = a / a.max()
        b = b / b.max()
    a_db = 20 * np.log10(a + 1e-30)
    b_db = 20 * np.log10(b + 1e-30)
    keep = a_db > a_db.max() + floor_db
    d = np.abs(a_db - b_db)[keep]
    return float(np.median(d)), float(np.percentile(d, 90)), float(d.max())


def _two_ray_field(src: RWPGaussSourceModel, x_grid, z_grid, eps=None, sigma=0.0):
    """Two-ray reference scaled to the PE reduced field (``sqrt(x)`` spreading removed)."""
    from pywaveprop.rwp.antennas import GaussAntenna
    from pywaveprop.rwp.environment import CustomMaterial, PerfectlyElectricConducting, Terrain, Troposphere
    from pywaveprop.rwp.tworay import TwoRayModel
    antenna = GaussAntenna(freq_hz=src.freq_hz, height=src.height_m, beam_width=src.beam_width_deg,
                           elevation_angle=src.elevation_angle_deg, polarz=src.polarization)
    env = Troposphere(flat=True)
    env.terrain = Terrain(ground_material=PerfectlyElectricConducting() if eps is None
                          else CustomMaterial(eps=eps, sigma=sigma))
    env.z_max = float(np.max(z_grid))
    x_grid = np.asarray(x_grid)
    z_grid = np.asarray(z_grid)
    val = TwoRayModel(src=antenna, env=env).calculate(x_grid[1:], z_grid)
    val = val * np.sqrt(x_grid[1:])[:, None]
    val = np.concatenate([np.zeros((1, val.shape[1]), dtype=complex), val])
    return RWPField(x_grid=x_grid, z_grid=z_grid, freq_hz=src.freq_hz, field=val)


def _report_setup(rep, src=None, env=None, params=None, model=None):
    """Standard setup tables for an RWP problem."""
    if src is not None:
        rep.params("Source", freq_hz=src.freq_hz, height_m=src.height_m, beam_width_deg=src.beam_width_deg,
                   elevation_angle_deg=src.elevation_angle_deg, polarization=src.polarization)
    if env is not None:
        gm = env.ground_material
        rep.params("Environment", N_profile=type(env.N_profile).__name__, M0=env.M0,
                   slope_M_per_m=env.slope, terrain="none" if env.terrain is None else "piecewise linear",
                   ground="PEC" if gm is None else f"eps={gm.eps}, sigma={gm.sigma} S/m",
                   vegetation="none" if env.vegetation is None else "slab")
    if params is not None:
        rep.params("Computational parameters", max_range_m=params.max_range_m, max_height_m=params.max_height_m,
                   dx_output_m=params.dx_m, dz_output_m=params.dz_m, precision=params.precision)
    if model is not None:
        rep.params("SSF grid", beta=model.beta, dx_m=model.dx_m, dz_m=model.dz_m, x_n=model.x_n,
                   z_n=model.z_n, z_n_physical=model.z_n_phys, lower_bc=model.lower_bc,
                   lower_impedance_alpha=model.lower_impedance_alpha)


def _compare_cuts(rep, ref, other, labels, heights, thresholds, x_skip=0.2, floor_db=-30.0, normalize=False):
    """Cut figures plus median/p90/max checks against ``thresholds``.

    ``thresholds`` maps 'median' / 'p90' / 'max' to the allowed dB value.
    """
    fields = {labels[0]: ref.field, labels[1]: other.field}
    rep.cuts(f"Horizontal cuts of |u| ({labels[0]} vs {labels[1]}) and their difference in dB.",
             fields, ref.x_grid, ref.z_grid, heights)
    rows = []
    for z in heights:
        med, p90, mx = _db_stats(ref, other, z, x_skip=x_skip, floor_db=floor_db, normalize=normalize)
        rows.append([f"{z:g}", med, p90, mx])
        for key, val in (("median", med), ("p90", p90), ("max", mx)):
            if key in thresholds:
                rep.check(f"{key} |dB difference| at z = {z:g} m", val, "<", thresholds[key], "dB")
    rep.table(f"dB difference statistics (x > {x_skip:.0%} of the range, points within {-floor_db:g} dB "
              f"of the cut maximum)", ["z (m)", "median", "90th percentile", "max"], rows)


class TestSSFPropagatorBasics(unittest.TestCase):

    def _make(self, lower_bc='dirichlet', alpha=None, **kw):
        freq = 300e6
        k0 = 2 * fm.pi * freq / 3e8
        return SSFHelmholtzPropagator.create(
            freq_hz=freq, wave_speed=ConstWaveSpeedModel(3e8), kz_max=k0 * fm.sin(fm.radians(10)),
            k_bounds=(k0 * 0.999, k0 * 1.001), precision=0.01,
            mesh_params=HelmholtzMeshParams2D(x_size_m=1000.0, z_size_m=200.0, dx_output_m=50.0, dz_output_m=1.0),
            lower_bc=lower_bc, lower_impedance_alpha=alpha, **kw)

    def test_grid_shapes_and_output(self):
        with TestReport(self, """
            The propagator must produce output on exactly the requested output grid
            (1 km range with 50 m output step, 200 m height with 1 m step), append an
            absorbing layer above the physical domain that is not part of the output,
            and return the untouched initial field in the first column.
            """) as rep:
            model = self._make()
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=50.0, beam_width_deg=10.0)
            _report_setup(rep, src=src, model=model)
            f = np.asarray(model.compute(src.aperture(model.z_computational_grid())))
            rep.check("z output decimation", model.z_grid_scale, "==", 1)
            rep.check("dx * x_grid_scale", model.dx_m * model.x_grid_scale, "==", 50.0, "m")
            rep.check("number of output ranges", len(model.x_output_grid()), "==", 21)
            rep.check("number of output heights", len(model.z_output_grid()), "==", 201)
            rep.check("absorbing layer points", model.z_n - model.z_n_phys, ">", 0)
            rep.check("output shape rows", f.shape[0], "==", 21)
            rep.check("output shape cols", f.shape[1], "==", 201)
            rep.check("all values finite", bool(np.all(np.isfinite(f))), "==", True)
            init_err = float(np.max(np.abs(f[0] - np.asarray(src.aperture(model.z_physical_grid())))))
            rep.check("first column equals initial field (max error)", init_err, "<", 1e-12)
            rep.field2d("Propagated field on the output grid.", {"SSF": f}, model.x_output_grid(),
                        model.z_output_grid())
            rep.lines("Initial field and absorbing window on the full computational grid.",
                      model.z_computational_grid(),
                      {"|initial|": np.abs(np.asarray(src.aperture(model.z_computational_grid()))),
                       "absorbing window": np.asarray(model._abs_window)}, xlabel="Height (m)")
            rep.assert_all()

    def test_explicit_steps_divide_output_steps(self):
        with TestReport(self, """
            User-supplied computational steps (dx = 12 m, dz = 0.3 m) must be rounded
            down so that the requested output steps (50 m, 1 m) are integer multiples:
            dx -> 10 m (scale 5), dz -> 0.25 m (scale 4). This mirrors the convention
            of the rational finite-difference propagator.
            """) as rep:
            model = self._make(dx_m=12.0, dz_m=0.3)
            _report_setup(rep, model=model)
            rep.params("Requested", dx_m=12.0, dz_m=0.3, dx_output_m=50.0, dz_output_m=1.0)
            rep.check("x_grid_scale", model.x_grid_scale, "==", 5)
            rep.check("dx_m", abs(model.dx_m - 10.0), "<", 1e-12, "m")
            rep.check("z_grid_scale", model.z_grid_scale, "==", 4)
            rep.check("dz_m", abs(model.dz_m - 0.25), "<", 1e-12, "m")
            rep.assert_all()

    def test_invalid_bc(self):
        with TestReport(self, """
            Constructing a propagator with an unknown lower boundary condition, or
            with the impedance condition but no impedance coefficient, must raise
            ValueError.
            """) as rep:
            for label, kw in (("unknown bc 'robin'", dict(lower_bc='robin')),
                              ("impedance without alpha", dict(lower_bc='impedance'))):
                try:
                    self._make(**kw)
                    raised = False
                except ValueError:
                    raised = True
                rep.check(f"{label} raises ValueError", raised, "==", True)
            rep.assert_all()

    def test_pytree_roundtrip_and_jit(self):
        with TestReport(self, """
            The propagator is registered as a JAX pytree. Flattening and unflattening
            it, then running compute inside an outer jax.jit, must reproduce the
            result of the original object to round-off (relative 1e-10).
            """) as rep:
            model = self._make(lower_bc='impedance', alpha=0.3 + 0.5j)
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=50.0, beam_width_deg=10.0, polarization='V')
            _report_setup(rep, src=src, model=model)
            leaves, treedef = jax.tree_util.tree_flatten(model)
            model2 = treedef.unflatten(leaves)
            init = src.aperture(model.z_computational_grid())
            f1 = np.asarray(model.compute(init))
            f2 = np.asarray(jax.jit(lambda m, u: m.compute(u))(model2, init))
            rel = float(np.max(np.abs(f1 - f2)) / np.max(np.abs(f1)))
            rep.params("Pytree", n_leaves=len(leaves))
            rep.check("max relative difference original vs unflattened+jit", rel, "<", 1e-10)
            rep.field2d("Field computed by the original object and by the unflattened copy under jit.",
                        {"original": f1, "unflattened + jit": f2}, model.x_output_grid(), model.z_output_grid())
            rep.assert_all()

    def test_dmft_is_identity_without_propagation(self):
        with TestReport(self, """
            With the spectral and surface-wave multipliers set to one, one
            impedance-boundary step reduces to forward DMFT followed by inverse DMFT.
            The transform pair (sine spectrum of w_j = (u_{j+1}-u_{j-1})/(2 dz) + alpha u_j
            plus the two discrete surface-wave modes r^j and (-1/r)^j with amplitudes
            from the end-point residuals) must therefore reproduce an arbitrary field
            to round-off, for lossy, Re(alpha) < 0 and strongly imaginary alpha.
            """) as rep:
            rows = []
            errs = {}
            for alpha in (0.3 + 0.2j, 2.0 - 1.5j, -0.4 + 3.0j):
                model = self._make(lower_bc='impedance', alpha=alpha)
                model._spectral_mult = jnp.ones_like(model._spectral_mult)
                model._surface_mult = jnp.asarray(1.0 + 0j)
                model._surface_mult_top = jnp.asarray(1.0 + 0j)
                z = model.z_computational_grid()
                u = jnp.exp(-((z - 80.0) / 20.0) ** 2) * jnp.exp(0.3j * z) + 0.2 * jnp.cos(0.05 * z)
                u_back = model._spectral_step_impedance(u)
                err = np.abs(np.asarray(u_back) - np.asarray(u))
                errs[f"alpha = {alpha}"] = err
                rows.append([alpha, model._r, abs(model._r), float(err.max())])
                rep.check(f"max |inverse(forward(u)) - u| for alpha = {alpha}", float(err.max()), "<", 1e-9)
            rep.table("Surface-wave root r of r^2 + 2 alpha dz r - 1 = 0 (|r| < 1) and round-trip error",
                      ["alpha (1/m)", "r", "|r|", "max error"], rows)
            rep.lines("Test field used for the round trip.", z, {"|u|": np.abs(np.asarray(u))}, xlabel="Height (m)")
            rep.lines("Round-trip error |inverse(forward(u)) - u| along height.", z, errs, xlabel="Height (m)",
                      ylabel="absolute error")
            rep.data(z=z, u=np.asarray(u))
            rep.assert_all()

    def test_dirichlet_and_neumann_steps_preserve_symmetry(self):
        with TestReport(self, """
            The sine-transform (Dirichlet) step must keep u(0) = 0 and the
            cosine-transform (Neumann) step must keep the field even about z = 0,
            i.e. u(dz) - u(0) must vanish to the discretisation order. The test field
            is a Gaussian at 60 m combined with its image at -60 m.
            """) as rep:
            model_d = self._make(lower_bc='dirichlet')
            model_n = self._make(lower_bc='neumann')
            _report_setup(rep, model=model_d)
            z = model_d.z_computational_grid()
            g = jnp.exp(-((z - 60.0) / 15.0) ** 2)
            odd = g - jnp.exp(-((-z - 60.0) / 15.0) ** 2)
            even = g + jnp.exp(-((-z - 60.0) / 15.0) ** 2)
            u_d = model_d._spectral_step(odd)
            u_n = model_n._spectral_step(even)
            scale = float(jnp.max(jnp.abs(u_n)))
            rep.check("|u_dirichlet(0)| / max|u|", abs(complex(u_d[0])) / scale, "<", 1e-12)
            rep.check("|u_neumann(dz) - u_neumann(0)| / max|u|", abs(complex(u_n[1] - u_n[0])) / scale, "<", 1e-6)
            rep.lines("Field before and after one Dirichlet (sine) and one Neumann (cosine) spectral step.",
                      z, {"|odd input|": np.abs(np.asarray(odd)), "|Dirichlet step|": np.abs(np.asarray(u_d)),
                          "|even input|": np.abs(np.asarray(even)), "|Neumann step|": np.abs(np.asarray(u_n))},
                      xlabel="Height (m)")
            rep.assert_all()

    def test_free_space_energy_conservation(self):
        with TestReport(self, """
            Without refraction, loss or absorbing layer the spectral step is a
            unitary operator: the L2 norm of a beam launched at ~23 deg must be
            unchanged after several steps, even though the beam moves and spreads.
            """) as rep:
            model = self._make(abs_layer_scale=0.0)
            _report_setup(rep, model=model)
            z = model.z_computational_grid()
            u0 = jnp.exp(-((z - 100.0) / 10.0) ** 2) * jnp.exp(0.4j * z)
            e0 = float(jnp.sum(jnp.abs(u0) ** 2))
            u = u0
            energies = [1.0]
            profiles = {"step 0": np.abs(np.asarray(u0))}
            for i in range(1, 4):
                u = model._spectral_step(u)
                energies.append(float(jnp.sum(jnp.abs(u) ** 2)) / e0)
                profiles[f"step {i}"] = np.abs(np.asarray(u))
            rep.check("relative energy after 3 steps", abs(energies[-1] - 1.0), "<", 1e-6)
            rep.table("Relative energy per step", ["step", "energy / energy_0"], list(enumerate(energies)))
            rep.lines("Beam amplitude after each free-space step (the beam tilts upwards).", z, profiles,
                      xlabel="Height (m)")
            rep.assert_all()

    def test_impedance_alpha(self):
        with TestReport(self, """
            The Leontovich coefficient alpha in du/dz + alpha u = 0 is
            i k0 sqrt(eps - 1) for horizontal and i k0 sqrt(eps - 1) / eps for vertical
            polarisation. For vertical polarisation over lossy ground Re(alpha) > 0,
            so the DMFT surface-wave mode decays with height (the physical Zenneck wave).
            """) as rep:
            k0 = 2 * fm.pi * 300e6 / 3e8
            eps = 3.0 + 0.0j
            a_h = impedance_alpha(k0, eps, 'H')
            a_v = impedance_alpha(k0, eps, 'V')
            a_sea = impedance_alpha(k0, 80 + 240j, 'V')
            rep.table("Computed coefficients", ["case", "alpha (1/m)", "expected"],
                      [["H, eps=3", a_h, 1j * k0 * fm.sqrt(2.0)], ["V, eps=3", a_v, 1j * k0 * fm.sqrt(2.0) / 3.0],
                       ["V, eps=80+240j", a_sea, "Re > 0"]])
            rep.check("H-pol error", abs(a_h - 1j * k0 * fm.sqrt(2.0)), "<", 1e-12)
            rep.check("V-pol error", abs(a_v - 1j * k0 * fm.sqrt(2.0) / 3.0), "<", 1e-12)
            rep.check("Re(alpha) for V-pol over sea water", a_sea.real, ">", 0.0)
            rep.assert_all()


class TestSSFvsTwoRay(unittest.TestCase):
    """Flat ground and homogeneous atmosphere: the two-ray model is exact."""

    def _case(self, pol, eps=None, sigma=0.0):
        with TestReport(self, f"""
            Flat ground ({'PEC' if eps is None else f'eps = {eps}, sigma = {sigma} S/m'}),
            homogeneous atmosphere, {pol}-polarisation, 300 MHz, antenna at 30 m with a
            5 deg beam. In this geometry the two-ray model (direct + Fresnel-reflected
            ray) is exact, so it serves as an analytical reference. The two-ray field
            is multiplied by sqrt(x) to match the PE reduced field, and both fields
            are normalised to their own maximum on every cut. Points deeper than 30 dB
            below the cut maximum (interference nulls) are excluded from the
            statistics.
            """) as rep:
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization=pol)
            env = TroposphereModel(slope=0, ground_material=None if eps is None else GroundMaterial(eps=eps, sigma=sigma))
            params = RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=50.0, dz_m=1.0)
            params.approx_method = 'ssf'
            model = create_rwp_ssf_model(src, env, params)
            _report_setup(rep, src, env, params, model)
            ssf = rwp_forward_task(src, env, params)
            two_ray = _two_ray_field(src, ssf.x_grid, ssf.z_grid, eps=eps, sigma=sigma)
            rep.field2d("Two-ray reference (times sqrt(x)) and SSF field.",
                        {"two-ray": two_ray.field, "SSF": ssf.field}, ssf.x_grid, ssf.z_grid)
            _compare_cuts(rep, two_ray, ssf, ("two-ray", "SSF"), [5.0, 30.0, 100.0, 200.0],
                          {"median": 0.1, "max": 0.5}, x_skip=0.3, floor_db=-30, normalize=True)
            rep.assert_all()

    def test_pec_h_pol(self):
        self._case('H')

    def test_dielectric_v_pol_brewster(self):
        self._case('V', eps=3.0)

    def test_dielectric_h_pol(self):
        self._case('H', eps=3.0)

    def test_sea_water_v_pol(self):
        self._case('V', eps=80.0, sigma=4.0)

    def test_sea_water_h_pol(self):
        self._case('H', eps=80.0, sigma=4.0)


class TestSSFvsRationalFD(unittest.TestCase):
    """SSF against the Padé finite-difference propagator on RWP problems."""

    def test_standard_atmosphere(self):
        with TestReport(self, """
            Standard atmosphere (M gradient 0.118 M-units/m) over PEC ground,
            300 MHz, H-polarisation, 10 km range. The SSF field is compared with the
            Padé (7, 8) finite-difference propagator of the package on the same
            output grid. Beyond 20% of the range the two solutions must agree to
            0.5 dB in the median and 2 dB at the 90th percentile.
            """) as rep:
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0)
            env = TroposphereModel()
            params = RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=50.0, dz_m=1.0, precision=0.01)
            _report_setup(rep, src, env, params, create_rwp_ssf_model(src, env, params))
            fd = _run(src, env, params, 'pade')
            ssf = _run(src, env, params, 'ssf')
            rep.check("same output shape", fd.field.shape == ssf.field.shape, "==", True)
            rep.check("max |x grid difference|", float(np.max(np.abs(fd.x_grid - ssf.x_grid))), "<", 1e-9, "m")
            rep.check("max |z grid difference|", float(np.max(np.abs(fd.z_grid - ssf.z_grid))), "<", 1e-9, "m")
            rep.field2d("Padé FD and SSF fields.", {"Padé FD": fd.field, "SSF": ssf.field}, fd.x_grid, fd.z_grid)
            _compare_cuts(rep, fd, ssf, ("Padé FD", "SSF"), [10.0, 30.0, 100.0], {"median": 0.5, "p90": 2.0})
            rep.assert_all()

    def test_evaporation_duct(self):
        with TestReport(self, """
            Evaporation duct of 20 m height at 3 GHz, antenna at 10 m with a 3 deg
            beam, 20 km range. The duct is a strong near-surface refractivity
            gradient that traps the field; the SSF handles it through the phase
            screen. The SSF is compared with the Padé FD solution near the surface,
            and additionally the trapped field at 10 m height and 20 km must exceed
            the standard-atmosphere level by more than 10 dB.
            """) as rep:
            src = RWPGaussSourceModel(freq_hz=3e9, height_m=10.0, beam_width_deg=3.0)
            env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
            params = RWPComputationalParams(max_range_m=20000.0, max_height_m=150.0, dx_m=100.0, dz_m=0.5, precision=0.01)
            _report_setup(rep, src, env, params, create_rwp_ssf_model(src, env, params))
            z_prof = np.linspace(0, 150, 301)
            rep.lines("Modified refractivity profile of the evaporation duct.", np.asarray(env.M_profile(z_prof)),
                      {"M(z)": z_prof}, xlabel="M (M-units)", ylabel="Height (m)")
            fd = _run(src, env, params, 'pade')
            ssf = _run(src, env, params, 'ssf')
            ssf_std = _run(src, TroposphereModel(), params, 'ssf')
            rep.field2d("Padé FD and SSF fields in the duct.", {"Padé FD": fd.field, "SSF": ssf.field},
                        fd.x_grid, fd.z_grid)
            _compare_cuts(rep, fd, ssf, ("Padé FD", "SSF"), [5.0, 10.0, 30.0], {"median": 0.5, "p90": 2.0})
            iz = int(np.argmin(np.abs(ssf.z_grid - 10.0)))
            gain = 20 * np.log10(np.abs(ssf.field[-1, iz]) / np.abs(ssf_std.field[-1, iz]))
            rep.cuts("SSF field at 10 m height with and without the duct.",
                     {"SSF, duct": ssf.field, "SSF, standard atmosphere": ssf_std.field},
                     ssf.x_grid, ssf.z_grid, [10.0], diff=False)
            rep.check("duct gain over standard atmosphere at z = 10 m, x = 20 km", float(gain), ">", 10.0, "dB")
            rep.assert_all()

    def test_surface_based_duct(self):
        with TestReport(self, """
            Surface-based duct given by a piecewise linear N profile (-20 N-units
            between 100 m and 200 m), 1 GHz, antenna at 50 m with a 4 deg beam, 20 km.
            The SSF is compared with the Padé FD solution inside and above the duct.
            """) as rep:
            z_grid = jnp.array([0.0, 100.0, 200.0, 300.0])
            N_vals = jnp.array([0.0, -20.0, -20.0, 0.0])
            env = TroposphereModel(N_profile=PiecewiseLinearNProfileModel(z_grid_m=z_grid, N_vals=N_vals))
            src = RWPGaussSourceModel(freq_hz=1e9, height_m=50.0, beam_width_deg=4.0)
            params = RWPComputationalParams(max_range_m=20000.0, max_height_m=400.0, dx_m=100.0, dz_m=1.0, precision=0.01)
            _report_setup(rep, src, env, params, create_rwp_ssf_model(src, env, params))
            z_prof = np.linspace(0, 400, 401)
            rep.lines("Modified refractivity profile of the surface-based duct.", np.asarray(env.M_profile(z_prof)),
                      {"M(z)": z_prof}, xlabel="M (M-units)", ylabel="Height (m)")
            fd = _run(src, env, params, 'pade')
            ssf = _run(src, env, params, 'ssf')
            rep.field2d("Padé FD and SSF fields.", {"Padé FD": fd.field, "SSF": ssf.field}, fd.x_grid, fd.z_grid)
            _compare_cuts(rep, fd, ssf, ("Padé FD", "SSF"), [20.0, 50.0, 150.0], {"median": 0.5, "p90": 2.0})
            rep.assert_all()

    def test_terrain_hill(self):
        with TestReport(self, """
            Triangular hill of 80 m height between 3 km and 7 km, 300 MHz, antenna at
            50 m with an 8 deg beam. Both solvers model terrain as a staircase mask
            applied after every range step, so the field inside the hill must be
            exactly zero in both, and in the diffraction zone behind the hill (last
            quarter of the range) the two solutions must agree to 1.5 dB in the
            median and 3 dB at most.
            """) as rep:
            x_t = jnp.array([0.0, 3000.0, 5000.0, 7000.0, 10000.0])
            h_t = jnp.array([0.0, 0.0, 80.0, 0.0, 0.0])
            terrain = PiecewiseLinearTerrainModel(x_grid_m=x_t, height=h_t)
            env = TroposphereModel(terrain=terrain)
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=50.0, beam_width_deg=8.0)
            params = RWPComputationalParams(max_range_m=10000.0, max_height_m=400.0, dx_m=50.0, dz_m=1.0, precision=0.01)
            _report_setup(rep, src, env, params, create_rwp_ssf_model(src, env, params))
            fd = _run(src, env, params, 'pade')
            ssf = _run(src, env, params, 'ssf')
            rep.field2d("Padé FD and SSF fields over the hill (black line: terrain).",
                        {"Padé FD": fd.field, "SSF": ssf.field}, fd.x_grid, fd.z_grid,
                        terrain=np.asarray(terrain(jnp.asarray(fd.x_grid))))
            ix = int(np.argmin(np.abs(ssf.x_grid - 5000.0)))
            rep.check("max |u| inside the hill (SSF, x = 5 km, z < 70 m)", float(np.max(np.abs(ssf.field[ix, :70]))), "==", 0.0)
            rep.check("max |u| inside the hill (FD, x = 5 km, z < 70 m)", float(np.max(np.abs(fd.field[ix, :70]))), "==", 0.0)
            _compare_cuts(rep, fd, ssf, ("Padé FD", "SSF"), [30.0, 100.0, 200.0], {"median": 1.5, "max": 3.0},
                          x_skip=0.75)
            rep.assert_all()

    def test_impedance_ground(self):
        params = RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=50.0, dz_m=1.0, precision=0.01)
        for pol, eps, sigma in (('V', 3.0, 0.0), ('H', 3.0, 0.0), ('V', 80.0, 4.0), ('H', 80.0, 4.0),
                                ('V', 15.0, 0.005)):
            with self.subTest(pol=pol, eps=eps, sigma=sigma):
                with TestReport(self, f"""
                    Finite-permittivity ground (eps = {eps}, sigma = {sigma} S/m),
                    {pol}-polarisation, standard atmosphere, 300 MHz, antenna at 30 m
                    with a 5 deg beam. The SSF uses the discrete mixed Fourier transform
                    with the Leontovich impedance condition; the Padé FD solver uses its
                    nonlocal boundary condition built from the Fresnel reflection
                    coefficient. The two must agree to 0.2 dB in the median and 1 dB at
                    most at 5 m, 30 m and 100 m height.
                    """, title=f"impedance ground {pol}-pol eps={eps:g} sigma={sigma:g}",
                                suffix=f"{pol}_eps{eps:g}") as rep:
                    src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization=pol)
                    env = TroposphereModel(ground_material=GroundMaterial(eps=eps, sigma=sigma))
                    _report_setup(rep, src, env, params, create_rwp_ssf_model(src, env, params))
                    fd = _run(src, env, params, 'pade')
                    ssf = _run(src, env, params, 'ssf')
                    rep.field2d("Padé FD (lower NLBC) and SSF (DMFT) fields.",
                                {"Padé FD": fd.field, "SSF": ssf.field}, fd.x_grid, fd.z_grid)
                    _compare_cuts(rep, fd, ssf, ("Padé FD", "SSF"), [5.0, 30.0, 100.0], {"median": 0.2, "max": 1.0})
                    rep.assert_all()

    def test_polarization_matters_over_dielectric(self):
        with TestReport(self, """
            Over a dielectric with eps = 3 the Brewster angle is 30 deg grazing, where
            the V-polarised reflection vanishes while the H-polarised one does not.
            With a wide (20 deg) beam and the cut at the antenna height (30 m) over
            the first kilometre, the reflected ray covers grazing angles of 3-17 deg,
            so the H and V interference patterns computed by the SSF (impedance
            boundary via DMFT) must differ noticeably.
            """) as rep:
            params = RWPComputationalParams(max_range_m=1000.0, max_height_m=200.0, dx_m=10.0, dz_m=1.0)
            params.approx_method = 'ssf'
            env = TroposphereModel(ground_material=GroundMaterial(eps=3.0, sigma=0.0))
            fields = {}
            for pol in ('H', 'V'):
                src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=20.0, polarization=pol)
                fields[pol] = rwp_forward_task(src, env, params)
                _report_setup(rep, src=src, model=create_rwp_ssf_model(src, env, params))
            _report_setup(rep, env=env, params=params)
            h_f, v_f = fields['H'], fields['V']
            rep.field2d("H- and V-polarised fields over the dielectric.", {"H-pol": h_f.field, "V-pol": v_f.field},
                        h_f.x_grid, h_f.z_grid)
            rep.cuts("Cuts at the antenna height and their difference.", {"H-pol": h_f.field, "V-pol": v_f.field},
                     h_f.x_grid, h_f.z_grid, [30.0, 100.0])
            iz = int(np.argmin(np.abs(h_f.z_grid - 30.0)))
            diff_db = np.abs(20 * np.log10(np.abs(h_f.field[20:, iz]) / np.abs(v_f.field[20:, iz])))
            rep.check("median |H - V| at z = 30 m, x > 200 m", float(np.median(diff_db)), ">", 0.3, "dB")
            rep.check("max |H - V| at z = 30 m, x > 200 m", float(np.max(diff_db)), ">", 3.0, "dB")
            rep.assert_all()

    def test_pec_lower_bc_follows_polarization(self):
        with TestReport(self, """
            The RWP front-end must select the lower boundary condition of the SSF
            propagator from the source polarisation and the ground material:
            Dirichlet (sine transform) for H over PEC, Neumann (cosine transform)
            for V over PEC, and the impedance condition (DMFT) whenever a ground
            material is given. create_rwp_model must dispatch to the SSF class when
            approx_method = 'ssf'.
            """) as rep:
            params = RWPComputationalParams(max_range_m=1000.0, max_height_m=200.0, dx_m=50.0, dz_m=1.0)
            env = TroposphereModel()
            rows = []
            for pol, bc in (('H', 'dirichlet'), ('V', 'neumann')):
                src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization=pol)
                got = create_rwp_ssf_model(src, env, params).lower_bc
                rows.append([pol, "PEC", bc, got])
                rep.check(f"lower_bc for {pol}-pol over PEC", got, "==", bc)
            env = TroposphereModel(ground_material=GroundMaterial(eps=15.0, sigma=0.005))
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization='V')
            params.approx_method = 'ssf'
            model = create_rwp_model(src, env, params)
            rows.append(['V', "eps=15, sigma=0.005", 'impedance', model.lower_bc])
            rep.table("Selected boundary conditions", ["polarisation", "ground", "expected", "selected"], rows)
            rep.check("create_rwp_model returns SSFHelmholtzPropagator", isinstance(model, SSFHelmholtzPropagator), "==", True)
            rep.check("lower_bc with ground material", model.lower_bc, "==", 'impedance')
            rep.assert_all()

    def test_vegetation_slab_attenuation_matches_theory(self):
        with TestReport(self, """
            A vegetation slab (eps = 1.05, sigma = 2e-4 S/m) thick enough to fill the
            whole computational domain turns the atmosphere into a homogeneous lossy
            dielectric. The ratio of the field with and without the slab at 500 m
            range and 100 m height must then follow the plane-wave attenuation
            -8.686 k0 Im(sqrt(eps)) x to within 2%. This checks the complex
            (lossy) heterogeneity path of the SSF phase screen.
            """) as rep:
            f = 300e6
            k0 = 2 * fm.pi * f / 3e8
            x = jnp.linspace(0, 1000, 11)
            terrain = PiecewiseLinearTerrainModel(x, jnp.zeros_like(x))
            material = GroundMaterial(eps=1.05, sigma=2e-4)
            src = RWPGaussSourceModel(freq_hz=f, height_m=100.0, beam_width_deg=5.0, polarization='V')
            params = RWPComputationalParams(max_range_m=1000, max_height_m=200, dx_m=20.0, dz_m=2.0)
            params.approx_method = 'ssf'
            fields = {}
            for tag, veg in (("air", None),
                             ("slab", VegetationLayerModel(
                                 thickness=PiecewiseLinearTerrainModel(x, jnp.full_like(x, 300.0)),
                                 material=material))):
                env = TroposphereModel(terrain=terrain, ground_material=GroundMaterial(eps=15, sigma=0.005),
                                       vegetation=veg)
                fields[tag] = rwp_forward_task(src, env, params)
            _report_setup(rep, src, env, params, create_rwp_ssf_model(src, env, params))
            rep.params("Slab material", eps=material.eps, sigma=material.sigma,
                       complex_permittivity=material.complex_permittivity(f))
            u_air, u_slab = fields["air"], fields["slab"]
            iz = int(np.argmin(np.abs(u_air.z_grid - 100.0)))
            ix = int(np.argmin(np.abs(u_air.x_grid - 500.0)))
            ratio_db = 20 * np.log10(np.abs(u_slab.field[ix, iz]) / np.abs(u_air.field[ix, iz]))
            theory_db = -8.686 * k0 * cmath.sqrt(material.complex_permittivity(f)).imag * u_air.x_grid[ix]
            rep.field2d("Field without and with the lossy slab.", {"air": u_air.field, "slab": u_slab.field},
                        u_air.x_grid, u_air.z_grid, dyn_range_db=200.0)
            ratio = 20 * np.log10(np.abs(u_slab.field[:, iz]) / (np.abs(u_air.field[:, iz]) + 1e-300) + 1e-300)
            rep.lines("Attenuation slab/air at 100 m height versus the plane-wave theory.", u_air.x_grid,
                      {"SSF slab / air": ratio,
                       "theory": -8.686 * k0 * cmath.sqrt(material.complex_permittivity(f)).imag * u_air.x_grid},
                      xlabel="Range (m)", ylabel="dB")
            rep.params("At x = 500 m, z = 100 m", ssf_ratio_db=float(ratio_db), theory_db=float(theory_db))
            rep.check("relative error of the attenuation", abs(ratio_db - theory_db) / abs(theory_db), "<", 0.02)
            rep.assert_all()


if __name__ == '__main__':
    unittest.main()
