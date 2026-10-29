"""
Tests for the JAX-based rank-adaptive split-step Fourier (RA-SSF) propagator.

The RA-SSF replaces the rank-one (phase screen x spectral multiplier)
factorisation of the one-way symbol used by the classical SSF with a
rank-adaptive cross approximation of the exact symbol
``exp(i dx/2 (sqrt(k(z)^2 - p^2) - beta))``. It is validated against

* the classical SSF (rank one must be recovered exactly for homogeneous media),
* the exact solution of the semi-discrete one-way equation for stratified
  media (eigen-decomposition of ``D2 + k^2(z)`` in the sine basis), which
  isolates the splitting / symbol error from all other approximations,
* the analytical two-ray model (flat ground, homogeneous atmosphere), and
* the JAX rational finite-difference (Padé) propagator on ducting, terrain,
  finite-permittivity ground and vegetation problems,

all with range steps equal to the output step (up to 1 km), i.e. far beyond
the paraxial step rule of the SSF.

Every test writes a self-contained HTML report (see ``reporting.py``).
"""
import cmath
import math as fm
import time
import unittest
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_jax import ConstWaveSpeedModel, PiecewiseLinearTerrainModel
from pywaveprop.helmholtz_rassf_jax import (RankAdaptiveSSFPropagator, cross_approximation,
                                            exact_one_way_stratified, symbol_factors)
from pywaveprop.helmholtz_ssf_jax import SSFHelmholtzPropagator
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
    create_rwp_rassf_model,
    create_rwp_ssf_model,
    rwp_forward_task,
)
from tests.pywaveprop_jax.reporting import TestReport
from tests.pywaveprop_jax.test_ssf_jax import _compare_cuts, _db_stats, _two_ray_field


def _run(src, env, params, method, **kwargs):
    p = deepcopy(params)
    p.approx_method = method
    if method not in ('ssf', 'rassf', 'ssw'):
        p.rational_approx_order = (7, 8)
    return rwp_forward_task(src, env, p)


def _report_setup(rep, src=None, env=None, params=None, model=None):
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
        rep.params("RA-SSF grid and symbol approximation", beta=model.beta, dx_m=model.dx_m, dz_m=model.dz_m,
                   x_n=model.x_n, z_n=model.z_n, z_n_physical=model.z_n_phys, lower_bc=model.lower_bc,
                   composition=model.composition, absorber=model.absorber, symbol_tol=model.symbol_tol,
                   ranks=list(model.ranks), ffts_per_step=model.ffts_per_step,
                   max_sampled_symbol_error=max(i['max_sampled_error'] for i in model.aca_info))


def _field(model, init):
    return RWPField(x_grid=np.asarray(model.x_output_grid()), z_grid=np.asarray(model.z_output_grid()),
                    freq_hz=model.freq_hz, field=np.asarray(model.compute(init)))


class TestCrossApproximation(unittest.TestCase):

    def test_random_low_rank_matrix_is_recovered(self):
        with TestReport(self, """
            The cross approximation must recover an exactly rank-5 complex matrix
            (200 x 300) to round-off with rank 5, and approximate the smooth
            oscillatory kernel exp(i a z p) (z, p in [0, 1], a = 12) to the requested
            tolerance with a rank that grows as the tolerance is tightened.
            """) as rep:
            rng = np.random.default_rng(1)
            a_low = (rng.standard_normal((200, 5)) + 1j * rng.standard_normal((200, 5))) @ \
                    (rng.standard_normal((5, 300)) + 1j * rng.standard_normal((5, 300)))
            u, v, info = cross_approximation(lambda i: a_low[i], lambda j: a_low[:, j], a_low.shape, 1e-10)
            err = np.max(np.abs(a_low - u @ v)) / np.max(np.abs(a_low))
            rep.check("rank of the exact rank-5 matrix", info['rank'], "==", 5)
            rep.check("relative max error, rank-5 matrix", err, "<", 1e-8)
            z = np.linspace(0, 1, 400)
            p = np.linspace(0, 1, 500)
            kern = np.exp(12j * np.outer(z, p))
            rows = []
            prev_rank = 0
            for tol in (1e-2, 1e-4, 1e-6, 1e-8):
                u, v, info = cross_approximation(lambda i: kern[i], lambda j: kern[:, j], kern.shape, tol)
                err = np.max(np.abs(kern - u @ v))
                rows.append([tol, info['rank'], err, info['rows_evaluated'], info['cols_evaluated']])
                rep.check(f"max error of exp(i 12 z p) at tol = {tol:g}", err, "<", 10 * tol)
                rep.check(f"rank does not decrease at tol = {tol:g}", info['rank'], ">=", prev_rank)
                prev_rank = info['rank']
            rep.table("Cross approximation of exp(i 12 z p)", ["tol", "rank", "max error", "rows", "cols"], rows)
            rep.assert_all()

    def test_symbol_rank_grows_with_step_and_contrast(self):
        with TestReport(self, """
            For the one-way symbol exp(i dx (sqrt(k(z)^2 - p^2) - beta)) with a linear
            refractivity profile (1 GHz, 300 m, propagation angles up to 10 deg) the
            adaptive rank at tolerance 1e-6 must be 1 for a homogeneous medium and
            must not decrease when the range step or the refractivity contrast is
            increased. The sampled maximum error must stay below the tolerance.
            """) as rep:
            beta = 2 * fm.pi * 1e9 / 3e8
            z = np.linspace(0, 300, 601)
            p = np.linspace(0, beta * fm.sin(fm.radians(10)), 401)
            rows = []
            ranks = {}
            for contrast in (0.0, 1e-4, 1e-3):
                for dx in (10.0, 100.0, 1000.0):
                    k2 = beta ** 2 * (1.0 + contrast * z / 300.0)
                    _, _, info = symbol_factors(k2, p ** 2, beta, dx, 1e-6)
                    ranks[(contrast, dx)] = info['rank']
                    rows.append([contrast, dx, info['rank'], info['max_sampled_error']])
                    rep.check(f"sampled error, contrast {contrast:g}, dx {dx:g}", info['max_sampled_error'], "<", 1e-6)
            rep.table("Adaptive rank of the symbol", ["het contrast", "dx (m)", "rank", "max sampled error"], rows)
            rep.check("homogeneous medium, dx = 1000 m: rank", ranks[(0.0, 1000.0)], "==", 1)
            for contrast in (1e-4, 1e-3):
                rep.check(f"rank(dx=1000) >= rank(dx=10) at contrast {contrast:g}",
                          ranks[(contrast, 1000.0)], ">=", ranks[(contrast, 10.0)])
            rep.check("rank(contrast 1e-3) >= rank(contrast 1e-4) at dx = 1000 m",
                      ranks[(1e-3, 1000.0)], ">=", ranks[(1e-4, 1000.0)])
            rep.check("rank(contrast 1e-3, dx = 1000 m) > 1", ranks[(1e-3, 1000.0)], ">", 1)
            rep.assert_all()


class TestRASSFBasics(unittest.TestCase):

    def _make(self, lower_bc='dirichlet', alpha=None, wave_speed=None, **kw):
        freq = 300e6
        k0 = 2 * fm.pi * freq / 3e8
        return RankAdaptiveSSFPropagator.create(
            freq_hz=freq, wave_speed=wave_speed or ConstWaveSpeedModel(3e8), kz_max=k0 * fm.sin(fm.radians(10)),
            k_bounds=(k0 * 0.999, k0 * 1.001), precision=0.01,
            mesh_params=HelmholtzMeshParams2D(x_size_m=1000.0, z_size_m=200.0, dx_output_m=50.0, dz_output_m=1.0),
            lower_bc=lower_bc, lower_impedance_alpha=alpha, **kw)

    def _make_ssf(self, lower_bc='dirichlet', alpha=None, **kw):
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
            (1 km range with 50 m output step, 200 m height with 1 m step), take the
            output step as its range step (no paraxial step rule), append an
            absorbing layer above the physical domain that is not part of the output,
            and return the untouched initial field in the first column.
            """) as rep:
            model = self._make()
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=50.0, beam_width_deg=10.0)
            _report_setup(rep, src=src, model=model)
            f = np.asarray(model.compute(src.aperture(model.z_computational_grid())))
            rep.check("z output decimation", model.z_grid_scale, "==", 1)
            rep.check("x_grid_scale (range step = output step)", model.x_grid_scale, "==", 1)
            rep.check("dx_m", model.dx_m, "==", 50.0, "m")
            rep.check("number of output ranges", len(model.x_output_grid()), "==", 21)
            rep.check("number of output heights", len(model.z_output_grid()), "==", 201)
            rep.check("absorbing layer points", model.z_n - model.z_n_phys, ">", 0)
            rep.check("output shape rows", f.shape[0], "==", 21)
            rep.check("output shape cols", f.shape[1], "==", 201)
            rep.check("all values finite", bool(np.all(np.isfinite(f))), "==", True)
            init_err = float(np.max(np.abs(f[0] - np.asarray(src.aperture(model.z_physical_grid())))))
            rep.check("first column equals initial field (max error)", init_err, "<", 1e-12)
            rep.field2d("Propagated field on the output grid.", {"RA-SSF": f}, model.x_output_grid(),
                        model.z_output_grid())
            rep.lines("Absorbing profile Im(het) on the full computational grid.", model.z_computational_grid(),
                      {"Im het": np.asarray(model.het).imag}, xlabel="Height (m)")
            rep.assert_all()

    def test_invalid_arguments(self):
        with TestReport(self, """
            Unknown lower boundary condition, composition or absorber, or the
            impedance condition without a coefficient, must raise ValueError.
            """) as rep:
            for label, kw in (("unknown bc 'robin'", dict(lower_bc='robin')),
                              ("impedance without alpha", dict(lower_bc='impedance')),
                              ("unknown composition", dict(composition='strang')),
                              ("unknown absorber", dict(absorber='pml'))):
                try:
                    self._make(**kw)
                    raised = False
                except ValueError:
                    raised = True
                rep.check(f"{label} raises ValueError", raised, "==", True)
            rep.assert_all()

    def test_pytree_roundtrip_and_jit(self):
        with TestReport(self, """
            The propagator is registered as a JAX pytree carrying its low-rank
            factors as leaves. Flattening and unflattening it, then running compute
            inside an outer jax.jit, must reproduce the result of the original
            object to round-off (relative 1e-10); the impedance boundary is used so
            that the DMFT path is covered.
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
            rep.assert_all()

    def test_homogeneous_medium_recovers_ssf_with_rank_one(self):
        with TestReport(self, """
            In a homogeneous medium the one-way symbol is separable, so the cross
            approximation must return rank 1 and the RA-SSF step must coincide with
            the classical SSF step (phase screen = 1, exact spectral multiplier) to
            round-off, for the Dirichlet, Neumann and impedance (DMFT, including the
            surface-wave modes) boundaries and for both compositions. The cos^2
            window absorber is used in both solvers so the comparison is exact.
            """) as rep:
            rows = []
            for bc, alpha, pol in (('dirichlet', None, 'H'), ('neumann', None, 'V'), ('impedance', 0.3 + 0.5j, 'V')):
                src = RWPGaussSourceModel(freq_hz=300e6, height_m=50.0, beam_width_deg=10.0, polarization=pol)
                ssf = self._make_ssf(lower_bc=bc, alpha=alpha, dx_m=50.0)
                f_ssf = np.asarray(ssf.compute(src.aperture(ssf.z_computational_grid())))
                for comp in ('symmetric', 'one_sided'):
                    model = self._make(lower_bc=bc, alpha=alpha, absorber='window', composition=comp)
                    f = np.asarray(model.compute(src.aperture(model.z_computational_grid())))
                    rel = float(np.max(np.abs(f - f_ssf)) / np.max(np.abs(f_ssf)))
                    rows.append([bc, comp, model.ranks[0], rel])
                    rep.check(f"rank ({bc}, {comp})", model.ranks[0], "==", 1)
                    rep.check(f"max relative difference to SSF ({bc}, {comp})", rel, "<", 1e-9)
            rep.table("Rank and difference to the SSF", ["lower bc", "composition", "rank", "max rel. diff"], rows)
            rep.assert_all()

    def test_free_space_energy_conservation(self):
        with TestReport(self, """
            Without refraction, loss or absorbing layer the rank-one step is a
            unitary operator: the L2 norm of a beam launched at ~23 deg must be
            unchanged after several steps.
            """) as rep:
            model = self._make(abs_layer_scale=0.0)
            _report_setup(rep, model=model)
            z = model.z_computational_grid()
            u0 = jnp.exp(-((z - 100.0) / 10.0) ** 2) * jnp.exp(0.4j * z)
            e0 = float(jnp.sum(jnp.abs(u0) ** 2))
            u = u0
            energies = [1.0]
            for i in range(1, 4):
                u = model.step(u)
                energies.append(float(jnp.sum(jnp.abs(u) ** 2)) / e0)
            rep.check("relative energy after 3 steps", abs(energies[-1] - 1.0), "<", 1e-6)
            rep.table("Relative energy per step", ["step", "energy / energy_0"], list(enumerate(energies)))
            rep.assert_all()

    def test_profile_absorber_works_with_one_step_per_layer_crossing(self):
        with TestReport(self, """
            A 300 MHz beam tilted by 8 deg is launched from 100 m towards the top of a
            200 m domain with a 200 m absorbing layer and propagated 3 km, so that
            it enters the layer, reaches the hard upper wall of the transform and
            would come back into the physical domain. With the range step equal to
            the whole 1 km output step the beam crosses the layer within one or two
            steps. The absorbing profile embedded in the symbol (40 dB one-way at
            theta_max = 10 deg) must still remove it: the energy left in the
            physical domain at 3 km must be below -40 dB of the initial energy.
            The per-step cos^2 window of the SSF, applied only three times, is at
            least 10 dB less effective, which is what limits the SSF step size.
            """) as rep:
            freq = 300e6
            k0 = 2 * fm.pi * freq / 3e8
            mesh = HelmholtzMeshParams2D(x_size_m=3000.0, z_size_m=200.0, dx_output_m=1000.0, dz_output_m=1.0)
            common = dict(freq_hz=freq, wave_speed=ConstWaveSpeedModel(3e8), kz_max=k0 * fm.sin(fm.radians(10)),
                          k_bounds=(k0 * 0.999, k0 * 1.001), precision=0.01, mesh_params=mesh, abs_layer_scale=1.0)
            src = RWPGaussSourceModel(freq_hz=freq, height_m=100.0, beam_width_deg=3.0, elevation_angle_deg=-8.0)
            results = {}
            rows = []
            for label, model in (("RA-SSF, profile absorber", RankAdaptiveSSFPropagator.create(**common)),
                                 ("SSF, cos^2 window", SSFHelmholtzPropagator.create(dx_m=1000.0, **common)),
                                 ("SSF, profile absorber", SSFHelmholtzPropagator.create(dx_m=1000.0, absorber='profile',
                                                                                         **common))):
                init = src.aperture(model.z_computational_grid())
                f = np.asarray(model.compute(init))
                e0 = float(np.sum(np.abs(np.asarray(init)[:model.z_n_phys]) ** 2))
                e_end = float(np.sum(np.abs(f[-1]) ** 2))
                results[label] = f
                rows.append([label, model.dx_m, 10 * np.log10(e_end / e0)])
            rep.table("Energy in the physical domain at 3 km relative to the initial energy",
                      ["solver", "dx (m)", "dB"], rows)
            rep.check("RA-SSF (profile absorber) residual energy", rows[0][2], "<", -40.0, "dB")
            rep.check("SSF (cos^2 window, 3 steps) residual energy minus RA-SSF residual energy",
                      rows[1][2] - rows[0][2], ">", 10.0, "dB")
            rep.check("SSF (profile absorber, 3 steps) residual energy", rows[2][2], "<", -40.0, "dB")
            m0 = RankAdaptiveSSFPropagator.create(**common)
            rep.field2d("Fields on the output grid.", results, m0.x_output_grid(), m0.z_output_grid())
            rep.assert_all()


class TestRASSFvsExactOneWay(unittest.TestCase):
    """Stratified media without absorbing layer: the semi-discrete one-way
    equation is solved exactly by eigen-decomposition."""

    def test_wide_angle_layer_converges_where_ssf_does_not(self):
        with TestReport(self, """
            1 GHz, 20 deg beam from 50 m, strong elevated refractivity layer
            (-50 N-units between 100 m and 150 m), 10 km range, 300 m physical domain
            plus a 150 m hard-walled extension without absorption. The exact solution
            of the semi-discrete one-way equation is obtained by eigen-decomposition
            of D2 + k^2(z).
            The classical SSF has a step-independent error floor caused by its
            phase-screen (rank-one) factorisation of the wide-angle symbol, while
            the RA-SSF must converge to the exact solution as the step is refined
            and must beat the SSF floor already with 250 m steps.
            """) as rep:
            z_grid = jnp.array([0.0, 100.0, 150.0, 400.0])
            env = TroposphereModel(N_profile=PiecewiseLinearNProfileModel(
                z_grid_m=z_grid, N_vals=jnp.array([0.0, 0.0, -50.0, -50.0])))
            src = RWPGaussSourceModel(freq_hz=1e9, height_m=50.0, beam_width_deg=20.0)
            params = RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=250.0, dz_m=1.0)
            ref_model = create_rwp_rassf_model(src, env, params, abs_attenuation_db=0.0)
            _report_setup(rep, src, env, params, ref_model)
            init = src.aperture(ref_model.z_computational_grid())
            t0 = time.time()
            exact = exact_one_way_stratified(ref_model, init, ref_model.x_output_grid())
            rep.params("Exact reference", z_points=ref_model.z_n, eig_time_s=time.time() - t0)
            ref = RWPField(x_grid=np.asarray(ref_model.x_output_grid()), z_grid=np.asarray(ref_model.z_output_grid()),
                           freq_hz=src.freq_hz, field=exact[:, :ref_model.z_n_phys:ref_model.z_grid_scale])
            heights = [30.0, 120.0, 250.0]
            rows = []
            stats = {}
            for label, dx, mk in (("SSF dx=5", 5.0, create_rwp_ssf_model), ("SSF dx=250", 250.0, create_rwp_ssf_model),
                                  ("RA-SSF dx=25", 25.0, create_rwp_rassf_model),
                                  ("RA-SSF dx=250", 250.0, create_rwp_rassf_model)):
                model = mk(src, env, params, dx_m=dx, absorber='profile', abs_attenuation_db=0.0)
                f = _field(model, src.aperture(model.z_computational_grid()))
                p90s = []
                for z in heights:
                    med, p90, mx = _db_stats(ref, f, z, x_skip=0.2, floor_db=-30.0)
                    rows.append([label, z, med, p90, mx])
                    p90s.append(p90)
                stats[label] = max(p90s)
                if label == "RA-SSF dx=25":
                    rep.cuts("Exact one-way solution vs RA-SSF (25 m steps).",
                             {"exact": ref.field, "RA-SSF dx=25": f.field}, ref.x_grid, ref.z_grid, heights)
                if label == "SSF dx=5":
                    rep.cuts("Exact one-way solution vs SSF (5 m steps).",
                             {"exact": ref.field, "SSF dx=5": f.field}, ref.x_grid, ref.z_grid, heights)
            rep.table("dB error statistics against the exact one-way solution (x > 2 km, within 30 dB of the maximum)",
                      ["solver", "z (m)", "median", "90th percentile", "max"], rows)
            rep.check("SSF dx=5 m: worst 90th percentile error (error floor)", stats["SSF dx=5"], ">", 0.3, "dB")
            rep.check("SSF dx=250 m has the same floor as dx=5 m (ratio)",
                      stats["SSF dx=250"] / stats["SSF dx=5"], ">", 0.5)
            rep.check("RA-SSF dx=25 m: worst 90th percentile error", stats["RA-SSF dx=25"], "<", 0.05, "dB")
            rep.check("RA-SSF dx=250 m beats the SSF floor", stats["RA-SSF dx=250"], "<", stats["SSF dx=5"], "dB")
            rep.assert_all()

    def test_large_steps_in_evaporation_duct_at_10_ghz(self):
        with TestReport(self, """
            Evaporation duct of 20 m height at 10 GHz, antenna at 10 m with a 5 deg
            beam, 2 km range, 350 m hard-walled domain without absorbing layer (no
            energy returns from the top within the range). The near-surface
            curvature of the duct profile limits the step of any split scheme at
            10 GHz: against the exact one-way solution the RA-SSF with 50 m steps
            must agree to 0.2 dB in the median and 0.5 dB at the 90th percentile at
            5, 10 and 30 m height, its error must shrink from 500 m to 50 m steps,
            and the adaptive rank must stay small.
            """) as rep:
            src = RWPGaussSourceModel(freq_hz=10e9, height_m=10.0, beam_width_deg=5.0)
            env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
            params = RWPComputationalParams(max_range_m=2000.0, max_height_m=350.0, dx_m=500.0, dz_m=0.5)
            model = create_rwp_rassf_model(src, env, params, dx_m=50.0, abs_layer_scale=0.0)
            _report_setup(rep, src, env, params, model)
            init = src.aperture(model.z_computational_grid())
            t0 = time.time()
            exact = exact_one_way_stratified(model, init, model.x_output_grid())
            rep.params("Exact reference", z_points=model.z_n, eig_time_s=time.time() - t0)
            ref = RWPField(x_grid=np.asarray(model.x_output_grid()), z_grid=np.asarray(model.z_output_grid()),
                           freq_hz=src.freq_hz, field=exact[:, :model.z_n_phys:model.z_grid_scale])
            f = _field(model, init)
            coarse = create_rwp_rassf_model(src, env, params, abs_layer_scale=0.0)
            f_coarse = _field(coarse, init)
            worst = {}
            for label, fld in (("50 m", f), ("500 m", f_coarse)):
                worst[label] = max(_db_stats(ref, fld, z, x_skip=0.25, floor_db=-30.0)[1] for z in (5.0, 10.0, 30.0))
            rep.params("Worst 90th percentile error", **{f"dx_{k}": v for k, v in worst.items()})
            rep.check("range step", model.dx_m, "==", 50.0, "m")
            rep.check("adaptive rank", model.rank_max, "<=", 6)
            rep.check("error with 50 m steps is below the error with 500 m steps", worst["50 m"], "<", worst["500 m"])
            rep.field2d("Exact one-way solution and RA-SSF with 50 m steps.", {"exact": ref.field, "RA-SSF": f.field},
                        ref.x_grid, ref.z_grid)
            _compare_cuts(rep, ref, f, ("exact", "RA-SSF"), [5.0, 10.0, 30.0], {"median": 0.2, "p90": 0.5},
                          x_skip=0.25)
            rep.assert_all()


class TestRASSFvsTwoRay(unittest.TestCase):
    """Flat ground and homogeneous atmosphere: the two-ray model is exact."""

    def _case(self, pol, eps=None, sigma=0.0):
        ground = "PEC" if eps is None else f"eps = {eps}, sigma = {sigma} S/m"
        with TestReport(self, f"""
            Flat {ground} ground, homogeneous atmosphere, 300 MHz, {pol}-polarisation,
            antenna at 30 m with a 5 deg beam, 5 km range. The RA-SSF takes the
            output step (100 m) as its range step. The reduced field must match the
            two-ray model (times sqrt(x), both normalised per cut) beyond 30% of the
            range to 0.1 dB in the median and 0.5 dB at most at 5, 30, 100 and 200 m.
            """, title=f"two-ray {pol}-pol {ground}", suffix=f"{pol}_{'pec' if eps is None else f'eps{eps:g}'}") as rep:
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization=pol)
            env = TroposphereModel(slope=0, ground_material=None if eps is None else GroundMaterial(eps=eps, sigma=sigma))
            params = RWPComputationalParams(max_range_m=5000.0, max_height_m=300.0, dx_m=100.0, dz_m=1.0)
            model = create_rwp_rassf_model(src, env, params)
            _report_setup(rep, src, env, params, model)
            f = _run(src, env, params, 'rassf')
            two_ray = _two_ray_field(src, f.x_grid, f.z_grid, eps=eps, sigma=sigma)
            rep.field2d("Two-ray model and RA-SSF fields.", {"two-ray": two_ray.field, "RA-SSF": f.field},
                        f.x_grid, f.z_grid)
            _compare_cuts(rep, two_ray, f, ("two-ray", "RA-SSF"), [5.0, 30.0, 100.0, 200.0],
                          {"median": 0.1, "max": 0.5}, x_skip=0.3, floor_db=-30, normalize=True)
            rep.assert_all()

    def test_pec_h_pol(self):
        self._case('H')

    def test_pec_v_pol(self):
        self._case('V')

    def test_dielectric_v_pol_brewster(self):
        self._case('V', eps=3.0)

    def test_sea_water_v_pol(self):
        self._case('V', eps=80.0, sigma=4.0)


class TestRASSFvsRationalFD(unittest.TestCase):
    """RA-SSF against the Padé finite-difference propagator on RWP problems,
    with range steps equal to the output step."""

    def _compare(self, rep, src, env, params, heights, thresholds, x_skip=0.2, terrain=None):
        model = create_rwp_rassf_model(src, env, params)
        _report_setup(rep, src, env, params, model)
        t0 = time.time()
        fd = _run(src, env, params, 'pade')
        t_fd = time.time() - t0
        t0 = time.time()
        f = _run(src, env, params, 'rassf')
        t_ra = time.time() - t0
        rep.params("Wall-clock time (including compilation)", pade_s=t_fd, rassf_s=t_ra)
        rep.check("same output shape", fd.field.shape == f.field.shape, "==", True)
        rep.field2d("Padé FD and RA-SSF fields.", {"Padé FD": fd.field, "RA-SSF": f.field}, fd.x_grid, fd.z_grid,
                    terrain=None if terrain is None else np.asarray(terrain(jnp.asarray(fd.x_grid))))
        _compare_cuts(rep, fd, f, ("Padé FD", "RA-SSF"), heights, thresholds, x_skip=x_skip)
        return fd, f, model

    def test_standard_atmosphere(self):
        with TestReport(self, """
            Standard atmosphere over PEC ground, 300 MHz, H-polarisation, 20 km
            range with 1 km output step, which the RA-SSF takes as its range step
            (20 steps in total). Beyond 20% of the range the RA-SSF and the Padé
            (7, 8) solution must agree to 0.5 dB in the median and 2 dB at the 90th
            percentile.
            """) as rep:
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0)
            env = TroposphereModel()
            params = RWPComputationalParams(max_range_m=20000.0, max_height_m=300.0, dx_m=1000.0, dz_m=1.0)
            _, _, model = self._compare(rep, src, env, params, [10.0, 30.0, 100.0], {"median": 0.5, "p90": 2.0})
            rep.check("range step", model.dx_m, "==", 1000.0, "m")
            rep.assert_all()

    def test_evaporation_duct(self):
        with TestReport(self, """
            Evaporation duct of 20 m height at 3 GHz, antenna at 10 m with a 3 deg
            beam, 20 km range with 1 km steps. The RA-SSF is compared with the Padé
            FD solution near the surface; the trapped field at 10 m and 20 km must
            exceed the standard-atmosphere level by more than 10 dB.
            """) as rep:
            src = RWPGaussSourceModel(freq_hz=3e9, height_m=10.0, beam_width_deg=3.0)
            env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
            params = RWPComputationalParams(max_range_m=20000.0, max_height_m=150.0, dx_m=1000.0, dz_m=0.5)
            _, f, model = self._compare(rep, src, env, params, [5.0, 10.0, 30.0], {"median": 0.5, "p90": 2.0})
            f_std = _run(src, TroposphereModel(), params, 'rassf')
            iz = int(np.argmin(np.abs(f.z_grid - 10.0)))
            gain = 20 * np.log10(np.abs(f.field[-1, iz]) / np.abs(f_std.field[-1, iz]))
            rep.check("duct gain over standard atmosphere at z = 10 m, x = 20 km", float(gain), ">", 10.0, "dB")
            rep.check("range step", model.dx_m, "==", 1000.0, "m")
            rep.assert_all()

    def test_surface_based_duct(self):
        with TestReport(self, """
            Surface-based duct (-20 N-units between 100 m and 200 m), 1 GHz, antenna
            at 50 m with a 4 deg beam, 20 km range with 1 km steps. The RA-SSF is
            compared with the Padé FD solution inside and above the duct.
            """) as rep:
            z_grid = jnp.array([0.0, 100.0, 200.0, 300.0])
            env = TroposphereModel(N_profile=PiecewiseLinearNProfileModel(
                z_grid_m=z_grid, N_vals=jnp.array([0.0, -20.0, -20.0, 0.0])))
            src = RWPGaussSourceModel(freq_hz=1e9, height_m=50.0, beam_width_deg=4.0)
            params = RWPComputationalParams(max_range_m=20000.0, max_height_m=400.0, dx_m=1000.0, dz_m=1.0)
            self._compare(rep, src, env, params, [20.0, 50.0, 150.0], {"median": 0.5, "p90": 2.0})
            rep.assert_all()

    def test_terrain_hill(self):
        with TestReport(self, """
            Triangular hill of 80 m height between 3 km and 7 km, 300 MHz, antenna at
            50 m with an 8 deg beam, 50 m output step. With a terrain the RA-SSF
            caps its range step by the SSF paraxial rule so the staircase is resolved
            as in the other solvers. The field inside the hill must be zero and in the
            diffraction zone behind the hill (last quarter of the range) the RA-SSF
            and Padé solutions must agree to 1.5 dB in the median and 3 dB at most.
            """) as rep:
            x_t = jnp.array([0.0, 3000.0, 5000.0, 7000.0, 10000.0])
            terrain = PiecewiseLinearTerrainModel(x_grid_m=x_t, height=jnp.array([0.0, 0.0, 80.0, 0.0, 0.0]))
            env = TroposphereModel(terrain=terrain)
            src = RWPGaussSourceModel(freq_hz=300e6, height_m=50.0, beam_width_deg=8.0)
            params = RWPComputationalParams(max_range_m=10000.0, max_height_m=400.0, dx_m=50.0, dz_m=1.0)
            fd, f, _ = self._compare(rep, src, env, params, [30.0, 100.0, 200.0], {"median": 1.5, "max": 3.0},
                                     x_skip=0.75, terrain=terrain)
            ix = int(np.argmin(np.abs(f.x_grid - 5000.0)))
            rep.check("max |u| inside the hill (RA-SSF, x = 5 km, z < 70 m)", float(np.max(np.abs(f.field[ix, :70]))),
                      "==", 0.0)
            rep.assert_all()

    def test_impedance_ground(self):
        params = RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=500.0, dz_m=1.0)
        for pol, eps, sigma in (('V', 3.0, 0.0), ('H', 3.0, 0.0), ('V', 80.0, 4.0)):
            with self.subTest(pol=pol, eps=eps, sigma=sigma):
                with TestReport(self, f"""
                    Finite-permittivity ground (eps = {eps}, sigma = {sigma} S/m),
                    {pol}-polarisation, standard atmosphere, 300 MHz, antenna at 30 m
                    with a 5 deg beam, 500 m steps. The RA-SSF propagates the DMFT sine
                    spectrum and the two discrete surface-wave modes with the low-rank
                    symbol factors; it must agree with the Padé FD solver (nonlocal
                    boundary condition) to 0.2 dB in the median and 1 dB at most.
                    """, title=f"impedance ground {pol}-pol eps={eps:g} sigma={sigma:g}",
                                suffix=f"{pol}_eps{eps:g}") as rep:
                    src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization=pol)
                    env = TroposphereModel(ground_material=GroundMaterial(eps=eps, sigma=sigma))
                    self._compare(rep, src, env, params, [5.0, 30.0, 100.0], {"median": 0.2, "max": 1.0})
                    rep.assert_all()

    def test_vegetation_slab_attenuation_matches_theory(self):
        with TestReport(self, """
            A vegetation slab (eps = 1.05, sigma = 2e-4 S/m) filling the whole
            computational domain turns the atmosphere into a homogeneous lossy
            dielectric; the slab is a range-dependent profile so the low-rank
            factors are built per distinct profile. The ratio of the field with and
            without the slab at 500 m range and 100 m height must follow the
            plane-wave attenuation -8.686 k0 Im(sqrt(eps)) x to within 2%.
            """) as rep:
            f = 300e6
            k0 = 2 * fm.pi * f / 3e8
            x = jnp.linspace(0, 1000, 11)
            terrain = PiecewiseLinearTerrainModel(x, jnp.zeros_like(x))
            material = GroundMaterial(eps=1.05, sigma=2e-4)
            src = RWPGaussSourceModel(freq_hz=f, height_m=100.0, beam_width_deg=5.0, polarization='V')
            params = RWPComputationalParams(max_range_m=1000, max_height_m=200, dx_m=100.0, dz_m=2.0)
            fields = {}
            for tag, veg in (("air", None),
                             ("slab", VegetationLayerModel(
                                 thickness=PiecewiseLinearTerrainModel(x, jnp.full_like(x, 300.0)),
                                 material=material))):
                env = TroposphereModel(terrain=terrain, ground_material=GroundMaterial(eps=15, sigma=0.005),
                                       vegetation=veg)
                fields[tag] = _run(src, env, params, 'rassf')
            _report_setup(rep, src, env, params, create_rwp_rassf_model(src, env, params))
            u_air, u_slab = fields["air"], fields["slab"]
            iz = int(np.argmin(np.abs(u_air.z_grid - 100.0)))
            ix = int(np.argmin(np.abs(u_air.x_grid - 500.0)))
            ratio_db = 20 * np.log10(np.abs(u_slab.field[ix, iz]) / np.abs(u_air.field[ix, iz]))
            theory_db = -8.686 * k0 * cmath.sqrt(material.complex_permittivity(f)).imag * u_air.x_grid[ix]
            rep.params("At x = 500 m, z = 100 m", rassf_ratio_db=float(ratio_db), theory_db=float(theory_db))
            rep.check("relative error of the attenuation", abs(ratio_db - theory_db) / abs(theory_db), "<", 0.02)
            rep.assert_all()


class TestRASSFFrontEnd(unittest.TestCase):

    def test_dispatch_and_lower_bc(self):
        with TestReport(self, """
            create_rwp_model must dispatch to the RA-SSF class when
            approx_method = 'rassf' and select the lower boundary from the
            polarisation and ground material exactly like for the SSF.
            """) as rep:
            params = RWPComputationalParams(max_range_m=1000.0, max_height_m=200.0, dx_m=50.0, dz_m=1.0)
            params.approx_method = 'rassf'
            rows = []
            for pol, gm, bc in (('H', None, 'dirichlet'), ('V', None, 'neumann'),
                                ('V', GroundMaterial(eps=15.0, sigma=0.005), 'impedance')):
                src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization=pol)
                model = create_rwp_model(src, TroposphereModel(ground_material=gm), params)
                rows.append([pol, "PEC" if gm is None else "eps=15", bc, model.lower_bc, type(model).__name__])
                rep.check(f"class for {pol}-pol", isinstance(model, RankAdaptiveSSFPropagator), "==", True)
                rep.check(f"lower_bc for {pol}-pol", model.lower_bc, "==", bc)
            rep.table("Dispatch", ["polarisation", "ground", "expected bc", "selected bc", "class"], rows)
            field = rwp_forward_task(RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0),
                                     TroposphereModel(), params)
            rep.check("rwp_forward_task output rows", field.field.shape[0], "==", 21)
            rep.assert_all()


if __name__ == '__main__':
    unittest.main()
