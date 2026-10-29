# Nonuniform vertical grids for the terrain-following propagator

This note collects the work on nonuniform vertical grids in
`TerrainFollowingHelmholtzPropagator` (`pywaveprop/helmholtz_tf_jax.py`,
branch `neural-propagator`):

- why a stretching costs nothing in the scheme;
- the three places it is used (absorbing layer, region above the distorted
  layer, inside the distorted layer);
- the discrete NLBC on a nonuniform grid;
- verification, cost, and the pitfalls met on the way.

The parts also appear in `nlbc_nonuniform.md` (full NLBC derivation) and in
§8–§9 of `grid_optimizer.md`. This file is the self-contained summary.

**Status (2026-10-27).** The absorbing-layer stretching and `upper_stretch`
are committed (`1904022`, `ab72502`). `interior_stretch` and its test are
implemented but not committed. Nothing of this is in the paper except the
stretched absorbing layer (steep-terrain cost paragraph).

---

## 1. Principle: a stretching is a metric

A computational grid η' = jΔη' that is uniform and mapped by a smooth monotone
η = Z(η') is a change of the vertical coordinate. For the orthogonal map
(ξ, η) → (x, z), the composite map (ξ, η') is still orthogonal, because each
coordinate is reparametrized separately:

    h_ξ unchanged,   h_η → h_η s,   s = dZ/dη' .

The terrain-following scheme is written for an arbitrary orthogonal map. Each
substep is

    b rt δ(rt⁻¹ δ ψ) + β² m (1 + b het) ψ   (Numerov weights)

with r = h_η/h_ξ, m = r², rt = ρr and V = (r^½)_ξξ / r^½. The stretching
therefore needs no new discretization: the tables take h_η s in place of
h_η. In detail:

- r → r s, m → m s², rt → rt s, and rt_half → rt_half s at the half nodes;
- V is unchanged, since s does not depend on ξ;
- het = (k² h_ξ² − V)/β² − 1 is unchanged; k² is evaluated at the physical
  point of the mapped level;
- the impedance closure uses h_η on the ground, so g = α h_η(0) s(0) Δη';
- the identity region (above the transition height, h_ξ = 1) has
  m = s², rt = ρ s, V = 0 and physical height z = z_off + Z(η').

In the identity region the weights of the second difference become
w^l_j = s_j/s_{j−½} and w^r_j = s_j/s_{j+½}.

**Exactness check.** A constant s is mathematically a uniform grid with step
sΔη'. With s = 23/19 and Δη' chosen so that the physical step equals that of
a uniform run, the fields agree to 2.5·10⁻¹¹ at the common rows, and the
output geometry to 10⁻¹⁰ m. With s ≡ 1 through the stretching code the
result matches the uniform path to 10⁻¹¹.

---

## 2. Where the grid can be coarsened

The vertical step is chosen by the grid optimizer (`grid_optimizer.md` §5).
The physical vertical wavenumber k_z on the local step must be resolved to the
requested precision. Three regions have room:

| region | what limits Δη there | what a uniform grid uses |
|---|---|---|
| absorbing layer (above the physical top) | only that the waves meeting the damping are resolved: k_z s Δη ≤ 1 | the full-accuracy step |
| identity region above the distorted layer | the flat step Δz (h_ξ = r = 1) | the step of the most distorted point |
| distorted layer | the local physical step h_η Δη at each point | one global bound h_ξ,max · r_max |

The rate functions are, respectively:
- `layer_stretch` (closed form, quintic smoothstep);
- `layer_stretch` with the start above the flux rows;
- `TabulatedStretch` built from the metric profile (§5).

---

## 3. Stretched absorbing layer (`absorb_stretch=True`, default)

Used when the NLBC cannot be applied, i.e. with one reference wavenumber per
class of columns (`beta_mode='column'`, Dirichlet top over `absorb_m` metres
damped by 0.02 i k² ramp²).

- **Rate.** s rises from 1 to s_max = max(1, 1/(Δη k_z,max)) over the lower
  third of the layer (quintic smoothstep, C²) and stays at s_max above.
- **Damping.** It is defined in physical height, so it does not change with
  the stretching.
- **Glen Etive.** s_max = 4.8 and 8.0 at 10⁻² and 10⁻³. The 300 m layer takes
  456 and 471 rows instead of 1800 and 3000.

| run | ε | z_n | create | first compute | march |
|---|---|---|---|---|---|
| global β, NLBC | 10⁻² | 13541 | 10.6 s | 6.8 s | 5.9 s |
| global β, stretched layer | 10⁻² | 13997 | 0.6 s | 7.0 s | 5.7 s |
| β per class, uniform layer | 10⁻² | 15341 | 2.6 s | 4.8 s | 4.3 s |
| β per class, stretched layer | 10⁻² | 13997 | 2.0 s | 4.7 s | 3.8 s |
| global β, NLBC | 10⁻³ | 24824 | 16.5 s | 12.3 s | 12.0 s |
| global β, stretched layer | 10⁻³ | 25297 | 0.8 s | 12.3 s | 11.3 s |
| β per class, uniform layer | 10⁻³ | 25568 | 3.4 s | 7.5 s | 6.9 s |
| β per class, stretched layer | 10⁻³ | 23038 | 2.3 s | 7.2 s | 6.3 s |

- **Accuracy.** Stretched against uniform layer: 0.000 dB median in all
  bands, at most 0.004 dB on the cuts. Against the NLBC run the difference
  is 0.002–0.007 dB median, the same as for the uniform layer.
- **Speed.** The march is about 10 % faster. With a global β the stretched
  layer is even cheaper end to end than the NLBC (7.6 s against 17.4 s at
  10⁻²), because it needs no kernel.
- **In the paper.** Steep-terrain cost paragraph: 456 rows instead of 1800;
  7 and 10 s; 0.007 dB median, 0.02 dB max against a single β.

Data: `papers/terrain_transform/figures/beta_column.json`,
`scripts/tap_beta_column.py`.

---

## 4. Discrete NLBC on a nonuniform grid

Derivation in full: `nlbc_nonuniform.md`. Summary:

**What depends on the grid.** The NLBC construction consists of:
- the Z-transform in the range-step index;
- the matrix function of M(t) = A(t)⁻¹B(t) over the Padé substeps;
- the inverse transform to the convolution kernel;
- the scalar R(σ) = u_{N−1}/u_N, the ratio of the decaying solution of the
  exterior problem L u = σ u.

Only R(σ) depends on the vertical grid. The common factor
P = τ²(I + αD_w)diag(m) cancels in every substep, so the range scheme is the
plain Padé chain in L = diag(het) + P⁻¹D_w.

**Exterior recurrence** (rows j ≥ N; g_j = (βΔη)² m_j (σ − het_j)):

    A⁻_j u_{j−1} + C_j u_j + A⁺_j u_{j+1} = 0,
    A⁻_j = w^l_j (1 − α g_{j−1}),   A⁺_j = w^r_j (1 − α g_{j+1}),
    C_j  = −(w^l_j + w^r_j)(1 − α g_j) − g_j .

For s ≡ 1 these are the coefficients of the existing uniform NLBC.

**Minimal solution.** The ratio r_j = u_j/u_{j−1} of the decaying solution
satisfies r_j = −A⁻_j/(C_j + A⁺_j r_{j+1}), a continued fraction. By
Pincherle's theorem it converges to the minimal solution, and the backward
direction is the stable one (as in Miller's algorithm). It is started at the
first node J of the uniform tail (s = s_top) from the exact uniform ratio
with Δz_eff = s_top Δη:
- the quadratic root `sqr_eq` for constant het;
- `bessel_ratio_4th_order` for a linear refractivity profile.

R(σ) = 1/r_N. The recursion is exact, with no truncation, and costs J − N
operations per σ. When the top lies in the tail (J = N) it is the existing
NLBC with step s_top Δη.

**Conditions.**
- **Range invariance.** The exterior must be the same at every step, so this
  works with one β only. `beta_mode='column'` keeps the absorbing layer.
- **Kernel convergence.** β Δz_eff < 2, hence s ≤ 2/(k̄ Δη).
- **Transparency.** The condition is exactly transparent for the discrete
  stretched medium. It cannot remove reflections caused by the stretching,
  so s must be smooth.

**Verification.** A taller domain on the same grid must give the same field.

| case | derived NLBC vs taller domain | naive (uniform NLBC, local step) |
|---|---|---|
| Glen Etive, s_top = 1.63, top inside the ramp | 2.6·10⁻⁵ dB max | 1.3·10⁻⁴ dB |
| beam 15° through the top, s_top = 1.17 | 4.2·10⁻⁶ dB max | 3.3·10⁻³ dB |
| same beam, s_top = 4 | 4.2·10⁻⁶ dB max | 1.9·10⁻² dB |
| interior stretch, 200 m hill, beam at the top (§5) | 7.6·10⁻⁷ relative | — |
| interior stretch, 60 m hill | 6.5·10⁻¹³ relative | — |

The recursion is consistent: starting it 50 or 400 rows deeper changes R(σ)
by ≤ 3·10⁻¹⁵ (constant het) and ≤ 8·10⁻¹² (linear het). Data:
`scripts/tap_nlbc_stretch.py`, `figures/nlbc_stretch.json`.

---

## 5. Region above the distorted layer (`upper_stretch=True`)

A first, simpler use of §4, now superseded by §6.
- **Rate.** s rises from 1, starting just above the flux rows
  (η_lab + 4λ + 6Δη), to s_top over 20λ (physical), with a quintic
  smoothstep.
- **Top rate.** s_top = min(h_ξ,max r_max, 2/(k̄ Δη)): the factor by which the
  global rule shrank Δη, which is not needed where h_ξ = r = 1.
- **Glen Etive** (s_top = 1.63, 10⁻²): 13541 → 11644 rows (−14 %), 18 → 15 s;
  the field changes by 0.010 dB max, 8·10⁻¹¹ dB median.
- **Highlands** (s_top = 1.52):

| | 10⁻² | 10⁻³ |
|---|---|---|
| rows | 49888 → 43353 | 89144 → 77464 |
| time | 298 → 268 s | 787 → 444 s |
| path loss change | ≤ 9·10⁻⁵ dB | ≤ 3·10⁻⁵ dB |

- **Limit.** It cannot be combined with an absorbing layer or with per-class
  β. The parameters `upper_stretch_start_m`, `_ramp_m` and `_s_top` exist
  mainly to put the top inside the ramp for the NLBC test.

---

## 6. Inside the distorted layer (`interior_stretch=True`)

### 6.1 Why the uniform rule is conservative

**Local condition.** The local scheme sees the vertical wavenumber k_z h_ξ on
the step rΔη, and the product is k_z h_η Δη. With Δz the flat optimizer's
step for k_z,max h_ξ,min (the same linear rescaling as the existing rule):

    h_η(ξ, η) Δη ≤ h_ξ,min Δz .

The uniform rule replaces max h_η by h_ξ,max · r_max. The two maxima are at
different places: h_ξ is extreme on the ground, r at about mid-height of the
transition layer.

Metric profiles (z_ref = mean terrain, as in the solver):

| case | h_ξ range | r_max | h_ξ,max r_max | max h_η | rows saved by the pointwise bound |
|---|---|---|---|---|---|
| Highlands (L = 1000 m) | [0.914, 1.209] | 1.254 | 1.516 | 1.27 | 16 % |
| Glen Etive (L = 1400 m) | [0.785, 1.208] | 1.346 | 1.626 | ≈ 1.39 | 15 % |
| 200 m test hill | [0.791, 1.262] | 1.197 | 1.511 | 1.279 | 15 % |

Q(η) = max_ξ h_η(ξ, η) on the Highlands path:

| η (m) | 0 | 62 | 188 | 312 | 438 | 562 | 688 | 812 | 938 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q | 1.23 | 1.19 | 1.21 | 1.25 | 1.27 | 1.23 | 1.16 | 1.07 | 1.01 | 1.00 |

**A stretching that follows Q.** s = Q_max/Q(η) keeps the worst physical
step max_ξ h_η s Δη' equal on every level. Inside the layer it saves another
7–11 % of the rows, depending on the shape of Q, and above the layer it
gives s = Q_max, the same physical step as `upper_stretch`.

**A caveat on the "linear" convention.** With e₄ ∝ (k_z Δz)⁴ and the
het-error prefactor k_z² h_ξ², a stricter bound would carry an extra factor
(h_ξ,min/h_ξ)^½. Measured per physical metre of range, however, the h_ξ²
cancels. The rule keeps the convention of the existing optimizer, and
accuracy is checked empirically (§6.4).

### 6.2 Construction

- **Profile.** Q(η) = max_ξ max(h_η, h_η,below) on levels spaced
  min(…, smooth/8) up to η_lab. The levels do not depend on the domain
  height; see pitfall 3 in §7.
- **Envelope.** Q̃ is a smooth upper envelope of Q: a max filter and a
  Gaussian of width max(4λ, L/40) on log Q, shifted so that Q̃ ≥ Q, then
  tapered to exactly 1 over a few widths above the layer. s is then exactly
  s_top for η ≥ η_tail.
- **Step.** Δη' = h_ξ,min Δz / Q̃_max, rounded down to align output rows
  (dz_out / ceil).
- **Recovering the rounding.** The stretching gives up the alignment anyway,
  so the rounding loss is returned through the scale
  s_scale = h_ξ,min Δz/(Q̃_max Δη') ≥ 1:

      s(η) = min(s_scale · Q̃_max/Q̃(η), s_cap) .

  For a 0.5 m output step on the hill this is 13 %.
- **Cap.** s_cap = 2/(k̄ Δη') with the NLBC, ∞ with an absorbing layer.
- **Absorbing layer.** s rises further, smoothly, from s_top to
  1/(Δη' k_z,max) over the lower third of the layer. The layer must start
  above η_tail.
- **Per-class β.** Class c allows the physical step D_c = Δη_c q_c, with
  q_c = max h_η over its columns and Q_c(η) its profile (max-filtered over
  the window a step reads). The requirement is
  s(η) Δη' ≤ min_c D_c/Q_c(η), equivalently a single profile

      q_eff(η) = max_c Q_c(η) D_min/D_c

  fed to the same construction, with Δη' = min_c Δη_c. A global Q profile
  with per-class Δη_c would *not* be conservative: it could exceed a
  class's bound by Q_max/q_c.
- **Table.** `TabulatedStretch` tabulates η'(η) = ∫dη/s by Simpson on a fine
  η grid (step min(Δη', smooth/8)), interpolates it by cubic Hermite (slope
  1/s), and inverts it by 4 Newton steps. s(η') is evaluated exactly as
  s_fun(Z(η')). Accuracy: inverse error 3·10⁻¹³; construction takes under
  1 ms (an earlier RK4 loop in Python took seconds).
- **Rows below the identity region.** n_lab = ceil(η'(η_lab)/Δη') + 2. The
  tables and the output geometry evaluate the map metrics at Z(η') of the
  half levels.
- **Output.** Rows are every z_step computational rows, so their physical
  spacing follows s.

API: `create(..., interior_stretch=True, interior_stretch_smooth_m=None)`,
front end `params.terrain_interior_stretch = True`. It replaces
`upper_stretch` and is not available for the strip map.

### 6.3 Results: test hills

3 km path, 300 MHz, V, wet ground, beam 6° tilted 2°, aperture 20°,
transition 400 m, smoothing 50 m. Two Gaussian hills of height H and H/2.
Errors are against the uniform grid at 10⁻⁴, as |u − u_ref|/max|u_ref|
along the output trajectories (§7, pitfall 1).

| H | ε | uniform: rows, time, max error | interior: rows, time, max error |
|---|---|---|---|
| 200 m | 10⁻² | 5775 (rule), 13.3 s, 2.8·10⁻³ | 4274, 10.0 s, 3.5·10⁻³ |
| 200 m | 10⁻³ | 21767 (rule), 36.7 s, 7.9·10⁻⁵ | 16107, 27.7 s, 8.1·10⁻⁵ |
| 60 m | 10⁻³ | 7337, 8.4 s, 3.6·10⁻⁴ | 6705, 9.1 s, 3.6·10⁻⁴ |

The 60 m rows predate the rounding recovery. The medians are unchanged:
3·10⁻⁷ and 6·10⁻⁹ for the 200 m hill.

Per-class β with a 150 m absorbing layer, 200 m hill:

| ε | uniform: time, max error | interior: rows, time, max error |
|---|---|---|
| 10⁻² | 5.9 s, 5.4·10⁻³ | 4517, 5.6 s, 6.1·10⁻³ |
| 10⁻³ | 7.2 s, 5.3·10⁻⁴ | 7737, 5.9 s, 5.9·10⁻⁴ |

### 6.4 Results: Highlands path

28 km, 300 MHz, H, PEC, GFS 2026-10-09 12 UTC, aperture 27.3°,
L = 1000 m. Each run was alone on the CPU (20 cores, WSL2).

| run | rows 10⁻² / 10⁻³ | time 10⁻² / 10⁻³ |
|---|---|---|
| one β, uniform | 49888 / 89144 | 298 / 787 s |
| one β, upper stretch (§5) | 43353 / 77464 | 268 / 444 s |
| one β, interior stretch | 36400 / 65194 | 233 / 378 s |
| β per class + absorbing layer, uniform | — | 139 / 367 s |
| β per class + absorbing layer, interior stretch | 35500 / 69087 | 111 / 279 s |
| staircase (package) | — | 50 / 80 s |

Accuracy at 10⁻³ against the uniform one-β run was measured with dense
output (0.25 m) and the complex reference interpolated along the same
trajectories:

| run | along trajectories: median / p99.9 / max (relative) | 10 m above ground (dB): median / p90 / max | 50 m above ground (dB): median / p90 / max |
|---|---|---|---|
| one β, interior | 1.8·10⁻⁴ / 5.9·10⁻³ / 8.5·10⁻³ | 0.058 / 0.16 / 0.52 | 0.060 / 0.24 / 1.46 |
| per class, interior | 1.6·10⁻⁴ / 5.1·10⁻³ / 7.4·10⁻³ | 0.048 / 0.14 / 0.46 | 0.053 / 0.22 / 1.02 |

These are below the 0.12 dB (median, 10 m) that separates the uniform
grid's own 10⁻² and 10⁻³ solutions. The terrain-following vs staircase
difference is unchanged: 1.0 dB at 10 m, 0.48 dB at 50 m.

**Cost ratio against the staircase.**

| | 10⁻² | 10⁻³ |
|---|---|---|
| one β, uniform | 5.9× | 9.8× |
| per class + interior stretch | 2.2× | 3.5× |

The paper still quotes the uniform/per-class numbers ("three to ten times
cheaper").

Data: `figures/highlands_cost.json` (keys `tf*`, `tfi*`, `tfcol*`,
`tfcoli*`, `st*`; `dense_*` for the 0.25 m runs);
`scripts/tap_highlands_cost.py VARIANT PRECISION OUT_DIR`
(env `HL_DZ_OUT`, `HL_SAVE_FIELD`).

---

## 7. Nonuniform range step (`range_steps='class'`)

**Construction.** Needs `beta_mode='column'`.
- Each output interval of `dx_output_m` gets a class from the columns it
  reads (h_ξ range, max r).
- It is crossed in n_c steps of dx_out/n_c. β_c, the Padé coefficients and
  n_c come from `class_grid_steps`, which picks one Δη for all classes
  jointly.
- The march (`_compute_class_steps`) nests a loop of n_c steps inside the
  loop over output intervals.

**Result (200 m two-hill test, 300 MHz, 10⁻³).**

| aperture | common step | class step | class − common, max |
|---|---|---|---|
| 20°, slope widening off | 8.4 s | 5.0 s (mean step 6.0 m) | 1.6·10⁻² |
| 30° (covers 20° + slopes), widening off | 15.8 s | 9.3 s (mean 4.4 m) | 6.6·10⁻⁴ |
| 100 m hill, widening on (about 30°) | 63.1 s | 52.7 s | 6.5·10⁻⁵ |

- **The large error was the aperture, not the march.** Rays reflected from
  9.7° slopes leave at up to 20° + 2·9.7°. The long class steps are sized at
  the error limit and mishandle that content; the common step hid it through
  spare margin. With an aperture that covers the slopes, the class steps are
  as accurate as the common one.
- **Ruled out on the way:** a march bug (forcing the common step through
  the class march gives 6.8·10⁻⁴), the per-class Padé budget (≤ 3·10⁻⁴),
  the potential V (about 10⁻⁴), β at the interval midpoint, and
  amplification (|R| ≤ 1 for every tested (β, Δ) pair).
- **The real gain is small.** With the slope widening on (the default), the
  wider spectrum shortens the steps of every class: about 1.2× on the 100 m
  hill, against 1.2–1.8× from the cost estimate for the Highlands and
  Glen Etive.
- **The trap.** With `terrain_slope_aperture=False` the class steps run
  fast and wrong without a warning. A guard is still to be added.

## 8. Local NLBC for per-class β: complex-step layer (`upper_bc='complex_layer'`)

**Why the convolution NLBC does not carry over.**
- With β_k, the Padé chain and the range step changing from step to step,
  the exterior is no longer range invariant.
- The Z-transform no longer decouples it, and the exact condition would
  need a kernel K(n, n′) that depends on the class sequence. Its memory is
  x_n² × p².

**Why a local condition does.**
- Above the transition height, every class's substep reduces in the
  exterior to (A − λ)w = 0, with λ = β_k²(1 + σ).
- A = k² + Δη⁻²[(I + αD_w)diag(m)]⁻¹D_w is the Numerov operator without β.
- So the exact NLBC ratio R is a function of λ alone, shared by all classes,
  substeps and step lengths. Any boundary condition that is local in range
  and reproduces R(λ) holds for all of them at once.

**Construction.**
- The backward continued fraction of §4 is itself a few rows of a grid.
- M rows with complex node and half-node stretches s_j, s_{j+1/2}, ending
  in a Dirichlet node, give a rational approximation of R(λ). This is the
  optimal-grid / discrete-PML idea of Druskin and coworkers.
- The steps are fitted by least squares on the reflection coefficient of
  the discrete plane waves μʲ (`complex_layer_steps`, `layer_reflection`):
  - propagating k_zΔz from `layer_min_angle_deg` (default 0.5°) to the
    aperture;
  - evanescent κΔz up to 1.5.
- M grows by two rows until the reflection is ≤ `layer_reflection`
  (default min(precision, 10⁻²)).
- The reflection must be measured at the last interior row: its
  coefficients use the first layer half-step s_{1/2}.
- k² is frozen at its top value inside the layer.
- Grazing waves (k_z → 0) cannot be absorbed by any finite layer; the
  minimum angle sets the band, and the rows grow like log(θ_max/θ_min).

**Reflection and cost of the fit** (Δz = 0.125 m, 100 MHz, 0.5–30°): 8 rows
for 5·10⁻³ (3 s), 12 for 2.4·10⁻⁴ (8 s), 16 for 1.2·10⁻⁵ (13 s), 34 for
1.1·10⁻⁶ (about 200 s). The fit is cached per band. A JAX Jacobian would
make the tight targets fast.

**Stability.** The step matrix on interior + layer (dense eigenvalues, order
(7, 8), β/k 0.9–1.05, Δx 2–12 m) has all |R| < 1. The layer is
dissipative, not amplifying.

**Exactness test.**
- Setup: 120 m hill, 100 MHz, per-class β, homogeneous atmosphere. A beam at
  8° leaves through the top at 650 m, at 92 % of its peak. The comparison
  is with a 1400 m domain on the same grid, by magnitude below 630 m.
- Magnitudes are compared because β differs slightly between tops when
  there is a gradient.

| upper boundary | rows | max ‖u| − |u_tall‖ / max |
|---|---|---|
| complex layer, 12 rows (target 10⁻³) | 2517 | 5.5·10⁻⁵ |
| same, class range steps | 2517 | 5.4·10⁻⁵ |
| complex layer, 34 rows (target 10⁻⁶) | 2528 | 1.6·10⁻⁷ |
| same, M gradient 0.118 /m, 16 / 34 rows | 2519 / 2528 | 9.9·10⁻⁶ / 2.1·10⁻⁷ |
| absorbing layer 150 m (198 rows) | 2610 | 4.5·10⁻⁴ |
| same, class range steps | 2610 | 3.8·10⁻⁴ |

The error follows the target, so the layer is the only limit, and it holds
with class range steps.

**Highlands, 10⁻²** (8 rows, reflection ≤ 5·10⁻³, replacing 1100 rows):

| run | absorbing layer | complex layer | path loss vs tf 10⁻³ at 10 m (median / max) |
|---|---|---|---|
| tfcol | 139 s | 149 s | 0.125 / 0.44 → 0.133 / 0.70 dB |
| tfcoli | 111 s | 118 s | 0.144 / 4.24 → 0.154 / 4.24 dB |

On this path the absorbing layer was only about 2 % of the rows, so the
time is unchanged within noise. The gain is in exactness and simplicity:
- no 300 m layer to size;
- the interior stretch keeps its tail up to the top;
- the error at the top is set by one parameter.

Rows saved matter most for low domains or high frequencies, where a
physical absorbing layer is many wavelengths thick.

## 9. Pitfalls met (and how they were resolved)

1. **Comparing runs on different vertical grids.**
   - **Cause.** On uniform grids the output rows of every run sit at
     η = j·dz_out, i.e. at identical physical points, so comparisons involve
     no interpolation. Stretched runs do not share these points.
   - **Symptom.** Interpolating |u| linearly between 1–2 m rows at λ = 1 m
     gave spurious "errors" of 1–14 %, and dB-level differences on the
     Highlands cuts (p90 0.7–0.9 dB), which do not shrink with precision.
   - **Second source.** `RWPField.on_regular_grid` interpolates along
     computational rows in x first (output step 25–50 m), which on slopes
     also dominates the maximum error.
   - **Fix.** Dense reference output (0.1–0.25 m). Interpolate the *complex*
     reference along the same trajectory (one carrier phase per column), then
     compare magnitudes. The carrier differs between runs with different β,
     so complex fields of two runs must not be compared directly.
2. **Domain top raised by the front end.** The front end uses
   max(max_height, 1.1 × N-profile height), so a long N profile makes
   different requested tops identical. Keep the profile below the top in
   NLBC tests.
3. **Grids that depend on the top.**
   - **Spectrum.** With a refractivity gradient, the k range depends on the
     domain height, so β and Δz differ slightly between tops. Once the
     rounding is recovered (§6.2) this changes s_scale and hence the grid.
     NLBC tests therefore use a homogeneous atmosphere.
   - **Provisional sampling.** Δη_p = min(L/200, Z/400) depends on the top
     below Z = 2L, which changes h_ξ,min at the 10⁻⁶ level. Tests use tops
     above 2L.
   - **Stretch profile.** It is sampled on its own levels, independent of
     Z, and its table ends at η_tail, not at the top.
4. **The per-class profile.** As noted in §6.2, a global Q with the
   per-class Δη is not conservative; q_eff fixes it.

---

## 10. Open questions

- **Time vs rows.** At 10⁻³ with one β, the stretched grids save more time
  than rows: −52 % time for −27 % rows (interior), −44 % for −13 %
  (upper). The cause has not been isolated. Candidates are the NLBC kernel
  computation or the flux rows n_flux.
- **Range step.** The remaining gap to the staircase is mostly in the range
  step: Δξ = 3.1 m (one β) or 4.5 / 2.9 m (per class) against 12.5 / 10 m.
  The h_ξ spread widens the interval the Padé approximant must cover. A
  finer t grid between 10 and 20 in the optimizer table would help. The
  nonuniform range step (§7) works with the complex layer (§8) but gains
  only about 1.2× once the aperture covers the slopes.
- **A sharper vertical bound.** The h_ξ^½ factor (§6.1) and the Numerov
  symbol being 2.7× better than the e₄ the optimizer uses
  (`grid_optimizer.md` §6) suggest the vertical rule could be both tighter
  and more rigorous. This needs a dispersion measurement.
- **Paper.** The interior stretch, the nonuniform NLBC and the Highlands
  numbers above are not in `main.tex` yet.

## 11. Files

| what | where |
|---|---|
| solver: `complex_layer_steps`, `layer_reflection`, `class_grid_steps`, `_compute_class_steps`, `layer_stretch`, `TabulatedStretch`, `interior_stretch_rate`, `stretch_eval/tail/inverse`, `stretched_exterior_ratio`, `_calc_upper_nlbc_stretched`, `_map_levels` | `pywaveprop/helmholtz_tf_jax.py` |
| front-end flags `terrain_absorb_stretch`, `terrain_upper_stretch`, `terrain_interior_stretch`, `terrain_range_steps`, `terrain_upper_bc`, `terrain_layer_min_angle_deg`, `terrain_layer_reflection` | `pywaveprop/rwp_jax.py` |
| tests `test_rwp_transform_stretched_nlbc`, `test_rwp_transform_interior_stretch`, `test_rwp_transform_complex_layer` | `tests/pywaveprop_jax/test_terrain_transform.py` |
| NLBC derivation | `papers/tap2026/nlbc_nonuniform.md` |
| optimizer context (§5 rule, §8 absorbing layer, §9 interior stretch) | `papers/tap2026/grid_optimizer.md` |
| scripts | `scripts/tap_beta_column.py`, `tap_nlbc_stretch.py`, `tap_highlands_cost.py` |
| data | `papers/terrain_transform/figures/beta_column.json`, `nlbc_stretch.json`, `highlands_cost.json` |
