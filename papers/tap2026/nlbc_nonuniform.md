# Discrete upper NLBC on a nonuniform vertical grid

How the upper discrete nonlocal boundary condition (NLBC) of the rational
propagator extends to a vertical grid that is not uniform near or above the
top boundary. Implementation:
- `TerrainFollowingHelmholtzPropagator.stretched_exterior_ratio` and
  `_calc_upper_nlbc_stretched` (`pywaveprop/helmholtz_tf_jax.py`);
- the front-end option `params.terrain_upper_stretch`.

Verification is in `scripts/tap_nlbc_stretch.py` and
`papers/terrain_transform/figures/nlbc_stretch.json`.

## 1. Result in one line

Only one quantity of the existing construction depends on the vertical grid:
the scalar ratio R(σ) = u_{N−1}/u_N of the decaying exterior solution of
L u = σ u. On a nonuniform grid, R(σ) comes from a backward three-term
recursion over the nonuniform exterior rows. It starts from the exact ratio of
the uniform tail. The inverse Z-transform, the matrix function over the Padé
substeps and the discrete convolution are unchanged.

## 2. The scheme with a vertical stretching

A stretched grid z = Z(η) on a uniform computational grid η_j = jΔη is a
coordinate metric h_η = s(η) = dZ/dη, with h_ξ = 1. The rows considered here
lie above the transition height, where nothing depends on range. In the
terrain-following scheme this gives
- r = s and m = r² = s²;
- rt = ρs (density ρ, constant at the top);
- V = 0.

Each Padé factor, (1 + c L) with c = a_l or b_l, is the tridiagonal matrix
(`_tf_substep`)

    M_c = τ² (I + α D_w) diag(m) + c [ D_w + τ² (I + α D_w) diag(m het) ],   τ = βΔη,

with the weighted second difference

    (D_w f)_j = w^l_j f_{j−1} − (w^l_j + w^r_j) f_j + w^r_j f_{j+1},
    w^l_j = rt_j / rt_{j−1/2} = s_j / s_{j−1/2},   w^r_j = s_j / s_{j+1/2}.

Here α = 1/12 (Numerov) and het = k²/β² − 1. Hence M_c = P (I + cL), with
P = τ² (I + αD_w) diag(m) and the discrete operator

    L = diag(het) + [τ² (I + α D_w) diag(m)]⁻¹ D_w .

The common factor P cancels in every substep (I + b_l L) v_l = (I + a_l L) v_{l−1}.
The range scheme is therefore the plain Padé chain in L, and the stretching
enters only through L.

## 3. Range direction: Z-transform and reduction to a scalar problem

The derivation follows the uniform case and does not use the vertical grid.

1. **Initial data.** The exterior j ≥ N holds no initial field: the source is
   below and the stretching is range independent.
2. **Z-transform.** Transform the substep fields in the step index n,
   V_l(t) = Σ_n v^n_l t⁻ⁿ. The chain with v^n_0 = v^{n−1}_p turns into a
   p × p system in the substep index whose matrix depends on t. Pointwise in
   height it reads L V = V M(t)ᵀ. Here M(t) = A(t)⁻¹ B(t) is the matrix
   `solve(matrix_a, matrix_b)` of `_calc_nlbc`, built from the Padé
   coefficients a_l, b_l only.
3. **Diagonalization.** Every eigenvector of M(t) with eigenvalue σ gives a
   scalar exterior problem (L − σ) w = 0 for j ≥ N. Its decaying solution
   fixes w_{N−1}/w_N = R(σ).
4. **Kernel.** Reassembling the modes gives the boundary relation
   V_{N−1}(t) = R(M(t)) V_N(t), with R applied as a matrix function (`funm`).
   The inverse Z-transform gives the kernel K_n = Z⁻¹[R(M(t))], and the
   boundary row of substep i is

       u_{N−1} − (K_0)_{ii} u_N = Σ_{k<i} (K_0)_{ik} u_N^{(k)} + Σ_{n'≥1} K_{n'} u_N^{n−n'} .

None of this depends on the vertical grid, except R(σ).

## 4. The exterior recurrence on a nonuniform grid

**Recurrence.** (L − σ) w = 0 means D_w w = τ² (I + αD_w) diag(m (σ − het)) w.
With g_j = τ² m_j (σ − het_j) and G_j = g_j w_j this is
D_w (w − αG) − G = 0. Row by row, for j ≥ N:

    A⁻_j w_{j−1} + C_j w_j + A⁺_j w_{j+1} = 0,
    A⁻_j = w^l_j (1 − α g_{j−1}),
    A⁺_j = w^r_j (1 − α g_{j+1}),
    C_j  = −(w^l_j + w^r_j)(1 − α g_j) − g_j .

- **Check against the uniform case.** For s ≡ 1 (w^l = w^r = 1, m = 1) this
  is A = 1 − ατ²(σ − het) and C = −2 + (2α − 1)τ²(σ − het): the coefficients
  `a_m1`, `a_1`, `c` of `_calc_upper_nlbc`.
- **Which values enter.** Row N couples w_{N−1}, w_N and w_{N+1}. Its
  coefficients use the interior values s_{N−1}, m_{N−1} and het_{N−1} and the
  half-node value s_{N−1/2}. They are evaluated with the actual grid, so the
  interior may be nonuniform right up to the boundary.

**Minimal solution.** For σ on the contour |t| = τ_z > 1 (`inv_z_transform_tau`),
the recurrence has a minimal (decaying) and a dominant solution. The ratio
r_j = w_j/w_{j−1} of the minimal solution satisfies

    r_j = − A⁻_j / (C_j + A⁺_j r_{j+1}),

i.e. r_N is the continued fraction

    r_N = −A⁻_N / (C_N − A⁺_N A⁻_{N+1} / (C_{N+1} − A⁺_{N+1} A⁻_{N+2} / (C_{N+2} − …))).

By Pincherle's theorem the continued fraction converges exactly when a
minimal solution exists, and then to its ratio. The backward recursion is the
stable direction for a minimal solution (as in Miller's algorithm).

**Uniform tail.** Above some node J the stretching is constant, s = s_top
(from node J − 1, including J − 1/2). There the recurrence is the uniform one
with step Δz_eff = s_top Δη, since τ² s_top² = (βΔz_eff)². Its ratio at J is
known exactly:
- **constant het:** 1/r_J is the inverse of the decaying root μ of
  A μ² + C μ + A = 0 (`sqr_eq`);
- **linear het:** het = h₀ + γ j Δz_eff, a refractivity gradient. The
  continued fraction `bessel_ratio_4th_order` with step Δz_eff, slope γ per
  physical metre and node index J gives it.

Starting from this value, the recursion runs over the rows J − 1, …, N, and
R(σ) = u_{N−1}/u_N = 1/r_N. It is exact (no truncation) and costs J − N
operations per σ. Above the top node, het is continued linearly in physical
height (γ from the two top nodes), as the uniform NLBC assumes.

## 5. Cases

| grid near the top | R(σ) |
|---|---|
| uniform (s ≡ 1) | quadratic root / Bessel CF with Δη: the existing NLBC |
| stretched interior, top in the uniform tail (J = N) | the existing NLBC with Δz_eff = s_top Δη and γ per physical metre: no recursion |
| top inside the stretching (exterior nonuniform) | backward recursion over J − N exterior rows from the exact tail ratio |
| exterior stretched without end (no uniform tail) | infinite continued fraction (§4), truncated adaptively |

## 6. Conditions and limits

- **Range invariance.** The exterior must be the same at every step, as for
  the uniform NLBC. That holds above the transition height with one
  reference wavenumber β. It does not hold with `beta_mode='column'`, where
  each column class has its own β and Padé coefficients; that mode keeps the
  absorbing layer.
- **Kernel convergence.** The uniform-tail kernel needs βΔz_eff < 2, the
  existing limit Δz ≤ 2/k̄ applied to the effective step. The automatic
  s_top respects it.
- **Transparency.** The condition is exactly transparent for the discrete
  problem on the infinite stretched grid. It does not remove reflections
  caused by the stretching itself, which are a property of that discrete
  medium. The stretching must therefore be smooth (quintic smoothstep, C²,
  over 20 wavelengths by default), and the coarse step must still resolve
  the aperture.
- **Choice of s_top.** Below the top, Δη resolves k_z h_ξ,max on the local
  step r Δη. Where h_ξ = r = 1, the same error budget allows s Δη up to
  Δη h_ξ,max r_max, so

      s_top = min(h_ξ,max r_max, 2/(k̄ Δη)).

  For Glen Etive this is 1.63. It is 1.17 for a 100 m hill.

## 7. How to use it

**A. Coarser grid above the distorted layer, NLBC on top** (practical case,
`params.terrain_upper_stretch = True`).
- The metric penalty on Δη, (h_ξ,min/h_ξ,max)/r_max, is needed only in the
  distorted layer.
- Above it, s rises from 1 at the end of the flux rows (η_lab + 4λ + 6Δη) to
  s_top over 20 λ (physical), then stays at s_top.
- The top lies in the uniform tail, so the NLBC is the uniform one with
  Δz_eff = s_top Δη.
- The saving grows with the share of the domain above the transition height.
  It is 14 % of the rows for Glen Etive (L = 1400 m, top 2640 m). Up to a
  fraction 1 − 1/s_top of the upper part is saved, so a tall tropospheric
  domain over a low transition height gains most.

**B. Top inside the stretching** (`upper_stretch_start_m`,
`upper_stretch_ramp_m`, `upper_stretch_s_top`). The exterior is nonuniform,
and the backward recursion of §4 gives the NLBC. This case exists mainly to
verify the derivation; there is no reason to stop a ramp at the boundary.

**Not needed: a stretched absorbing layer with the NLBC.** The NLBC removes
the exterior altogether. The absorbing layer (with its own stretching) is
only for runs where the NLBC cannot be used, such as `beta_mode='column'`.

## 8. Verification

The test of an exact discrete NLBC is that it gives the solution of a
*taller* domain on the *same* grid. The comparisons below use path loss on a
Cartesian grid up to 20 m below the top, in the region below 130 dB.
"Naive" applies the uniform NLBC with the local step s_N Δη at the top,
ignoring the stretching of the exterior.

| case | comparison | median | max |
|---|---|---|---|
| Glen Etive, 100 MHz, V, s_top = 1.63 | tail − uniform grid | 8·10⁻¹¹ dB | 0.010 dB |
| | top inside ramp − taller domain | 2·10⁻⁶ dB | 2.6·10⁻⁵ dB |
| | naive − taller domain | 2·10⁻⁶ dB | 1.3·10⁻⁴ dB |
| beam 15° up through the top, 100 m hill, s_top = 1.17 | top inside ramp − taller domain | 2·10⁻⁸ dB | 4.2·10⁻⁶ dB |
| | naive − taller domain | 5·10⁻⁶ dB | 3.3·10⁻³ dB |
| same, strong stretching s_top = 4 | top inside ramp − taller domain | 2·10⁻⁸ dB | 4.2·10⁻⁶ dB |
| | naive − taller domain | 4·10⁻⁵ dB | 1.9·10⁻² dB |

- The derived NLBC agrees with the taller domain to the tolerance of the
  kernel quadrature (about 4·10⁻⁶ dB), independent of the stretching. The
  naive condition is 800–4600 times worse when the beam reaches the top.
- The recursion is consistent: starting it 50 or 400 rows deeper in the
  uniform tail changes R(σ) by at most 3·10⁻¹⁵ (constant het) and 8·10⁻¹² (linear
  het, the continued-fraction tolerance).
- "tail − uniform grid" is the discretization error of the coarser grid
  aloft, not an NLBC error:
  - 0.01 dB for Glen Etive and 0.2 dB (max, near nulls) for the beam case;
  - with s_top = 4, which is not resolved, it grows to 3.3 dB. That is why
    s_top comes from the error budget.
- Regression test: `test_rwp_transform_stretched_nlbc`. The beam leaves
  through a strongly stretched top, and the relative field difference
  against a taller domain is below 10⁻⁵.

**Cost (Glen Etive, 10⁻², run alone).**

| grid | rows | wall time |
|---|---|---|
| uniform grid | 13541 | 18 s |
| stretched aloft | 11644 | 15 s |

The NLBC kernel takes the same time on both grids: the recursion has no rows
when the top is in the tail.
