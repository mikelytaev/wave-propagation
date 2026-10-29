# Grid selection for the rational (Padé) propagator

How `pywaveprop` chooses the reference wavenumber $\beta$, the range step
$\Delta x$, the vertical step $\Delta z$ and the Padé order for a given
precision. The flat-terrain (staircase) path is in `pywaveprop/grid_optimizer.py`
(`_OptTable`, `get_optimal_grid`) and `RationalHelmholtzPropagator.create`
(`pywaveprop/helmholtz_jax.py`). The terrain-following path adds a metric
rescaling in `TerrainFollowingHelmholtzPropagator.create`
(`pywaveprop/helmholtz_tf_jax.py`).

## 1. Error model

One range step of the one-way equation multiplies a plane wave with vertical
wavenumber $k_z$ in a medium of wavenumber $k$ by

$$
P(\xi)=\exp\!\big(i\beta\Delta x(\sqrt{1+\xi}-1)\big),\qquad
\xi=\frac{k^2-k_z^2}{\beta^2}-1 .
$$

The scheme replaces $P$ by the Padé product $R(\xi)=\prod_l(1+a_l\xi)/(1+b_l\xi)$,
whose coefficients depend only on $t=\beta\Delta x$ and the order $(m,n)$. It
also replaces $-k_z^2$ by the discrete vertical symbol $D_{\Delta z}(k_z)$.
The optimizer bounds the two resulting per-step errors separately, for every
wave in the spectrum.

**Spectrum.** The propagating waves have $k\in[k_{\min},k_{\max}]$ (from the
refractivity profile up to the domain top, widened to $\pm0.1\%$ in a
homogeneous medium) and $|k_z|\le k_{z,\max}=k_0\sin\theta_{\max}$, with
$\theta_{\max}$ the aperture. The longitudinal wavenumber squared
$k^2-k_z^2$ therefore lies in

$$
[a,b]=[k_{\min}^2-k_{z,\max}^2,\;k_{\max}^2],
\qquad
\xi\in\Big[\frac{a}{\beta^2}-1,\;\frac{b}{\beta^2}-1\Big].
$$

**Precision.** The user precision $\varepsilon$ (`params.precision`, default
$10^{-2}$) is the error accumulated over the whole range $X$. Per-step errors
are assumed to add linearly, so a step with error $\delta$ is admissible if

$$
\delta/\Delta x\le\varepsilon/X .
$$

## 2. Precomputed table (per Padé order)

`_OptTable.compute()` runs once per order and is cached in `grid_optimizer.dump`.

- $t=\beta\Delta x$ runs over a fixed list of 52 values:
  0.01, 0.05, 0.1–1, 2–10, 20–100, 200–1000, 2000–10000, 20000–50000.
- The per-step tolerance runs over $\delta\in\{10^{-10},\dots,10^{-5}\}$.
- For each pair, the Padé coefficients are computed for $\beta=t$, $\Delta x=1$.
  Bisection then finds the largest interval $[\xi_a,\xi_b]\ni0$ on which
  $|P(\xi)-R(\xi)|\le\delta$:
  - $\xi_a$ is searched in $[-1,0]$;
  - $\xi_b$ is searched in $[0,10]$ and set to 10 if the error never reaches $\delta$.

The bisection assumes that the error grows monotonically away from $\xi=0$ on
each side.

## 3. Query: β, Δx, Δz for one order

`_OptTable.get_optimal(kz_max, k_min, k_max, required_prec, dx_max, dz_max)`
loops over all table entries $(t,\delta)$.

1. **β.** The spectrum must fit into the table interval:
   $a/\beta^2-1\ge\xi_a$ and $b/\beta^2-1\le\xi_b$, i.e.
   $$
   \beta\in[\ell,r],\qquad \ell=\sqrt{b/(1+\xi_b)},\quad r=\sqrt{a/(1+\xi_a)} .
   $$
   The entry is skipped if $\ell>r$. Otherwise $\beta=\ell$, the smallest
   admissible value, which gives the largest $\Delta x=t/\beta$.
2. **Δx.** $\Delta x=t/\beta$. The entry is skipped if $\Delta x>\Delta x_{\max}$
   or $\delta/\Delta x>\varepsilon/X$.
3. **Δz.** The error of the discrete vertical symbol perturbs
   $\xi$ by $\delta\xi$, and the phase of the step by
   $|d\phi/d\xi|\,\delta\xi$, where $\phi=t(\sqrt{1+\xi}-1)$ and $|d\phi/d\xi|=t/(2\sqrt{1+\xi})$. This is largest at
   $\xi_{\min}=a/\beta^2-1$. The code requires
   $$
   e_4(k_{z,\max},\Delta z)\le\frac{t^2\,\delta}{t/(2\sqrt{1+\xi_{\min}})},
   $$
   where
   $$
   e_4(k_z,\Delta z)=\Big|\frac{s-s^2/12}{\Delta z^2}+k_z^2\Big|,\qquad s=2\cos(k_z\Delta z)-2,
   $$
   is the error of the fourth-order symbol (`fourth_order_error_kz`), evaluated
   only at $k_{z,\max}$. If $\Delta z_{\max}$ already satisfies the bound, then
   $\Delta z=\Delta z_{\max}$. Otherwise $\Delta z$ is found by bisection in
   $[10^{-4},\Delta z_{\max}]$.
4. **Objective.** The entry with the largest cell area $\Delta x\,\Delta z$
   wins. If no entry is admissible, $\beta$ is NaN.

The per-step tolerance $\delta$ is a free variable of the search. A small
$\delta$ allows a large $\Delta x$ through the accumulation condition but
narrows $[\xi_a,\xi_b]$ and tightens $\Delta z$. A large $\delta$ does the
opposite. The search trades these off over the discrete table.

## 4. Choice of order and output grid (`RationalHelmholtzPropagator.create`)

1. **Bounds.**
   - $\Delta x_{\max}$ is the output range step, or $X/(n_x/2-1)$ when only a
     point count is given.
   - $\Delta z_{\max}$ is the output height step (or the analogous count-based
     value), further limited to $\Delta z\le2/\bar k$ with
     $\bar k=(k_{\min}+k_{\max})/2$. This limit is needed for the discrete
     upper NLBC (convergence of the discrete Green's function).
2. **Orders.** Either the requested order, or all of $(1,2),\dots,(7,8)$.
3. **Per order.** Run §3. Snap the steps to the output grid:
   $\Delta x\leftarrow\Delta x_{\rm out}/\lceil\Delta x_{\rm out}/\Delta x\rceil$, and
   likewise for $\Delta z$. Compute $x_n=\lceil X/\Delta x\rceil+1$ and
   $z_n=\lceil Z/\Delta z\rceil+1$.
4. **Pick the order.** Choose the order with the smallest cost
   $x_n\,z_n\,n$, where $n$ is the denominator degree, i.e. the number of
   tridiagonal solves per step. The call raises an error if every order fails.

## 5. Terrain-following coordinates (`TerrainFollowingHelmholtzPropagator.create`)

The transformed equation is locally the flat equation with wavenumber
$k\,h_\xi$ and effective vertical step $r\,\Delta\eta$, where
$r=h_\eta/h_\xi$ (local equivalence, Section "Discrete Dispersion Analysis" of the paper). The
flat optimizer is called with global bounds of the metric.

1. **Sample the metrics.** $h_\xi$ and $r$ are evaluated on a provisional
   grid:
   - range spacing $\min(L,\sigma)/8$, where $L$ is the transition height and
     $\sigma$ the terrain smoothing (at most 4000 columns);
   - levels every $\Delta\eta_p/2$ up to the transition height, with
     $\Delta\eta_p=\min(L/200,Z/400)$.

   The extremes lie on the ground. A range spacing of $L/4$, used until
   2026-10, missed them.
2. **Rescale the spectrum.**
   $$
   k\in[k_{\min}h_{\xi,\min},\;k_{\max}h_{\xi,\max}],\qquad
   k_{z,\max}\leftarrow k_{z,\max}h_{\xi,\min}.
   $$
   The scaling of $k_{z,\max}$ by the minimum keeps the smallest longitudinal
   wavenumber $h_\xi^2(k_{\min}^2-k_z^2)$ correct.
3. **Step bounds.** $\Delta z_{\max}\leftarrow\min(\Delta z_{\max}\,r_{\max},\;2/\bar k)$,
   with $\bar k$ the mean of the rescaled $k$ bounds.
4. **Per order.** Run §3. The returned $\Delta z$ resolves $k_{z,\max}h_{\xi,\min}$,
   but the largest local vertical wavenumber is $k_{z,\max}h_{\xi,\max}$, and the
   effective step is $r\,\Delta\eta$. Hence
   $$
   \Delta\eta=\Delta z\,\frac{h_{\xi,\min}}{h_{\xi,\max}}\,\frac1{r_{\max}} ,
   $$
   which is then snapped to the output grid. The order is chosen by cost
   $x_n z_n n$ as in §4.

**Consequences.**
- The metric spread enters twice: it widens the $\xi$ interval the Padé
  approximant must cover (smaller $\Delta x$), and it shrinks $\Delta\eta$ by
  $(h_{\xi,\min}/h_{\xi,\max})/r_{\max}$, about 0.49 for Glen Etive.
- The transition height trades $r_{\max}$ against the $h_\xi$ spread. For
  Glen Etive (smoothing 100 m), raising $L$ from 700 to 5600 m lowers $r_{\max}$
  from 1.78 to 1.05 but widens $h_\xi$ from [0.88, 1.17] to [0.69, 1.32].
  The grid cost barely changes (§7).
- The bounds are global, so the most distorted point of the map sets the grid
  everywhere.

**Aperture.** In both paths $k_{z,\max}$ comes from `params.max_angle_deg`, or
from the source when that is not given. With `terrain_slope_aperture=True`
(the default) the terrain-following path adds twice the maximum slope. Angles
in the terrain-following grid are measured from the coordinate lines, so one
slope is usually enough. The Glen Etive study sets the aperture explicitly
(beam + tilt + slope) and checks it against beam + tilt + 2·slope.

## 6. Observations on the current rule

- **Units of the Δz bound.** The phase argument of §3.3 gives
  $\delta(k_z^2)\le\beta^2\delta/|d\phi/d\xi|$. The code uses $t^2=\beta^2\Delta x^2$
  in place of $\beta^2$, so the tolerance carries an extra factor
  $\Delta x^2$ (in metres). For $\Delta x>1$ m the vertical bound is looser
  than the derivation, by $\Delta x^{1/2}$ in $\Delta z$ since
  $e_4\propto\Delta z^4$; for $\Delta x<1$ m it is tighter. In the tests,
  solutions at $\varepsilon=10^{-2}$ and $10^{-3}$ agree, so the rule works in
  practice. The factor should still be checked against a dispersion
  measurement before it is quoted as a bound.
- **Symbol of the scheme.** $e_4$ is the error of the explicit fourth-order
  difference $\delta^2-\delta^4/12$ (leading term $k_z^6\Delta z^4/90$). The
  scheme is Numerov, $\delta^2/(1+\delta^2/12)$, with leading term
  $k_z^6\Delta z^4/240$, so the bound is conservative by a factor of about 2.7.
- **Worst case only.** The vertical error is evaluated at $k_{z,\max}$ and
  $\xi_{\min}$, and the Padé error at the ends of the $\xi$ interval. Both
  are worst cases; the two errors are not added.
- **Discreteness.** $t$ and $\delta$ come from fixed lists (52 × 6 entries),
  so $\Delta x$ jumps between table values and the optimum is approximate.
  $\beta$ is always the lower end $\ell$ of the admissible interval.
- **Linear accumulation.** $\delta/\Delta x\le\varepsilon/X$ assumes the per-step
  errors add coherently, which is pessimistic for random-phase errors and
  exact for a systematic phase drift.
- **Cost.** Grid selection itself is a table lookup (milliseconds). It is not
  the bottleneck of a run. In the Glen Etive case at $10^{-2}$, the upper NLBC
  kernel (about 9 s) and the march (about 6 s) dominate.

## 7. Staircase and terrain-following grids compared

`scripts/tap_grid_compare.py` runs both constructors up to grid selection
(no march) and writes `papers/terrain_transform/figures/grid_compare.json`.

**Setup.**
- Base case: Glen Etive, 100 MHz, V polarization, wet ground, ε = 10⁻², beam 10°
  tilted 3°, smoothing 100 m, L = 1400 m, output grid 25 m × 1 m.
- Both solvers get the same physical aperture, beam + tilt + slope.
- The staircase domain starts at z = 0. The terrain-following domain starts at
  the mean terrain height, so it is about 15 % shorter.

**Cost.** Cost is x_n z_n n (Mcell). "Ratio" is the TF cost over the staircase
cost. "Raw" is the same ratio from the optimizer steps before snapping them to
the output grid, which divides 25 m and 1 m into integers.

| case | slope | staircase (m,n) Δx/Δz [m] | Mcell | TF (m,n) Δξ/Δη [m] | Mcell | h_ξ range, r_max | ratio | raw |
|---|---|---|---|---|---|---|---|---|
| Gaussian hill 25 m  | 1.7° | (2,3) 12.5/0.50 | 7.5 | (3,4) 12.5/0.50 | 10.0 | 0.99–1.01, 1.01 | 1.3 | 0.8 |
| Gaussian hill 50 m  | 3.3° | (3,4) 12.5/0.50 | 10.2 | (3,4) 12.5/0.50 | 10.2 | 0.98–1.01, 1.03 | 1.0 | 1.8 |
| Gaussian hill 100 m | 6.7° | (3,4) 12.5/0.50 | 10.6 | (4,5) 12.5/0.50 | 13.1 | 0.96–1.02, 1.06 | 1.2 | 1.9 |
| Gaussian hill 200 m | 13° | (4,5) 12.5/0.50 | 14.1 | (6,7) 12.5/0.50 | 19.5 | 0.95–1.05, 1.11 | 1.4 | 1.6 |
| Gaussian hill 400 m | 25° | (7,8) 12.5/0.33 | 38.1 | (7,8) 8.33/0.20 | 92.8 | 0.87–1.15, 1.11 | 2.4 | 3.0 |
| Gaussian hill 800 m | 43° | (6,7) 6.25/0.083 | 326 | map singular | – | – | – | – |
| Glen Etive (base) | 23.5° | (5,6) 12.5/0.33 | 20.3 | (6,7) 5.0/0.167 | 101 | 0.79–1.21, 1.35 | 5.0 | 5.8 |
| 30 MHz   | 23.5° | (3,4) 12.5/1.0 | 4.5 | (5,6) 12.5/0.50 | 11.6 | same | 2.6 | 2.5 |
| 300 MHz  | 23.5° | (7,8) 8.33/0.071 | 189 | (7,8) 1.79/0.018 | 2960 | same | 16 | 17 |
| 1 GHz    | 23.5° | (7,8) 2.08/0.0064 | 8.4·10³ | (7,8) 0.54/0.0030 | 5.9·10⁴ | same | 7.0 | 7.1 |
| 3 GHz    | 23.5° | (7,8) 0.71/0.0012 | 1.3·10⁵ | (7,8) 0.18/0.00032 | 1.7·10⁶ | same | 13 | 13 |
| ε = 10⁻¹ | 23.5° | identical to 10⁻² | | | | | 5.0 | 5.8 |
| ε = 10⁻³ | 23.5° | (7,8) 12.5/0.20 | 45.1 | (7,8) 5.0/0.091 | 212 | same | 4.7 | 7.1 |
| ε = 10⁻⁴ | 23.5° | (7,8) 12.5/0.10 | 90.2 | (7,8) 5.0/0.053 | 366 | same | 4.1 | 5.3 |
| L = 700 m  | 23.5° | (5,6) 12.5/0.33 | 20.3 | (7,8) 8.33/0.143 | 80.9 | 0.88–1.17, 1.78 | 4.0 | 3.9 |
| L = 2800 m | 23.5° | (5,6) 12.5/0.33 | 35.1 | (7,8) 5.0/0.167 | 214 | 0.72–1.26, 1.14 | 6.1 | 6.0 |
| L = 5600 m | 23.5° | (5,6) 12.5/0.33 | 58.8 | (7,8) 5.0/0.20 | 310 | 0.69–1.32, 1.05 | 5.3 | 6.3 |
| beam 2°, no tilt  | 23.5° | (4,5) 12.5/0.50 | 11.3 | (7,8) 8.33/0.25 | 46.2 | same | 4.1 | 5.0 |
| beam 5°, no tilt  | 23.5° | (5,6) 12.5/0.50 | 13.5 | (6,7) 5.0/0.20 | 84.2 | same | 6.2 | 8.1 |
| beam 20°, no tilt | 23.5° | (7,8) 12.5/0.25 | 36.1 | (6,7) 5.0/0.125 | 135 | same | 3.7 | 4.7 |

The hill cases use σ = 500 m over 10 km, L = max(2·relief, 400 m) and a domain
top 1000 m above the summit. The L rows raise the domain top to L + 950 m
where needed, which is why their staircase cost also grows.

**Where the factor comes from (Glen Etive, raw steps).**

| factor | value | cause |
|---|---|---|
| Δx / Δξ | 15.9 / 5.5 = 2.9 | the Padé interval must cover k ∈ [0.79 k_min, 1.21 k_max] |
| Δz / Δη | 0.339 / 0.169 = 2.0 | rescaling (h_ξ,min/h_ξ,max)/r_max = 0.48; the optimizer Δz is nearly the same (0.351) |
| n_TF / n_st | 7/6 = 1.17 | higher order chosen for the wider ξ interval |
| Z_TF / Z_st | 0.85 | domain starts at the mean terrain height |
| **total** | **5.8** | |

**Trends.**
- **Slope.** The cost of terrain-following coordinates follows the metric
  spread, and therefore the slope:
  - below about 7°, h_ξ stays within ±4 % and the grids are nearly the same
    (ratio 1–2);
  - at 13°, the ratio is 1.6; at 25°, 3–6;
  - at 43° (an 800 m hill with σ = 500 m), the map is singular, and only a
    larger smoothing or transition height would make it usable.
- **Frequency.** The ratio grows with frequency, from 2.5 at 30 MHz to 7–17
  above 300 MHz:
  - At 30 MHz, the NLBC and output limits set both vertical steps, so only
    the Δη rescaling remains.
  - At higher frequency, the dispersion bound sets Δz, and it reacts to the
    widened k interval: the optimizer Δz alone drops by 2× at 300 MHz and
    3 GHz.
  - The ratio does not change smoothly from case to case, because the
    (t, δ) table is discrete.
  - The absolute grids above 300 MHz are impractical for both methods at this
    aperture. The fourth-order Δz bound with the t² factor (§6) is very
    restrictive there.
- **Precision.** The ratio is 4–7 for ε from 10⁻² to 10⁻⁴. ε = 10⁻¹ gives the
  same grid as 10⁻²: the admissible per-step error ε Δx/X exceeds the largest
  table tolerance δ = 10⁻⁵, so the table caps it.
- **Transition height.** The ratio is 4–6 for L from 700 m to 5600 m. A
  smaller r_max is offset by a wider h_ξ spread.
- **Aperture.** The ratio is 4–8 for beams from 2° to 20°. In every case it
  is dominated by the terrain-induced spread, not by the beam.

**Interpretation.**
- The staircase optimizer does not see the terrain. Its grid controls only the
  Padé and dispersion errors of the flat scheme. It does not control the
  staircase geometry error, which in the Glen Etive study stays at about
  10 dB aloft (V) at both 10⁻² and 10⁻³.
- The cheaper staircase grid therefore does not buy accuracy. The
  terrain-following grid is larger because it resolves the stretched local
  spectrum, the quantity that actually limits accuracy over steep terrain.
- Most of the factor comes from the global bounds. A reference wavenumber
  that changes with range removes part of it (§8).

## 8. Reference wavenumber per column class (`beta_mode='column'`)

`TerrainFollowingHelmholtzPropagator.create(..., beta_mode='column')`
(front-end: `params.terrain_beta_mode = 'column'`) gives each class of columns
its own β.

**Classes.**
- On the provisional grid of §5, every column gets the range of h_ξ over its
  height. A min/max filter widens it to the columns that one step can read.
- Columns are binned by log h_min + log h_max (`n_beta_classes`, default 16).
- A class carries (h_min, h_max, r_max) over its members.

**Grid** (`class_grid`).
- The range and vertical steps stay global. Candidates are Δξ = Δx_out/n.
- A candidate is admissible if every class has a table entry (t_j, δ_j) and a
  β in that entry's interval [ℓ_j, r_j] with:
  - β Δξ ≤ t_j, assuming the Padé error does not grow when t decreases;
  - δ_j β/t_j ≤ ε/X, i.e. accumulation over the entry's own step, as the
    flat optimizer does before snapping.
- Each class takes β = ℓ_j of the entry with the largest vertical step.
- Δη is the smallest class value of Δz (h_min/h_max)/r_max. The order is
  chosen by cost as before.
- With one class the search returns the global grid exactly.
- The Padé error of each class approximant is evaluated directly over its
  spectrum and printed. For Glen Etive it is 10⁻⁹–10⁻⁶, against the
  tabulated δ = 10⁻⁵.

**March.** Step n uses the class of its midpoint column.
- exp(iΔξ√B) = exp(iβ_kΔξ) R_k(L_k), with L_k = (β/β_k)²(1+L) − 1. The
  envelope keeps the global β, so the step is multiplied by
  exp(i(β_k − β)Δξ).
- The coefficient tables and the flux factors (exact powers of 1 + L) keep
  the global β.

**Upper boundary.** The discrete NLBC is a convolution over the boundary
history with one kernel. It holds only if every step is the same, so this mode
uses a Dirichlet top over an absorbing layer (`absorb_m`, damping
0.02 i k² ramped quadratically; `params.terrain_absorb_m`).

**Glen Etive, V polarization, wet ground** (`scripts/tap_beta_column.py`,
`papers/terrain_transform/figures/beta_column.json`). The table below gives the
grid, cost and times. Times were measured with each run alone; "march" is the
second, compiled call of `compute`.

| run | ε | order | Δξ [m] | Δη [m] | Mcell | create | first compute | march |
|---|---|---|---|---|---|---|---|---|
| global β, NLBC | 10⁻² | (6,7) | 5.0 | 0.167 | 101 | 10.1 s | 7.6 s | 5.9 s |
| global β, absorbing | 10⁻² | (6,7) | 5.0 | 0.167 | 115 | 0.7 s | 7.4 s | 6.2 s |
| β per class, absorbing | 10⁻² | (7,8) | 8.33 | 0.167 | 78.5 | 2.5 s | 5.6 s | 4.1 s |
| global β, NLBC | 10⁻³ | (7,8) | 5.0 | 0.091 | 212 | 16.1 s | 13.1 s | 11.7 s |
| global β, absorbing | 10⁻³ | (7,8) | 5.0 | 0.091 | 240 | 0.8 s | 13.3 s | 13.1 s |
| β per class, absorbing | 10⁻³ | (7,8) | 8.33 | 0.100 | 131 | 3.0 s | 7.8 s | 7.1 s |

The absorbing rows include the 300 m layer, about 13 % more rows. The
NLBC "create" time is the kernel computation.

**Accuracy.** Path-loss differences against the global-β NLBC run, median over
the illuminated map in the height bands 0–100 / 100–500 / 500–2000 m:

- global β, absorbing − NLBC: 0.000 dB in all bands, at both precisions.
- β per class − NLBC: 0.002 / 0.005 / 0.004 dB at 10⁻², and
  0.003 / 0.007 / 0.006 dB at 10⁻³. The maximum on the 10 m and 50 m cuts is
  0.02 dB.
- For scale, NLBC 10⁻² − 10⁻³: 0.001 / 0.002 / 0.002 dB.

**Result.**
- The range step grows by 1.67 (5.0 → 8.33 m), and the cost drops by 1.5×
  at 10⁻² and 1.8× at 10⁻³.
- End to end, create plus first compute goes from 17.7 s to 8.1 s at 10⁻²,
  and from 29 s to 11 s at 10⁻³. Part of this saving comes from dropping the
  NLBC kernel, which the absorbing layer alone already does.
- The TF/staircase cost ratio for Glen Etive falls from 5.0 to about 3.9 at
  10⁻² (78.5 against 20.3 Mcell, with the absorbing layer counted).
- Four classes already give the full gain; more classes change nothing.

**Stretched absorbing layer** (`absorb_stretch=True`, default;
`params.terrain_absorb_stretch`).
- On the uniform grid the 300 m layer takes 1800 rows at Δη = 0.167 m
  (12 % of the column) and 3000 rows at 0.1 m.
- The stretched layer maps z = z_off + Z(η) with s = dZ/dη:
  - s rises from 1 to s_max over the lower third of the layer (quintic
    smoothstep, C²) and stays at s_max above;
  - s_max = 1/(Δη k_z,max), so that k_z s Δη ≤ 1 and the waves of the aperture
    stay resolved where they meet the damping;
  - the damping is still defined in physical height.
- In the scheme this is a metric h_η = s, h_ξ = 1 of the identity region:
  m = s², rt = ρs and V = 0. No new discretization is needed.
- For Glen Etive: s_max = 4.8 and 8.0, and the layer takes 456 and 471 rows.

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

- **Accuracy.** Stretched against uniform layer: 0.000 dB median in all bands,
  and at most 0.004 dB on the cuts. Against the NLBC run, the result is as for
  the uniform layer (0.002–0.007 dB median).
- **Speed.** The march is about 10 % faster. The stretched layer costs fewer
  rows than the NLBC domain would need at the same Δη, and it needs no
  kernel. Even with a global β it is therefore cheaper than the NLBC
  (7.6 s against 17.4 s end to end at 10⁻²).
- **Remaining cost.** The flux factors are solved on the lower n_flux rows,
  which do not change. The rest of the time is the march below the layer.

**What still limits Δξ.**
- Below the transition height, the narrowest class still spans
  h_ξ ∈ [0.785, 1]. With k_z,max h_min in a, its longitudinal range
  b/a = 2.5 is wider than the staircase's 1.55.
- The table jumps from t = 10 to t = 20, and t = 20 is barely admissible even
  for the flattest class. So t ≤ 10 and Δξ ≤ 10/β ≈ 6–8 m.
- A finer t grid between 10 and 20 in `_OptTable` would let Δξ land between
  the current values. The flat optimizer would gain from it too.
- The Δη factor (h_min/h_max)/r_max is now evaluated per class, but the
  binding class still has r_max = 1.35. That part comes from the vertical
  stretching itself.

## 9. Vertical step following the distortion (`interior_stretch=True`)

§5 chooses one $\Delta\eta$ for the whole map from the product of two global
extremes, $h_{\xi,\max}\,r_{\max}$. The local requirement is on the physical
vertical step. At a node, the local scheme has vertical wavenumber
$k_z h_\xi$ on the step $r\,\Delta\eta$, so the product is
$k_z h_\eta\Delta\eta$. Hence the condition is
$$
h_\eta(\xi,\eta)\,\Delta\eta\le h_{\xi,\min}\,\Delta z ,
$$
with $\Delta z$ the step returned by the flat optimizer (the same linear
rescaling as §5.4). Two consequences:

- **Pointwise bound.** $\max h_\eta$ is smaller than
  $h_{\xi,\max}\,r_{\max}$, because the two extremes are not at the same point:
  $h_\xi$ is extreme on the ground, $r$ around mid-height of the transition
  layer. On the Highlands path the values are 1.27 against 1.52, and for the
  200 m hill 1.28 against 1.51.
- **Level by level.** $Q(\eta)=\max_\xi h_\eta(\xi,\eta)$ varies with height
  and is 1 above the layer. A stretching $\eta=Z(\eta')$ with rate
  $s=dZ/d\eta'=Q_{\max}/Q(\eta)$ keeps the worst physical step
  $\max_\xi h_\eta s\,\Delta\eta'$ the same on every level.

**Implementation.**
- **Composite map.** The stretching composes with the orthogonal map. The
  coordinates stay orthogonal, $h_\xi$ is unchanged and
  $h_\eta\to h_\eta s$. The tables, the identity region and the impedance
  closure ($h_{\eta}$ on the ground) therefore take $h_\eta s$; $V$ does not
  change, because $s$ does not depend on $\xi$.
- **Profile.** $Q$ is sampled on levels independent of the domain height.
  It is replaced by a smooth upper envelope: a max filter and a Gaussian on
  $\log Q$, of width $\max(4\lambda, L/40)$, tapered to 1 above the layer.
- **Rounding.** $\Delta\eta'$ is still rounded down to align the output rows.
  The loss is returned through the scale $s(0)=h_{\xi,\min}\Delta z/(Q_{\max}\Delta\eta')\ge1$,
  because the stretching gives up the alignment anyway.
- **Table.** $\eta'(\eta)=\int d\eta/s$ is tabulated (Simpson) and inverted by
  Newton iterations on its cubic Hermite interpolant.
- **Upper condition.** With the NLBC, $s\le 2/(\bar k\Delta\eta')$ and the
  exterior is the stretched one of `nlbc_nonuniform.md`. With an absorbing
  layer, $s$ rises further to $1/(\Delta\eta' k_{z,\max})$ inside the layer.
- **Per-class β.** Class $c$ allows the physical step
  $D_c=\Delta\eta_c\,q_c$, with $q_c$ the largest $h_\eta$ of the class and
  $Q_c(\eta)$ its profile. The rate is then
  $s(\eta)\,\Delta\eta'=\min_c D_c/Q_c(\eta)$, i.e. a single profile
  $q_{\rm eff}=\max_c Q_c\,D_{\min}/D_c$.
- **Exclusions.** It replaces `upper_stretch` and is not available for the
  strip map.

**Checks.** The script is `scripts/tap_highlands_cost.py`. The hills run on a
3 km path at 300 MHz, compared with the uniform grid at $10^{-4}$:
- A constant $s$ reproduces the uniform grid with the same physical step to
  $2.5\cdot10^{-11}$ at the common rows. $s\equiv1$ reproduces it to $10^{-11}$.
- 200 m hill, $10^{-3}$: 16107 rows instead of 21767, 27.7 s instead of 36.7 s;
  largest error $8.1\cdot10^{-5}$ against $7.9\cdot10^{-5}$ for the uniform grid.
  At $10^{-2}$: 4274 instead of 5775 rows, $3.5\cdot10^{-3}$ against $2.8\cdot10^{-3}$.
- NLBC: domain tops of 500 and 900 m on the same stretched grid, with the beam
  leaving through the top, agree to $7.6\cdot10^{-7}$ (200 m hill) and
  $6.5\cdot10^{-13}$ (60 m hill).
- **Comparing runs.** On uniform grids the output rows of all runs sit at
  the same heights ($\eta=j\,\Delta z_{\rm out}$). On a stretched grid they do
  not, and interpolating $|u|$ from 2 m rows at $\lambda=1$ m produces dB-level
  differences that are not solver error. Comparisons therefore use dense
  output (0.25 m) and the complex reference interpolated along the same
  trajectory.

**Highlands path** (CPU, each run alone; dense output for the comparison):

| run | rows at $10^{-2}$ / $10^{-3}$ | time at $10^{-2}$ / $10^{-3}$ | 10 m above ground, vs uniform at $10^{-3}$ |
|---|---|---|---|
| one β, uniform | 49888 / 89144 | 298 / 787 s | — |
| one β, interior stretch | 36400 / 65194 | 233 / 378 s | 0.058 dB median, 0.16 p90, 0.52 max |
| β per class + absorbing layer, uniform | — | 139 / 367 s | — |
| β per class + absorbing layer, interior stretch | 35500 / 69087 | 111 / 279 s | 0.048 dB median, 0.14 p90, 0.46 max |
| staircase | — | 50 / 80 s | 1.0 dB median |

The differences are below the 0.12 dB that separates the uniform grid's own
$10^{-2}$ and $10^{-3}$ solutions. At $10^{-3}$, one β gains more time
(−52 %) than rows (−27 %). The same happened with `upper_stretch` (−44 % time
for −13 % rows); the cause has not been isolated.
