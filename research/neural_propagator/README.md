# Learned range step for the terrain-following PE: first study

Branch `neural-propagator`, started 2026-10-16.

**Question.** Can a neural operator replace one range step of the terrain-following rational PE
propagator, marched autoregressively, and make the solver faster at equal accuracy?

**Short answer, for now: no.**
- **Structure is what matters.** A generic deep operator (FNO) learns a one-step map, to 5 % error
  per 25 m step, but blows up when rolled out.
- **Best model so far.** A physics-structured operator with a learned linear correction is stable.
  It reaches 29 % median relative L2 error over 3 km.
- **Comparison with the rational solver.** The rational solver gets about 10⁻⁴ in 2–5 s. The best
  learned step needs 0.25 s to march plus 2.2 s to prepare the terrain-dependent screens, so it is
  only about 2× faster and three orders of magnitude less accurate.
- **The speed potential is real.** The march alone is 10–20× faster than the rational solver.
- **The accuracy gap has an identifiable cause** (Section 5).

The literature survey is in [literature.md](literature.md).

## 1. Setup

| | |
|---|---|
| Physics | 300 MHz (λ = 1 m); perfectly conducting ground; homogeneous atmosphere; random smooth hills (1–4 Gaussians, heights −40…70 m, widths 100–350 m, max slope 12°); 1–3 Gaussian beams at −8…10° |
| Coordinates | orthogonal terrain-following map, transition layer L = 150 m, `z_ref = max T` |
| Reference | `helmholtz_tf_jax` rational propagator, precision 1e-2, discrete NLBC at 420 m. Its grid is Padé (6,7)–(7,8), Δξ = 3.6–5 m, Δη = 0.025–0.042 m. Accuracy is about 10⁻⁴: it changes by 1.4·10⁻⁴ at precision 1e-3 and by 3·10⁻⁴ with 1 m metric tables. |
| Learned-step grid | 25 m range steps (50 m in one variant); a window of 0 ≤ η < 256 m at Δη = 0.25 m (1024 rows). Errors are measured below 200 m. Above 200 m the window has a quadratic absorbing layer. |
| Inputs | field column v_n (complex envelope relative to K0); log h_ξ and log h_η at 6 range columns across the step |
| Data | 240 scenarios of 3 km (200 train, 40 test); 121 columns each, so 24 000 training pairs. Generation took about 25 min on 6 CPU processes. |
| Hardware | CPU only (20 cores; JAX has no CUDA in this environment) |

The first data set used Δη = 0.5 m. It is kept in `data_d050/`, and the one-step numbers marked "(0.5 m)"
below come from it. Where h_η > 1, the map stretches the field, so its computational wavenumber
exceeds K0. Up to 0.3 % of the energy was then above 0.8 × Nyquist, so the data were regenerated at
0.25 m. This turned out not to be the main error source (Section 5).

## 2. Models (`models.py`)

| Model | Structure | Linear in v? | Params |
|---|---|---|---|
| `FreeStep` | exact free-space phase in the sine basis; ignores the terrain | yes | 0 |
| `prior` | 1–4 split steps with the phase screen K0(h_ξ − 1)Δ | yes | 0 |
| `LearnedSplitStep` | K split steps. A CNN hypernetwork produces amplitude/phase screens from the metrics, and the free-space multipliers get learned corrections. | yes | 180 k |
| `SymbolStep` | Pseudo-differential operator. The local frozen-coefficient symbol (q_a/q_b)^½ exp(iΔ(h_ξ√(K0² − κ²/h_η²) − K0)), i.e. the flux-conserving step of the paper, is expanded in a 64-term POD basis P_l(κ): Σ_l w_l(η) F⁻¹[P_l F v]. The Liouville factors r^±½ and the bulk phase are screens. | yes | 0 |
| `SymbolStep` + learning (`symres`) | The above plus bounded learned corrections of the effective metrics (≤ 10 %), amplitude/phase screens, and a learned rank-8 linear residual Σ_m g_m(η) F⁻¹[Q_m(κ) F[h_m(η) v]] that can represent odd symbols such as first derivatives | yes | 394 k |
| `FNOStep` | 4-layer FNO (width 32, 320 modes) on [Re/Im v, Re/Im of the free step, metrics]. It predicts a correction to the free step, and the input is normalized by max\|v\|. | no | 2.6 M |

**Design rules from the survey that were adopted.**
- The step is linear in the field and conditioned on the local coefficients through a hypernetwork.
- The arithmetic is complex.
- The vertical basis is non-periodic: a sine series, which keeps v₀ = 0 exactly.
- The model is the identity correction wherever the metrics are the identity, because the screens are net(f) − net(0).

## 3. Results

### One step (median relative L2 per 25 m step, below 200 m)

| Model | Error |
|---|---|
| free step (0.5 m) | 0.56 |
| phase-screen split step, no learning (0.5 m) | 0.15 |
| learned split step after 1000 iterations (0.5 m) | 0.33, plateaued |
| symbol step, no learning (0.5 m) | 6.1·10⁻³ |
| symbol step + learned effective metrics only, without the residual (0.5 m) | 7.4·10⁻³; median unchanged from 7.5·10⁻³ before training, mean squared error 2.8× lower |
| symbol step, no learning (0.25 m) | 1.1·10⁻² |
| **symbol step + learned correction (0.25 m)** | **7.9·10⁻³** (mean squared error 5.6× lower than without learning) |
| FNO, 4000 iterations (0.25 m) | 5.0·10⁻² |

### Rollout over 3 km (120 steps of 25 m; medians over 40 unseen test scenarios)

| Model | max error / max\|v\| | rel. L2 (all columns) | rel. L2 at 3 km | Screens (s) | March (s) |
|---|---|---|---|---|---|
| free step | 0.93 | 1.01 | 1.13 | – | 0.007 |
| phase-screen split step | 0.84 | 0.98 | 1.11 | 0.008 | 0.007 |
| symbol step, no learning | 0.30 | 0.40 | 0.66 | 1.65 | 0.28 |
| symbol step, 50 m steps, rank 100 | 0.32 | 0.39 | 0.66 | 0.91 | 0.21 |
| **symbol step + learned correction** | **0.22** | **0.29** | **0.51** | 2.16 | 0.25 |
| FNO | 7·10² (diverges) | 4·10² | 2·10³ | – | 0.41 |
| **rational TF reference** (6 scenarios) | ≈10⁻⁴ | ≈10⁻⁴ | ≈10⁻⁴ | – | **2.2–5.3** |

Timings are single-scenario CPU wall times after JIT compilation. For the rational solver, precisions
1e-1, 3e-2 and 1e-2 gave the same grid: it is set by the NLBC and anisotropy bounds of the optimizer,
not by the precision. Its cost is therefore fixed at 2.2–5.3 s for these scenarios.

## 4. Findings

1. **Generic operators fail on this problem.**
   - *Learned split step:* phase screens multiplied by global spectral multipliers cannot represent
     the vertical stretch κ → κ/h_η of the map (r ranges over 0.5–1.9 here), so it stalls at 33 %
     per step.
   - *FNO:* it fits a single step to 5 %, but is nonlinear in the field and does not respect
     energy, so autoregressive rollouts diverge within a few hundred metres. This is the
     instability reported by McCabe et al. (see the survey).
2. **Physics structure gives most of the accuracy.** The frozen-coefficient symbol of the
   flux-conserving step reduces the one-step error by 30–100× relative to the generic models, with
   no training at all. Only this model and its learned extension stay stable over 120 steps.
3. **Learning helps, but only by a factor of about 1.4 in rollout error** (0.40 → 0.29). The learned
   corrections have to be bounded or low-rank. Unconstrained corrections of the 64 basis weights
   made training diverge twice (amplification).
4. **Sub-stepping does not help.** Splitting the 25 m step into 2–5 stages changes the one-step
   error by less than 15 %. The remaining error is not a range-splitting error.
5. **The speed potential is real, but not yet usable.** The march costs 0.25 s for 3 km, against
   2.2–5.3 s for the rational solver, so it is 10–20× faster. The advantage comes from about 9×
   coarser vertical sampling (spectral at 0.25 m instead of fourth-order FD at about 0.03 m) and
   about 7× longer steps. In the present implementation, computing the terrain-dependent weights
   (2.2 s) eats most of it. That cost could be cached per terrain or computed on a coarser grid,
   but the accuracy gap of about 3·10³ would remain.

## 5. Where the remaining error comes from (diagnosis so far)

The window, the absorbing layer and the 0.25 m sampling are not the problem. On flat ground the free
step reproduces the reference to 1.5·10⁻⁴ per step. The symbol step's error of about 5·10⁻³ per
step over hills was traced as follows:

| Candidate cause | Test | Effect |
|---|---|---|
| range sub-stepping / midpoint rule | 1–5 stages, Newton–Cotes averaging | < 15 % |
| flux factor (q_a/q_b)^½ | included in the symbol | none |
| POD rank | 32 → 100 | < 10 % |
| aliasing at 0.5 m | regenerated at 0.25 m | none (error per step not reduced) |
| accuracy of the reference metric tables | 1 m and 0.25 m tables | reference changes by < 3·10⁻⁴ over 3 km |
| left versus symmetric quantization of the symbol | symmetric version | none |
| analytic first-derivative term −Δa′κ/(2q) | added explicitly | worse (the term is not captured correctly) |

The error is spread over all heights and appears in every scenario with relief. The most likely
remaining cause is the η-dependence of the operator. Over one 25 m step a wave moves up to about
15 m vertically, while h_η varies on the 100 m scale of the hills. The frozen-coefficient
(row-local) symbol has no way to represent the curvature of rays in the transformed coordinates.
The learned rank-8 residual captures part of this, which is the 1.4× gain.

## 6. Next steps that could change the conclusion

- **Train on rollouts** (`--rollout R`, already supported): optimize the 3 km error, not the
  one-step error. This is the standard fix in the literature and has not been tried here yet.
- **Higher-rank residual and wider hypernetwork, on a GPU.** On 20 CPU cores, one 4000-iteration
  run took about 1 h.
- **Learned correction of the rational step instead of a spectral step.** Keep the Padé factors
  and the exact NLBC, and learn coefficients of a larger-Δξ / coarser-Δη scheme from the local
  metrics. The survey (Bar-Sinai et al., Kochkov et al.) suggests 4–10× coarser grids are
  realistic. This keeps the transparent boundary, whereas the spectral window needs an absorber.
- **Map to the physical normal coordinate first.** Resampling each column to arc length
  ∫h_η dη would remove the stretch. The symbol would then depend almost only on h_ξ, and its rank
  would drop.
- **Smaller relief or a higher transition layer.** The error scales with the metric gradients,
  which are moderate here (up to 12° slopes, r up to 1.9).

## 7. Files

| File | Content |
|---|---|
| `common.py` | constants, scenario sampling, reference solver call, metric features |
| `gen_data.py` | generates `data_d025/scn_XXXX.npz` (not committed) |
| `models.py` | the models of Section 2 |
| `train.py` | training (one-step or rollout loss; Adam with warm-up and cosine decay); writes `models/<name>.{eqx,json}` |
| `evaluate.py` | rollouts, timings, rational-solver timings; writes `results/*.json` |
| `literature.md` | survey of related work (about 35 entries) |
| `results/` | JSON results quoted above |
| `models/*.json` | training logs and hyperparameters (the weights `*.eqx` are not committed) |

**Reproduce:**

```
python gen_data.py 0 240                      # about 25 min with 6 processes
python train.py --model sym --stages 1 --step 1 --rank 64 --width 48 --n-res 8 \
       --iters 4000 --lr 1e-3 --batch 16 --name symres_s1
python train.py --model fno --step 1 --width 32 --modes 320 --iters 4000 --lr 1e-3 --batch 16 --name fno_s1
python evaluate.py symres_s1 fno_s1 --baselines
python evaluate.py --rational
```

The study depends on `pywaveprop/helmholtz_tf_jax.py` and `pywaveprop/terrain_transform.py`.
Those files are in the working tree but not yet committed.

## 8. Latent encoder–decoder propagators (`latent.py`)

The encoder maps a column u_j(η) to a latent vector s_j. The propagator maps s_j to s_{j+1} and takes
as parameters the step dx (25, 50 or 100 m; one model covers all three) and the metric profiles
log h_ξ, log h_η at 6 columns across the step, which include h_ξh_η. The decoder maps s back to u.

### How compressible are the columns?

POD of 8000 training columns, tested on unseen scenarios (relative L2 below 200 m, median):

| m (complex) | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| reconstruction error | 0.71 | 0.50 | 6.3·10⁻² | 5.1·10⁻³ | 1.1·10⁻⁷ |

The fields are band-limited (|κ| ≲ 3 rad/m in a 256 m window), so a linear latent space of 128–256
complex dimensions holds them almost exactly.

### Models

- **`LinearLatent`** (reduced-order / Koopman model):
  - The encoder and decoder are linear and tied: s = Φᴴu, u = Φs. Φ holds 128 POD modes; it is
    frozen in the runs reported, because training it degraded reconstruction.
  - The propagator is s' = expm(dx L) s, with L = L₀ + Σ_r a_r(metrics, dx) B_r, optionally plus a
    low-rank term U Vᴴ that also comes from the metrics.
  - L₀ is the Galerkin projection of the free-space generator. a, U and V come from a CNN of the
    metric columns and are zero for identity metrics.
  - For fixed metrics the propagator is a semigroup in dx.
- **`NonlinearLatent`**: a CNN encoder and decoder with a 256-dimensional real latent vector plus the
  log-norm of the field, and a residual MLP propagator s' = s + f(s, dx, code(metrics)).

The loss combines reconstruction, decoded prediction and latent consistency, with one step per sample
and dx drawn from {25, 50, 100} m; 4000 iterations.

### Results

| Model | One-step pred. error (val, rms) | 3 km rollout rel. L2 (median, 40 test scenarios), dx = 25 / 50 / 100 m | Prep (s) | March (s) |
|---|---|---|---|---|
| linear, 16 terms, pooled metric code | 0.33 | 0.79 / 0.79 / 0.79 | 1.0 / 0.6 / 0.3 | 0.007 / 0.004 / 0.003 |
| linear, 128 terms + rank 8, positional metric code | 0.32 | 0.79 / 0.78 / 0.77 | 0.75 / 0.49 / 0.29 | 0.005 / 0.003 / 0.002 |
| nonlinear autoencoder | 0.55 (reconstruction 0.34) | diverges (NaN) at all dx | – | 0.8 / 0.4 / 0.2 |
| **oracle**: Galerkin projection of the symbol step (no learning), m = 128 | – | 0.38, 0.32, 1.15 on 3 scenarios; full-space symbol step 0.39, 0.32, 1.17 | | |

**Findings.**
1. **The latent space is not the bottleneck.** With m = 128, the projected physics step reproduces the
   full-space symbol step to within 0.01–0.02. Increasing m to 192 or 256 changes nothing.
2. **The learned latent propagator is the bottleneck.** From 200 scenarios it learns only part of the
   terrain dependence. The rollout error is 0.79 against 0.39 for the physics projection, and about
   1.0 for free space.
   - The linear model's error is the same for dx = 25, 50 and 100 m. The dx conditioning works
     (semigroup); the terrain conditioning does not.
   - Neither 8× more affine terms nor a metric encoder that keeps the vertical position helped within
     4000 iterations.
3. **The nonlinear autoencoder diverges.** It does not reconstruct well (34 %), and the propagated
   log-norm drifts until the decoded field overflows. It is not usable for marching without a
   stability constraint.
4. **The march is extremely cheap.** It takes 3–7 ms for 3 km, against 2.2–5.3 s for the rational
   solver: about 1000× faster. The per-terrain preparation (the 30–120 step matrices expm(dx L))
   takes 0.3–1 s.

**Next step suggested by the oracle.** Use the Galerkin projection of the physics (symbol) step as
the latent propagator and learn only a correction: A = A_phys(metrics, dx) · expm(dx ΔL(metrics, dx)).
This starts from 0.39 rather than 1.0. The obstacle is cost. A_phys needs m applications of the
symbol step per range step, unless it is assembled from a precomputed affine tensor in the symbol
weights.
