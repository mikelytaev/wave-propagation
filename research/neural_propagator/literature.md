# Learned range-step propagators for PE wave propagation: literature survey

Compiled 2026-10-16. Scope: prior work relevant to replacing **one range step** of a wide-angle
(Padé) PE propagator in terrain-following coordinates (phi_n -> phi_{n+1}, conditioned on local
medium/metric coefficients, with a discrete transparent boundary condition) by a learned operator
marched autoregressively, with the aim of being **faster at equal accuracy**.

**How each entry was read** (shown on every entry):
- **[full]**: full text read or skimmed in this session (PDF/HTML).
- **[partial]**: abstract plus parts of the full text (intro/method/selected results).
- **[abstract]**: abstract, metadata or search snippets only. Numbers come from the abstract.
- **[background]**: well-known paper cited from prior knowledge. Metadata was not re-fetched this
  session; check before citing.

Bibliographic data (DOI, volume, pages) was checked against Crossref or the publisher/arXiv page
where noted. I did not invent any reference. Where I am unsure, the entry says so.

---

## A. Learned surrogates for parabolic-equation / PE-type propagation (closest to our setting)

### A1. Xu, Zhang, Zhang (2023). *Training a deep operator network as a surrogate solver for two-dimensional parabolic-equation models.*
JASA 154(5):3276–3284. DOI 10.1121/10.0022460 (checked on Crossref). **[abstract]** (paywalled)
- DeepONet with Fourier features and multiple inputs (complex pressure and sound speed at a depth)
  is trained to approximate the PE **square-root operator** for 2D underwater propagation, which
  avoids the usual rational/Taylor approximation of the cross-derivative term.
- The abstract claims "good accuracy" and "lower computational cost" but gives no numbers. It is
  also unclear whether the network is applied per range step or once.
- **Relevance: very high.** This is the only work found that targets the PE *step operator* itself
  rather than the full field. It is a direct precedent and a baseline to cite and contrast with.

### A2. Zheng, Xia, Wang, Zhang, Duan, Liu, Tang (2025). *A Fourier Neural Operator-enhanced parabolic equation framework for highly efficient underwater acoustic field prediction.*
Front. Mar. Sci. 12. DOI 10.3389/fmars.2025.1692899. **[full]** (HTML via fetch)
- An FNO replaces the split-step Padé part of a RAM-like PE. Its input has six channels (Re/Im
  pressure, wavenumber, density, range, depth). It is trained on RAM v1.5 and outputs complex
  pressure over a block of consecutive range segments, so it is closer to block prediction than a
  one-step marcher.
- Reported: R² > 0.95, RMSE ≤ 0.04 dB on average, and **only 25–35 % lower latency than RAM**
  "at equivalent accuracy".
- Failures and inconsistencies: R² < 0 for a positive sound-speed gradient case. In the deep-sea
  case R² ≈ 0.89–0.91, and the reported timings (3.5 s vs 6.75 s) do not match the stated "2.95×".
- **Relevance: high.** It is the nearest "learned Padé step" attempt. Its modest real speedup over a
  CPU RAM code is a useful sobering datapoint for our speed goal.

### A3. Sun, Cheng, Li, Gerstoft et al. (2025). *Hankel-FNO: Fast Underwater Acoustic Charting via Physics-Encoded Fourier Neural Operator.*
arXiv:2512.06417. **[full]** (HTML via fetch)
- A full-field FNO (4 layers, 64 channels, 64 modes) maps the (r, z) environment directly to the TL
  field in one pass, with no marching. Inputs add a Hankel-amplitude channel |H0(k0 r)| and a
  bathymetry encoding. Training data: RAM at 200 Hz, 3,456 samples, one region and one date.
- At 7.5 km: RMSE 1.43 vs 2.61 for vanilla FNO and 9.59 for OFormer (units unstated). The gain at
  40 km is modest. Accuracy degrades above about 400 Hz. Out-of-distribution
  frequency/depth/bathymetry raises RMSE about 3–8× unless fine-tuned. **No numeric speedup is
  given.**
- **Relevance: medium.** It shows the limits of the full-field approach at long range: performance
  degrades with range and needs fine-tuning out of distribution. That supports the per-step
  formulation we propose.

### A4. Sun, Fang, Zhang, Cheng, Li, Gerstoft (2026). *Mitigating Spectral Bias in Neural Operators for Underwater Transmission Loss Prediction* (S2RL).
arXiv:2608.18141. **[abstract]**
- The FNO "global propagator" stage over-smooths interference fringes because of mode truncation. A
  spatial U-Net "local refiner" then learns the high-frequency residual. Real-data South China Sea
  dataset; "millisecond inference"; no numbers in the abstract.
- **Relevance: medium.** It documents that spectral bias destroys interference detail, which for a
  coherent PE field is a phase-accuracy problem.

### A5. Huang, Qin, Hou, Zhang, Zhang (2025). *A Generalizable Physics-Guided Convolutional Neural Network for Irregular Terrain Propagation.*
IEEE TAP 73(6):3975–3985. DOI 10.1109/TAP.2025.3533739 (checked on Crossref). **[abstract]**
- A CNN surrogate for **split-step PE over irregular terrain** (radio). Prior knowledge from
  deterministic models is embedded in the architecture, and the network is shaped to match terrain
  propagation physics. The authors claim it extrapolates beyond the training terrain and antenna
  configurations.
- Companion papers: ACES-China 2025 (*Physics-Guided CNN Architecture Design for Irregular Terrain
  Propagation Modeling*: an SSPE-inspired parameter-embedding block plus a nested U-Net, with
  validation on measured data), and Yang et al., IEEE IWS 2025 (a CNN driven by two-way SSPE that
  predicts RSS). **[abstract]**
- **Relevance: very high for the application.** These are the closest radio-over-terrain PE
  surrogates. They predict received power along the path or as a field image, not a coherent
  complex field, and they do not march in range.

### A6. Zhang group: CNN/RNN correction of coarse-mesh vector-PE in tunnels (2022–2026)
**[abstract]** for all.
- Wang et al., IEEE AP-S/URSI 2022, *Vector Parabolic Equation Driven CNN Models for Radio Wave
  Propagation in Tunnels*: a CNN compensates the numerical dispersion of **coarse-mesh VPE** and
  recovers fine-mesh RSS, with "significantly reduced simulation time".
- Huang et al., IET MAP 2023 (*Generalisable CNN model for radio wave propagation in tunnels*):
  studies interpolation vs extrapolation. Qin et al., Electronics Letters 2023: a physics-based RNN
  driven by coarse-mesh VPE. Huang et al., IEEE TAP 2024: a CNN maps low-frequency PE results to
  high frequency ("frequency conversion"). Wu et al., IEEE TMTT 2025 (PLAPN): an attention network
  with **sliding-window inference to reduce error accumulation over long range**. Wu et al.,
  arXiv:2601.02007 (2026, PRBPN): recurrent back-projection super-resolution of coarse PWE slices.
- **Relevance: high.** "Coarse PE plus learned correction" is exactly our speed argument (coarser
  grid at equal accuracy). Their range-sequence models report error accumulation as a central issue.
  All of this work is post-hoc on RSS rather than a learned step operator.

### A7. Wessinger, Smith, Gull, Gehman, Beever, Kammerer (2025). *A Deep Learning Framework for Two-Dimensional, Multi-Frequency Propagation Factor Estimation.*
arXiv:2505.15802 (submitted to IEEE). **[abstract]**
- Image-to-image generators map a 2D modified-refractivity field to the pattern propagation factor
  in the marine atmospheric boundary layer, over multiple frequencies. PE simulations are the
  reference. The abstract only says the networks "reasonably predict" the factor.
- **Relevance: medium.** This is the tropospheric-duct analogue of full-field image surrogates. It
  has no terrain and no field phase.

### A8. Mallik, Jaiman, Jelovica (2022). *Predicting transmission loss in underwater acoustics using convolutional recurrent autoencoder network.*
JASA 152(3):1627–1638. DOI 10.1121/10.0013894. **[abstract]**
- A convolutional autoencoder compresses the field and a recurrent net evolves the latent state to
  predict far-field TL in 2D. It captures spreading, refraction and surface/bottom reflections. The
  follow-up by Deo, Venkateshwaran, Jaiman (arXiv:2404.08091, 2024) argues autoregressive models
  struggle at long range and proposes a range-*conditional* CNN with continual learning across
  bathymetry.
- **Relevance: medium.** These are latent-space range marchers. The follow-up is an explicit report
  that autoregressive TL prediction degrades at long range.

### A9. Lytaev (2025). *Neural parabolic wave equation for refractivity estimation.*
ICCS 2025, LNCS. DOI 10.1007/978-3-031-97626-1_6. **[partial]** (first pages)
- A differentiable finite-difference PE in JAX, with the refractive index represented by a neural
  network and solved by autodiff. Inverse problem for tropospheric and underwater tomography.
- **Relevance:** this is our own prior differentiable PE (not a learned propagator). The learned
  stepper must remain differentiable to keep this use case.

---

## B. Neural operators for wave equations more broadly (Helmholtz, seismic, EM, optics)

### B1. Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar (2021). *Fourier Neural Operator for Parametric PDEs.*
ICLR 2021, arXiv:2010.08895. **[background]**
- Global spectral convolution with truncated Fourier modes plus a pointwise linear path. Reported
  speedups of up to about 1000× on Navier–Stokes and Darcy flow at about 1 % relative L2 error.
  These benchmarks are smooth and dissipative.
- **Relevance:** the default baseline. Its Fourier layer is structurally close to an SSF step, but
  its nonlinearity and mode truncation are the wrong inductive biases for a linear, unitary,
  oscillatory propagator.

### B2. Lu, Jin, Pang, Zhang, Karniadakis (2021). *Learning nonlinear operators via DeepONet.*
Nat. Mach. Intell. 3:218–229. DOI 10.1038/s42256-021-00302-5. **[background]**
- Branch/trunk architecture. The output is linear in the trunk basis, with coefficients that are
  nonlinear in the inputs.
- **Relevance:** if the branch is fed only coefficients and the field enters linearly, this gives a
  natural "linear in phi, nonlinear in medium" design.

### B3. Kovachki et al. (2023). *Neural Operator: Learning Maps Between Function Spaces.*
JMLR 24(89). arXiv:2108.08481. **[background]** Also related: U-NO (Rahman, Ross, Azizzadenesheli,
TMLR 2023, arXiv:2204.11127), WNO (Tripura & Chakraborty, CMAME 2023, arXiv:2205.02191) and CNO
(Raonić et al., NeurIPS 2023, arXiv:2302.01178).
- These are the main architecture families. The wavelet NO is attractive for localized boundary
  effects, but one rollout study (MPNO, arXiv:2608.25744, **[abstract]**) reports that WNO diverged
  autoregressively.

### B4. Fanaskov, Oseledets (2023). *Spectral Neural Operators.*
Doklady Mathematics 108 (Suppl 2):S226–S232. DOI 10.1134/S1064562423701107. arXiv:2205.10573.
**[abstract]**
- Inputs and outputs are Chebyshev/Fourier coefficient vectors, with no aliasing and with
  "transparent output". The authors report that it beats FNO and DeepONet on many operators.
- **Relevance: medium–high.** Our vertical z-grid with a nonlocal boundary is non-periodic, so a
  Chebyshev/Legendre or sine basis is more natural than FFT-periodic FNO layers.

### B5. Chen, Liu, Lin, Chen, Shi (2024). *NSNO: Neumann Series Neural Operator for Solving Helmholtz Equations in Inhomogeneous Medium.*
arXiv:2401.13494. **[abstract]**
- A Neumann-series (Born) structure with an embedded U-Net. The field-dependence is linear by
  construction and the network learns the scattering kernel. It reports at least 60 % lower relative
  L2 error than FNO, especially at high wavenumber, about 50 % lower cost, and less data.
- **Relevance: high.** It is an example of building the operator's algebraic structure (series
  expansion) into the network. Analogue for us: learn a perturbation of the rational/Padé step
  rather than the whole step.

### B6. You, Xu, Cai (2024). *MscaleFNO: Multi-scale Fourier Neural Operator Learning for Oscillatory Function Spaces.*
arXiv:2412.20183. **[abstract]**
- Parallel FNOs on rescaled inputs counter spectral bias. Substantial gains over FNO for
  high-frequency Helmholtz scattering (no numbers in the abstract).
- **Relevance:** evidence that plain FNO is biased toward low frequencies, which matters for
  wide-angle content near the PE angular limit.

### B7. Stanziola, Arridge, Cox, Treeby (2021). *A Helmholtz equation solver using unsupervised learning: application to transcranial ultrasound.*
JCP 441:110430. DOI 10.1016/j.jcp.2021.110430. **[abstract]**
- A learned iterative optimizer (a UNet with hidden state) is trained with a physics residual loss,
  not reference data. It generalizes to larger domains and CT-derived skull speeds.
- **Relevance:** physics-loss (residual-based) training without reference data, and a "learned
  iteration" alternative to a learned direct map.

### B8. Yang, Gao, Ross, Azizzadenesheli, Clayton (2021). *Seismic wave propagation and inversion with neural operators.*
The Seismic Record 1(3) (venue per the authors' PDF filename "Yang2021TSR"; DOI not verified).
**[abstract]**
- FNO trained on random velocity models and sources for 2D acoustic waves, used for tomography
  through autodiff. About 10× faster than conventional FWI.
- Related: Zou, Azizzadenesheli, Ross, Clayton (2024, Caltech), *Deep Neural Helmholtz Operators
  for 3D Elastic Wave Propagation and Inversion* (40× less memory than the time-domain version,
  about 100× faster than a spectral-element method) **[abstract]**. Lehmann, Gatti, Bertin,
  Clouteau (2023), 3D FNO for seismic ground motion, arXiv:2304.10242, related DOI
  10.1016/j.cma.2023.116718 **[abstract]**.
- I found **no** neural-operator work on the seismic *one-way* (depth-marching) wave equation. The
  closest is CNN post-correction of one-way migration images toward RTM (Yu et al., Geophysics
  2023, **[abstract via snippet]**; details unverified).

### B9. Voytan, Li (2025). *Are Fourier Neural Operators Really Faster for Time-Domain Wave Propagation?*
arXiv:2508.11119. **[abstract]** (PDF downloaded; only the abstract was used)
- Inference-only benchmark of a Tucker-FNO against GPU finite differences for 4 wave physics. On the
  **same grid FNO is 1.4–80× slower**. On a 2.5× coarser grid, small FNOs are up to 10× faster, but
  larger (accurate) FNOs range from 1.09× faster to 5.9× slower. Only boundary-only outputs give
  about 4000×.
- **Relevance: very high for our speed claim.** A learned stepper must beat an **optimized** JAX
  Padé step on GPU, not a CPU reference code. The win has to come from larger dx or a coarser z
  grid, not from the network being cheap per point.

### B10. Augenstein, Repän, Rockstuhl (2023). *A neural operator-based surrogate solver for free-form electromagnetic inverse design.*
ACS Photonics (arXiv:2302.01934). **[abstract]** (venue from memory: **unverified**) Also:
Pornthisan, Markidis (2023), *Fast Electromagnetic Field Pattern Calculation with Fourier Neural
Operators*, ICCS 2023, DOI 10.1007/978-3-031-36021-3_24 **[partial]**: FNO is 31× faster than Meep
on a laptop CPU "at the cost of a small accuracy loss".
- **Relevance:** frequency-domain EM with complex fields. The FNO kernel is complex-valued, but the
  activations are real (Re/Im split).

### B11. Jiang, Salvadori, Ota, Shankar, Shukla (2024/2026). *Complex Valued Deep Operator Network for Three Dimensional Maxwell's Equations.*
arXiv:2411.18733. **[abstract]**
- A complex-domain forward pass with redefined affine maps, nonlinearities and backprop, and a
  reparametrization for training. It combines several DeepONets per field component to enforce the
  BC.
- **Relevance:** among the few explicitly complex-valued operator networks for EM fields.

### B12. Duruisseaux, Gray, Jiang, Zhou, George, Azizzadenesheli, Marandi, Anandkumar (2025). *Fourier Neural Operators for Fast Simulation and Inverse Design of Second-Harmonic Generation in Nanophotonics.*
Listed on the NeurIPS 2025 virtual site (track and archival status not verified; I found no arXiv
id). **[abstract]**
- FNO trained on a **Fourier split-step** (BPM-type) solver for waveguide SHG. About 4 % relative L2
  error and about 550,000× speedup in batched GPU mode, used for gradient-based inverse design.
- **Relevance:** the clearest "neural operator replaces split-step" result, but it predicts
  end-to-end (no marching) and its speedup counts batching against a per-sample solver. Not an
  equal-accuracy comparison.

---

## C. Learned steppers, learned split-step and structure-preserving propagators

### C1. Kamilov, Papadopoulos, Shoreh, Goy, Vonesch, Unser, Psaltis (2015). *Learning approach to optical tomography.*
Optica 2(6):517–522. DOI 10.1364/OPTICA.2.000517. **[abstract]**
- The **BPM (split-step paraxial) is written as a multilayer network** whose weights are the voxel
  refractive indices, and it is trained by error backpropagation on holographic measurements. The
  forward model is fixed physics and the learning is over the medium.
- **Relevance:** the origin of the "unrolled BPM = network" idea that our differentiable PE also
  uses. The weights are physical, so this is not a speed surrogate.

### C2. Häger, Pfister (2021). *Physics-Based Deep Learning for Fiber-Optic Communication Systems.*
IEEE JSAC 39(1):280–294. DOI 10.1109/JSAC.2020.3036950. **[abstract]**
- The split-step method for the NLSE is parameterized: the linear (dispersion) steps become
  **trainable short FIR filters** ("learned digital backpropagation"). The filters prune to **3
  taps/step** with no loss of performance, cutting complexity by orders of magnitude relative to
  standard SSFM-based DBP with **few, large steps**.
- Follow-ups: Lin et al., JLT 2021, perturbation-aided LDBP with multiple spans per step.
- **Relevance: very high conceptually.** This is the strongest evidence that a **learned
  linear-step operator, structured like the classical scheme, allows far fewer and cheaper steps at
  equal accuracy**, which is exactly our goal. It is also linear in the field and interpretable.

### C3. Nguyen, Tsai (2023). *Numerical wave propagation aided by deep learning.*
JCP 475:111828. DOI 10.1016/j.jcp.2022.111828 (checked on Crossref). arXiv:2107.13184.
**[partial]** (intro and method read)
- A CNN corrects a cheap **coarse-grid, large-time-step** solver toward a fine solver for the 2D
  wave equation in multiscale media: u_{n+1} = F(IRu_n) + [learned correction]. The paper uses a
  wave-energy-based loss and samples training data to respect causality. The learned correction
  sharply reduces numerical dispersion and the loss of high-frequency energy. Combined with parareal
  for stability.
- Follow-up: Kaiser, Tsai, Klingenberg (2025), ENUMATH 2023 proceedings, LNCSE, DOI
  10.1007/978-3-031-86169-7_2, arXiv:2402.02304 **[partial]**: end-to-end training through
  multiple steps, which improves accuracy "without sacrificing speed".
- **Relevance: very high.** This is the closest methodological template: a classical coarse step
  plus a learned correction, conditioned on the medium, marched in time, with energy-aware loss.

### C4. Bar-Sinai, Hoyer, Hickey, Brenner (2019). *Learning data-driven discretizations for PDEs.*
PNAS 116(31):15344–15349. DOI 10.1073/pnas.1814058116 (author list not re-verified). Also: Kochkov
et al. (2021), *Machine learning-accelerated computational fluid dynamics*, PNAS 118(21):e2101784118.
**[abstract]**
- Learned stencil coefficients inside a conservative solver give 4–8× coarser grids in 1D. Learned
  corrections inside a CFD solver match baselines with 8–10× finer resolution, giving a **40–80×
  speedup at equal accuracy**.
- **Relevance: high.** This is the cleanest "equal accuracy" methodology. Keeping the physics
  structure (flux form, or for us the Padé/rational form) and learning only its coefficients is the
  most reliable route to honest speedups.

### C5. Patel, Subedi, Tewari (2025/2026). *Operator Learning for Schrödinger Equation: Unitarity, Error Bounds, and Time Generalization.*
arXiv:2505.18288. **[partial]**
- A **linear** estimator of the TDSE evolution operator that is exactly unitary on the queried
  subspace and non-expansive overall ("weak unitarity"), learned from actively chosen queries. It
  has upper and lower error bounds and time-generalization bounds. It is up to 100× lower relative
  error than FNO and DeepONet.
- **Relevance: high.** The PE in z is a Schrödinger-type evolution in range. This supports
  exploiting linearity and (near-)unitarity. With our TBC and absorption the step is
  **non-expansive**, not unitary, which is the right constraint to impose.

### C6. Melchers, Prins, Abdelmalik (2024). *Neural Green's Operators for Parametric Partial Differential Equations.*
arXiv:2406.01857. **[partial]**
- The output is **exactly linear in the inhomogeneity/data field**, while a network maps
  (weighted averages of) PDE coefficients to the Green's function. It generalizes better out of
  distribution than DeepONet, VarMiON and FNO.
- **Relevance: high.** This is the "linear in phi, nonlinear in coefficients" pattern. Our step is
  phi_{n+1} = G[c_n] phi_n with G a matrix-valued function of local coefficients.

### C7. Li et al. (2026). *SGNO: Spectral Generator Neural Operators for Stable Long Horizon PDE Rollouts.*
arXiv:2602.18801. **[abstract]**
- An exponential-time-differencing step in Fourier space with a **learned diagonal generator whose
  real part is constrained ≤ 0**, so iteration cannot amplify the linear part. It derives one-step
  amplification and finite-horizon error bounds. Longer stable rollouts on APEBench.
- Related: Symplectic Neural Operators (arXiv:2605.15881), NIGO (skew-symmetric plus NSD latent
  generator, ICML 2026), and Conservation-Constrained FNO (ICML 2026: hard projection beats soft
  penalties). All **[abstract]**.
- **Relevance: high.** This is a concrete recipe for a step operator that is non-expansive by
  construction, and maps directly onto exp(i dx (sqrt(1+L) - 1)) or a Padé form.

### C8. Mao, Chimitt, Chan (2021). *Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform.*
ICCV 2021, arXiv:2107.11627. **[abstract]**
- Replaces split-step phase-screen propagation for imaging through turbulence. **300–1000×
  speedup** while preserving turbulence statistics (statistical, not deterministic, accuracy).
- **Relevance: low–medium.** It shows that large speedups are easy when only statistics must match,
  which is not our setting.

### C9. Xie, Chen, Li (2021). *A Machine-Learning Method for Time-Dependent Wave Equations over Unbounded Domains.*
arXiv:2101.05807. **[abstract]**
- Learns the map from compactly supported initial data to the solution on a truncated domain,
  **as an alternative to artificial/absorbing BCs**. Trained on wave packets (bandwidth,
  wavenumber), tested on the wave equation and (nonlinear) Schrödinger. Good interpolation, limited
  extrapolation.
- **Relevance: high for item 3.** This is one of the very few learned-ABC works for
  Schrödinger-type equations.

---

## D. Rollout stability, error accumulation and honest benchmarking

### D1. Brandstetter, Worrall, Welling (2022). *Message Passing Neural PDE Solvers.*
ICLR 2022, arXiv:2202.03376. **[abstract]**
- The **pushforward trick** (train on the model's own one-step-ahead outputs, without backprop
  through the first step) and **temporal bundling** (predict K steps per call) reduce
  train/test distribution shift in rollouts.
- **Relevance:** standard training tricks for our marcher. Bundling corresponds to "one network
  call = several Padé steps".

### D2. McCabe, Harrington, Subramanian, Brown (2023). *Towards Stability of Autoregressive Neural Operators.*
TMLR (11/2023), arXiv:2306.10619. **[partial]**
- Traces FNO rollout blow-up to **aliasing from pointwise nonlinearities and energy pile-up near
  Nyquist**, like pseudospectral instabilities. Fixes: **frequency-domain spectral normalization**
  of the spectral filters (polar parameterization with a squashed magnitude), learned filtering,
  and depthwise-separable spectral convolutions. Up to 800 % longer stable forecasts.
- **Relevance: very high.** Spectral normalization with |multiplier| ≤ 1 is exactly the
  non-expansiveness we want for a PE step.

### D3. Lippe, Veeling, Perdikaris, Turner, Brandstetter (2023). *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers.*
NeurIPS 2023, arXiv:2308.05732. **[abstract]**
- Rollout failure is attributed to **neglect of low-amplitude, non-dominant spatial frequencies**,
  fixed with diffusion-style multi-step refinement (which also gives uncertainty).
- **Relevance:** in PE fields the low-amplitude parts (shadow zones, diffracted field behind
  terrain) are exactly what matters for path loss. An MSE loss in linear amplitude will ignore
  them, so use a dB or relative per-mode loss.

### D4. Ye et al. (2025). *Recurrent Neural Operators: Stable Long-Term PDE Prediction.*
arXiv. **[abstract]** Also: Nayak et al. (2025), *TI-DeepONet* (learn the derivative and integrate
with a classical scheme; 96 % lower extrapolation error than autoregressive training); Roy et al.
(2025), *ANCHOR* (residual-monitored hybrid that falls back to the numerical solver).
- Recurrent (unrolled) training reduces worst-case error growth from exponential to linear in the
  number of steps. Learning the generator rather than the step stabilizes long horizons. Hybrid
  fallback gives per-trajectory error control.
- **Relevance:** training through many range steps is cheap for us (a linear step, JAX), and an
  ANCHOR-style fallback to the exact Padé step is an easy safety net.

### D5. McGreivy, Hakim (2024). *Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related PDEs.*
Nat. Mach. Intell. 6(10):1256–1269. DOI 10.1038/s42256-024-00897-5. **[abstract]**
- 79 % (60/76) of the surveyed ML-for-PDE papers claiming to beat numerical methods used **weak
  baselines**. Reporting biases are widespread.
- **Relevance: very high for how we report.** Compare against the best classical setting
  (optimized Padé order, dx, dz and TBC on the same GPU), at matched error, on a Pareto plot of
  error vs wall time.

### D6. Seretis, Sarris (2022). *An Overview of Machine Learning Techniques for Radiowave Propagation Modeling.*
IEEE TAP, DOI 10.1109/TAP.2021.3098616, arXiv:2101.11760. **[abstract]** Also: Levie, Yapar,
Kutyniok, Caire (2021), *RadioUNet*, IEEE TWC 20(6):4001–4015 (whole urban radio map in 1–10 ms,
about 1 dB RMSE vs ray tracing) **[abstract]**.
- Survey of radio-propagation ML. Almost all of it is path-loss regression or full-map
  image-to-image, with no learned field propagator.
- **Relevance:** background, and support for the claim that a coherent learned PE step is a gap in
  radio propagation.

---

## E. Learned boundary conditions / transparent boundaries

The literature is **thin**.
- **Xie, Chen, Li 2021** (C9): learned solution map in place of an ABC for Schrödinger-type
  equations.
- **LightGBM absorbing BC for a wave-equation meshless method** (arXiv:2505.05781,
  **[abstract]**): an ML model predicts boundary field values each time step.
- **PINN-with-PML** variants (Gabor-basis PINNs for scattered wavefields): not steppers.
- **Rodionov**, *Operator learning for constructing transparent BC for hyperbolic equations*
  (Innopolis University slides, **[snippet only; content unverified]**).
- **NGO and complex DeepONet** (C6, B11) enforce BCs by construction rather than learning them.
- I found **no** paper that learns a discrete transparent (convolution-in-range) BC for a PE
  inside a neural range stepper. The classical discrete TBC theory (Arnold, Ehrhardt; review by
  Antoine, Arnold, Besse, Ehrhardt, Schädle, CiCP 4 (2008) 729–796, **[background]**) remains the
  reference.

---

## Synthesis

### (a) Most promising architecture choices for a linear, complex, range-marching propagator with variable coefficients

1. **Keep the step exactly linear in phi and make only the operator depend nonlinearly on the
   medium.** Write phi_{n+1} = G_theta[c_n, metric_n] phi_n, with a network (hypernetwork) that
   outputs the operator's parameters: NGO (C6), DeepONet branch on coefficients (B2), and the
   learned-filter view of Häger–Pfister (C2). This keeps superposition, so a source or antenna
   pattern is just a different initial column, and it allows exact linear-algebra bookkeeping of
   energy.
2. **Learn a structured perturbation of the classical step, not a black box.** Options:
   (i) learn the Padé coefficients a_j(x), b_j(x) and/or local corrections of the tridiagonal
   operators as functions of local coefficients, giving a "learned Padé step" for large dx,
   following C2, C4 and B5; (ii) a coarse classical step plus a learned residual (C3, A6). Both
   reuse the existing JAX tridiagonal solver, so the cost per step stays near the classical one and
   the speed gain comes from larger dx and coarser dz. That is the only regime where Voytan & Li
   (B9) and Kochkov et al. (C4) report real wins.
3. **Impose non-expansiveness by construction.** Options: a generator with Re ≤ 0 (SGNO, C7),
   spectral normalization |multiplier| ≤ 1 (D2), or a Cayley/Padé form whose rational symbol is
   unitary for real arguments (the [m/m] Padé of exp is already "A-stable"). Exact unitarity is
   wrong because the TBC and ground losses dissipate. Non-expansive is right (C5).
4. **Use complex arithmetic natively.** Use complex weights and a basis suited to a non-periodic z
   domain with an impedance boundary below and a TBC above (sine/Chebyshev in the style of SNO, B4,
   or banded/local operators), rather than periodic FFT-FNO with real Re/Im channels. Use no
   pointwise nonlinearity on phi at all (this also removes the aliasing source in D2).
5. **Train through many steps.** Use recurrent/unrolled training (D4), the pushforward trick (D1),
   and a phase-sensitive, dB-aware loss (relative complex L2 per column plus a log-amplitude term)
   so shadow-zone and low-amplitude modes are not ignored (D3, A4).

### (b) Known pitfalls

- **Error accumulation and instability in rollouts.** Exponential growth under teacher forcing (D4).
  Aliasing and Nyquist pile-up in FNO (D2). WNO divergence (B3). Range-sequence models in tunnels
  needed sliding windows (A6), and Deo et al. report autoregressive TL models failing at long range
  (A8).
- **Spectral bias and phase errors.** Truncated-mode FNOs smooth interference fringes (A4, B6).
  Hankel-FNO used amplitude only because phase varies too quickly (A3). For a coherent PE, a small
  per-step phase-speed error accumulates linearly into large phase errors at 100+ km. Any
  evaluation must report complex-field error, not only TL.
- **Generalization to unseen media.** FNO-PE failed (R² < 0) for an unseen sound-speed gradient
  (A2). Hankel-FNO needed fine-tuning for new frequency, depth or bathymetry (A3). The terrain CNNs
  stress that out-of-distribution terrain is the key challenge (A5). Local, per-step conditioning on
  *local* coefficients is the strongest argument that a stepper generalizes better than full-field
  maps: terrain shapes are unseen, but local slopes and curvatures are not.
- **Illusory speedups.** Weak baselines (D5). Batch-vs-single and GPU-vs-CPU comparisons (B12, A2).
  Same-grid FNO slower than FD (B9). The real speedup in A2 is only 25–35 %.
- **Boundary handling.** A learned step that sees only the computational window cannot represent
  the nonlocal-in-range TBC memory unless the TBC history is an explicit input or is kept classical.
  No precedent exists (Section E).

### (c) Gaps our work could fill

1. **First learned *range-step* operator for tropospheric/radio PE over terrain** that outputs the
   complex field. Existing radio work (A5–A7, D6) predicts RSS, PPF or path-loss images, and the
   only step-operator precedent is underwater DeepONet (A1), which has no terrain and no speed
   comparison at equal accuracy.
2. **Learned large-step wide-angle propagator in terrain-following coordinates.** Condition on
   metric terms, linear in phi, non-expansive by construction, and learned as a correction to Padé.
   This combines C2/C4 (learned structured steps) with C5–C7 (structure preservation), which has
   not been done for PE.
3. **Coupling a learned interior step with an exact discrete TBC**, or learning a compressed
   (sum-of-exponentials) TBC kernel for large dx. This is essentially open (Section E).
4. **Honest Pareto benchmarking.** Error vs wall time against an optimized JAX Padé solver with
   tuned order, dx, dz and TBC on the same hardware, reporting complex-field relative L2 and dB
   error after 100+ km, including phase. This directly answers D5 and B9.
5. **A differentiable, fast propagator for inversion.** Tie-in with refractivity inversion (A9):
   a learned stepper that keeps exact linearity and differentiability could accelerate the inverse
   problem.

---

## Search notes

- Tools: WebSearch, Consensus semantic search, WebFetch of arXiv, Crossref, Frontiers and other
  pages. arXiv PDFs and the ICCS PDFs were read locally as text with pymupdf.
- Searches with **no relevant hits**: FNO or neural operators for the *seismic one-way* wave
  equation; a learned *discrete transparent BC* for PE/Schrödinger inside neural steppers; neural
  operators applied directly to the *tropospheric radio PE* with complex-field output.
- Items needing verification before citing: B8 venue/DOI, B10 venue, B12 venue/arXiv, C4 author
  list, the Rodionov slides, and the Yu et al. 2023 Geophysics entry.
