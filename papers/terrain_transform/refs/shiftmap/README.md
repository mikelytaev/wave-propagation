# Shift-map (Beilis–Tappert) finite-difference PE over irregular terrain: survey of recent work (2010–2026)

Compiled 2026-10-12 for the TAP paper "terrain-following higher-order rational PE with discrete NLBC". This file extends sections A, D and E of `../README.md` and does not repeat them. Papers that are already in `../refs.bib` are cited here by their existing keys (marked "†"). Some of them get more detail below, taken from full texts that were not available before or from papers that reproduce their schemes. Every other key is in `shiftmap.bib`, and none of those keys clash with `../refs.bib`.

**Sources and their reliability**
- Downloaded files are untrusted third-party data. Text was extracted with `pypdf` under `python -I`, and nothing in the files was executed.
- IEEE, JASA/AIP, Wiley/Hindawi (2024+), MDPI and TechRxiv blocked scripted downloads from this host (Cloudflare or SSL reset). Hindawi and Atlantis PDFs were taken from Wayback snapshots where they existed.
- Each note is tagged with what it is based on:
  - **[full]**: the full text was read.
  - **[abstract]**: only the publisher abstract (Crossref/OpenAlex) was read.
  - **[secondary]**: reconstructed from a paper by the same group that reproduces the scheme. These notes say which paper.
- The Chinese Journal of Radio Science (电波科学学报) is not indexed in Crossref under English metadata, and web searches turned up no shift-map FD papers there. That venue is therefore **not covered**. The same groups (Long/SYSU, Xi/XUT, Wu/Xidian) publish their main results in IEEE TAP and AWPL, and those papers are covered.

---

## 1. Table

| key | authors | year | title | venue | DOI | file |
|---|---|---|---|---|---|---|
| holm2007wasm† | P. D. Holm | 2007 | Wide-angle shift-map PE for a piecewise linear terrain — a finite-difference approach | IEEE TAP 55(10):2773–2789 | 10.1109/TAP.2007.905865 | DOI only |
| zhou2015greene† | C. Zhou, Q. Zeng, W. Ding, Y. Long | 2015 | Greene second-order approximation of the wide-angle equation for terrain | APMC 2015 | 10.1109/APMC.2015.7413341 | DOI only |
| guo2017greene† | Q. Guo, C. Zhou, Y. Long | 2017 | Greene approximation wide-angle PE for radio propagation | IEEE TAP 65(11):6048–6056 | 10.1109/TAP.2017.2748222 | DOI only |
| guo2017pade† | Q. Guo, Y. Long | 2017 | Padé second-order PE modeling for propagation over irregular terrain | IEEE AWPL 16 | 10.1109/LAWP.2017.2749603 | DOI only |
| guo2018twoway† | Q. Guo, Z. Li, Y. Long | 2018 | Two-way PE modeling for radio propagation over irregular terrain | ICMMT 2018 | 10.1109/ICMMT.2018.8563284 | DOI only |
| guo2020roughsea2w | Q. Guo, Y. Long | 2020 | Two-way PE method for radio propagation over rough sea surface | IEEE TAP 68(6):4839–4847 | 10.1109/TAP.2020.2975288 | DOI only |
| guo2021idpe | Q. Guo | 2021 | Inverse diffraction PE model based on FDM for passive location on irregular terrain | ICICAS 2021, 90–94 | 10.1109/ICICAS53977.2021.00025 | DOI only |
| guo2023forestry | Q. Guo, Y. Lan, X. Deng, Y. Diao | 2023 | Application of the FD PE model in forestry remote telemetry | Int. J. Antennas Propag. 2023:6825971 | 10.1155/2023/6825971 | `guo_2023_ijap_fdpe_forestry.pdf` |
| ding2016fractalsea | W. Ding, K. Wang, Y. Long | 2016 | EM propagation over the improved fractal sea surface based on PE | IJAP 2016:8235752 | 10.1155/2016/8235752 | `ding_2016_ijap_fractal_sea_pe.pdf` |
| zhang2023ibcwasm | N. Zhang, Y. Long | 2023 | New difference schemes for impedance boundary conditions of WA shift-map PE | IEEE TAP 71(10):8378–8383 | 10.1109/TAP.2023.3266061 | DOI only |
| zhang2024wasm† | N. Zhang, Z. Liang, Y. Long | 2024 | An efficient difference scheme for wide-angle shift-map PE | IEEE TAP 72(4):3545–3552 | 10.1109/TAP.2024.3363496 | DOI only |
| zhang2024techrxiv† | N. Zhang, Y. Long | 2024 | New shift-map PEs based on finite difference method | TechRxiv | 10.36227/techrxiv.170619205.59357076/v1 | `zhang_long_2024_techrxiv_new_smpe.html` (archived abstract page; PDF blocked) |
| zhang2026qoperator | N. Zhang, Y. Long | 2026 | A new error analysis based on the Q-operator for shift-map PE and application to long-range propagation | IEEE AWPL (early access) | 10.1109/LAWP.2026.3739961 | DOI only (no abstract available) |
| wu2023fdpade3d | X. Wu, Z. Li, Z. Liang, Y. Long | 2023 | Higher order FD-Padé scheme for 3-D PE in radio-wave propagation | IEEE AWPL 22(6):1251–1255 | 10.1109/LAWP.2023.3238474 | DOI only |
| wu2023ihaadi | X. Wu, Z. Li, Z. Liang, Y. Long | 2023 | An iterative high-accuracy ADI method for the 3D PE | IJAP 2023:9955888 | 10.1155/2023/9955888 | DOI only (Wiley blocked) |
| wang2012twoway† / ozgun2014comments / wangk2014reply | K. Wang, Y. Long; O. Ozgun et al. | 2012/2014 | Improved 2W-PE over irregular terrain (boundary shift), comment and reply | IEEE TAP 60(9); 62(7) | 10.1109/TAP.2012.2207063; 10.1109/TAP.2014.2328018; 10.1109/TAP.2014.2327665 | DOI only |
| wang2016loranasf | D.-D. Wang, X.-L. Xi, Y.-R. Pu, J.-F. Liu, L.-L. Zhou | 2016 | PE method for Loran-C ASF prediction over irregular terrain | IEEE AWPL 15:734–737 | 10.1109/LAWP.2015.2471079 | DOI only |
| zhou2017lfhighangle | L. Zhou, D. Wang, Z. Mu, X. Xi, L. He | 2017 | LF radio wave prediction at short ranges with high propagation angles over irregular terrain | IEEE AWPL 16:732–735 | 10.1109/LAWP.2016.2601138 | DOI only |
| zhouz2017msam | Z. Zhou, D. Wang, Y. Pu, X. Xi | 2017 | Accuracy of PE method for modeling LF EM wave propagation | MSAM 2017 (Atlantis AISR 132) | 10.2991/msam-17.2017.1 | `zhou_2017_msam_lf_pe_accuracy.pdf` |
| wang2018pulseasf | D.-D. Wang et al. | 2018 | Pulse PE method for Loran-C ASF prediction over irregular terrain | IEEE AWPL 17(1):168–171 | 10.1109/LAWP.2017.2778736 | DOI only |
| wang2019hybrid† | D.-D. Wang, X.-L. Xi, Y.-R. Pu, J.-S. Zhang, L.-L. Zhou | 2019 | LF groundwave over irregular terrain by hybrid 2W-PE | IEEE TAP 67(10):6596–6601 | 10.1109/TAP.2019.2925155 | DOI only |
| wang2020smpe† | D.-D. Wang, Y.-R. Pu, X.-L. Xi, L.-L. Zhou | 2020 | An analysis of narrow-angle and wide-angle shift-map PEs | IEEE TAP 68(5):3911–3918 | 10.1109/TAP.2020.2963903 | DOI only |
| zhangr2024maxangle† | R. Zhang, H. Wang, Z. Wu, L. Lin, Z. Zhao | 2024 | Prediction method of maximum propagation angle in PE model over irregular terrain | IJAP 2024:4988555 | 10.1155/2024/4988555 | DOI only (Wiley blocked) |
| gao2019roughsea† | Y. Gao, Q. Shao, B. Yan, Q. Li, S. Guo | 2019 | PE modeling of EM propagation over rough sea surfaces | Sensors 19(5):1252 | 10.3390/s19051252 | `gao_2019_sensors_rough_sea.pdf` |
| gilbert2022wabt | K. E. Gilbert | 2022 | Wide-angle formulation of the Beilis–Tappert method | JASA 152(2):1170–1179 | 10.1121/10.0013727 | DOI only |
| gilbert2018exactbt | K. E. Gilbert, X. Di, C. R. Hart | 2018 | Exact wide-angle formulation of the Beilis–Tappert method | JASA 143(3 Suppl):1740 (abstract) | 10.1121/1.5035679 | DOI only |
| parakkal2011jasaabs (+ parakkal2012beilis†) | S. Parakkal, K. E. Gilbert, X. Di | 2011/2012 | BTPE for long-range sound propagation over irregular terrain | JASA 130(4 Suppl):2438; JASA 131 | 10.1121/1.3654779; 10.1121/1.3676693 | DOI only |
| khodr2020iterative† (+ khodr2018jasaabs, khodr2021jasaabs) | C. Khodr, M. Azarpeyvand, D. Green | 2020 | Iterative 3D PE solver above irregular boundaries | JASA 148(2):1089–1100 | 10.1121/10.0001766 | `../khodr_2020_3dpe_terrain_bristol.pdf` |
| nyborg2023† | C. M. Nyborg, K. Bolin, I. Karasalo, A. Fischer | 2023 | Inter-model comparison of PE methods (BTPE vs GTPE) for wind turbines | JASA 154(2):1299–1314 | 10.1121/10.0020562 | DOI only (AIP blocked) |
| huang2026gass | X. Huang, A. Chabory, R. Douvenot, H. Zhou | 2026 | Propagation modeling over a piecewise terrain in 3D using the shift map method including cross terms | URSI GASS 2026 | 10.46620/ursigass26/tfnw7844 | `huang_2026_ursigass_shiftmap3d.pdf` |
| ligny2021phasecorr | L. Ligny, Z. El Ahdab, R. Douvenot | 2021 | Phase correction for long-distance propagation using split-step methods in nonrectangular domains | IEEE AWPL 20(12):2476–2480 | 10.1109/LAWP.2021.3115256 | DOI only |
| ozbayat2011abc | S. Ozbayat, R. Janaswamy | 2011 | Effective local absorbing BCs for an FD implementation of the PE | IEEE TAP 59(5):1616–1625 | 10.1109/TAP.2011.2122300 | DOI only |
| ozbayat2019diss | S. Ozbayat | 2019 | Techniques to increase computational efficiency in deterministic and random EM propagation problems | PhD, UMass Amherst | 10.7275/dfk6-fm12 | DOI only (repository blocked) |
| zhangp2014window | P. Zhang, L. Bai, Z. Wu, F. Li | 2014 | Effect of window function on absorbing layers top boundary in PE | APCAP 2014, 849–852 | 10.1109/APCAP.2014.6992632 | DOI only |
| zhangp2016review | P. Zhang, L. Bai, Z. Wu, L. Guo | 2016 | Applying the PE to tropospheric groundwave propagation: a review | IEEE APM 58(3):31–44 | 10.1109/MAP.2016.2541620 | DOI only |
| wangr2017dnlbc3d | R. Wang, G. Lu, R. Zhang, W. Xu | 2017 | Improved diffraction NLBC + 3D WA-PE decomposition | IJAP 2017:2728380 | 10.1155/2017/2728380 | `wang_2017_ijap_dnlbc_3dwape.pdf` |
| li2016troposcatter | L. Li, L.-K. Lin, Z.-S. Wu, Z.-W. Zhao | 2016 | Maximum calculation height and maximum propagation angle of troposcatter WAPE | IET MAP 10(6):686–691 | 10.1049/iet-map.2015.0293 | DOI only |
| zhang_sarris2015† | X. Zhang, C. D. Sarris | 2015 | Error analysis and comparative study of numerical methods for the PE (tunnels) | IEEE TAP 63(7):3025–3034 | 10.1109/TAP.2015.2421974 | DOI only |
| zhangx2014adi | X. Zhang, C. D. Sarris | 2014 | High-accuracy ADI scheme for the vector PE (tunnels) | IEEE AWPL 13:650–653 | 10.1109/LAWP.2014.2313737 | DOI only |
| zhangx2014masc | X. Zhang | 2014 | Development, analysis, and validation of PE/ray-tracing techniques in railway environments | MASc thesis, Univ. Toronto | — (URL in bib) | `zhang_x_2014_tspace_railway_pe_thesis.pdf` |
| qin2024sfft2w | H. Qin, S. An, X. Zhang | 2024 | Efficient 2W-PE with sparse Fourier transform over irregular terrain | PIERS 2024 | 10.1109/PIERS62282.2024.10618483 | DOI only |
| qin2025sspeterrain | H. Qin, Y. Mu, S. Huang, X. Zhang, X. Zhang | 2025 | Modeling radio wave propagation over irregular terrain via SSPE | PIERS-Fall 2025 | 10.23919/PIERS-Fall62445.2025.11394536 | DOI only |
| huang2025pgcnn | S. Huang, H. Qin, W. Hou, X. Zhang, X. Zhang | 2025 | Physics-guided CNN for irregular terrain propagation | IEEE TAP 73(6):3975–3985 | 10.1109/TAP.2025.3533739 | DOI only |
| an2026apce | S. An, L. Codecasa, H. Qin, X. Zhang, L. Di Rienzo | 2026 | Adaptive PCE for UQ of radio propagation over irregular terrains | IEEE TAP (early access) | 10.1109/TAP.2026.3731420 | DOI only |
| magno2011imoc | F. N. B. Magno et al. | 2011 | Path loss in mixed-path environment using NA and WA PEs | IMOC 2011, 166–170 | 10.1109/IMOC.2011.6169363 | DOI only |
| lytaev2019icumt | M. S. Lytaev, A. G. Vladyko | 2019 | Comparative analysis of PE method and Longley–Rice model | ICUMT 2019 | 10.1109/ICUMT48472.2019.8970714 | DOI only |
| lytaev2020electronics | M. Lytaev, E. Borisov, A. Vladyko | 2020 | V2I propagation loss in simplified urban environment: a two-way PE approach | Electronics 9(12):2011 | 10.3390/electronics9122011 | DOI only (MDPI blocked) |
| vavilov2020knifeedge | S. A. Vavilov, M. S. Lytaev | 2020 | Modeling equation for multiple knife-edge diffraction | IEEE TAP 68(5):3869–3877 | 10.1109/TAP.2019.2957085 | DOI only |
| lytaev2024piersuav | M. S. Lytaev | 2024 | UAV radio relay in complex propagation conditions using the PE method | PIERS 2024 | 10.1109/PIERS62282.2024.10618264 | DOI only |

---

## 2. Reference formulation (what all of these papers share)

This section collects the equations so that a "faithful" NA/WA shift-map FD reference implementation can be written once. Notation: 2-D, $e^{-i\omega t}$, $\psi=e^{ikx}u$, terrain $z=T(x)$, $\zeta=z-T(x)$, and $m$ is the modified refractive index, which includes Earth flattening.

**Exact Helmholtz in the shifted frame** (Beilis–Tappert; Holm; Donohue–Kuttler):
$$u_{xx}+2iku_x-2T'(u_{x\zeta}+iku_\zeta)-T''u_\zeta+(1+T'^2)u_{\zeta\zeta}+k^2(m^2-1)u=0 .$$

**NA shift map (SMPE, TPEM/Barrios, Levy ch. 7, Khodr, Parakkal).** Drop $u_{xx}$, $T'u_{x\zeta}$ and $T'^2u_{\zeta\zeta}$, then apply the gauge
$$u=v\,\exp\!\Big(ikT'\zeta+\tfrac{ik}{2}\!\int^xT'^2\Big).$$
The cross term $-2ikT'u_\zeta$ cancels exactly against the gauge, and the $\pm k^2T'^2$ terms cancel too, provided $T'^2u_{\zeta\zeta}$ was dropped. What remains is
$$2ik\,v_x+v_{\zeta\zeta}+k^2\big(m^2-1-2\zeta T''\big)v=0 \qquad(\text{dropped: }-T''v_\zeta-ikT'T''v).$$
- **Ground conditions.** The Dirichlet condition is unchanged. An impedance condition $u_\zeta+\alpha u=0$ becomes $v_\zeta+(\alpha+ikT')v=0$ to this order; zhang2023ibcwasm shows that how this is discretized matters.
- **Piecewise-linear terrain.** $T''$ vanishes on each facet. At a slope break the gauge jumps, and $v$ is multiplied by $\exp[ik\zeta(T'_{\rm old}-T'_{\rm new})]$ (Donohue–Kuttler). In the WA PLSM version the jump is $\exp\{ik\zeta[T_1'/(1+T_1'^2)-T_2'/(1+T_2'^2)]\}$, as quoted in Nilsson 2021, Sec. 2.7.
- **Accuracy.** The paraxial error of the NA SMPE is independent of slope (wang2020smpe).

**WA shift map, first order in slope.** This is the Donohue–Kuttler PLSM, Holm's "first-order solution", and the scheme in gao2019roughsea.

*Phase factor.* Keep $\theta=k_0T'\zeta+f(x)$ with $k_0=k/\sqrt{1+T'^2}=k\cos\beta$, $T'=\tan\beta$. The square root of the non-commuting sum is then split by
$$\sqrt{1+a+b+c}\approx\sqrt{1+a}+\sqrt{1+b}+\tfrac c2-1 .$$

*Resulting equation.* This gives
$$u_x=i\sqrt{k^2\cos^2\!\beta+\partial_\zeta^2}\;u+ik\Big(\sqrt{m^2-\sin^2\!\beta}-1\Big)u$$
(gao2019roughsea eq. 30; Nilsson eqs. 145–150). In words, it is the flat-ground WA PE with $k\to k\cos\beta$ in the diffraction term and $m^2\to m^2-\sin^2\beta$ in the refraction term.

*Discretization.* In FD form (Holm), $\sqrt{\cdot}$ uses Claerbout [1/1]. The scheme is tridiagonal for first order in $T'$ and pentadiagonal when $O(T'^2)$ is kept (Holm's "second-order solution").

*Equivalent form.* Khodr 2020 thesis eq. 3.59 gives the same result in operator form: $\partial_\xi\psi=ik(\sqrt{1+Z}-1)\psi-ik\eta T''\psi$.

**Flat-ground WA FD operators used in this literature** (these are what "WA FD PE" means in the TAP/AWPL papers above).

*Claerbout [1/1].* With $\sqrt{1+Z}\approx(1+\tfrac34Z)/(1+\tfrac14Z)$ and $k^2Z=\partial_z^2+k^2(m^2-1)$:
$$\big(\partial_z^2+k^2(m^2+3)\big)u_x=2ik\big(\partial_z^2+k^2(m^2-1)\big)u,$$
i.e. $u_{xzz}-2iku_{zz}+k^2(m^2+3)u_x-2ik^3(m^2-1)u=0$ (ding2016fractalsea eq. 3 [full]; wangr2017dnlbc3d eq. 3 [full], "up to 40°").

*Greene.* $\sqrt{1+Z}\approx(\chi_1+\chi_2Z)/(\chi_3+\chi_4Z)$ with $\chi_1=0.99987$, $\chi_2=0.79624$, $\chi_3=1$, $\chi_4=0.30102$ (guo2023forestry eqs. 10–11 [full]; guo2017greene).

*Padé(2,2).* guo2017pade and guo2020roughsea2w.

*Range discretization.* Always Crank–Nicolson: midpoint $\xi_m=(x_{m-1}+x_m)/2$, $u(\xi_m)=\tfrac12(u^m+u^{m-1})$, $u_x\approx(u^m-u^{m-1})/\Delta x$. The mixed derivative $u_{xzz}$ is $(\delta_z^2u^m-\delta_z^2u^{m-1})/\Delta x$ (wangr2017dnlbc3d eqs. 7–11).

*Height discretization.* Two variants appear:
- Holm, Ding 2016 and Wang 2017 use second-order three-point $\delta_z^2$, which gives a tridiagonal system.
- The Long group (guo2017greene, guo2017pade, guo2023forestry) uses the **fourth-order five-point stencil** $(-1,16,-30,16,-1)/12\Delta z^2$, which gives a pentadiagonal system. The Leontovich ground condition is closed with a one-sided third-order formula $u_0=(48u_1-3u_2)/(45-42\xi+18\xi^2-4\xi^3)$ (guo2023forestry eqs. 12–17 [full]).

**Domain truncation as published.** None of the FD shift-map papers uses a transparent BC.
- **Tukey window**, top 30 % of the domain (guo2023forestry [full]).
- **Hanning (Hann) window** layer (Levy; Nilsson: about 100 m is enough for near-paraxial fields; wangr2017dnlbc3d for the SSF reference). zhangp2014window shows that the Tukey and Hanning windows used in PE codes are the same expression.
- **Complex wavenumber.** $\tilde k=k+i\kappa_0((z-z_a)/(z_{\max}-z_a))^2$ in the top third of the domain (Khodr 2020 [full], after Salomons). The authors advise making the layer thicker rather than increasing $\kappa_0$.
- **PML adapted to CN.** ozbayat2011abc, and levy2001pml.
- **Discrete/diffraction NLBC with recursive convolution**, for a flat-ground FD WA PE only (wangr2017dnlbc3d).

---

## 3. Per-paper notes

### 3.1 Long group (SYSU): FD WA shift-map PE

#### holm2007wasm† [abstract + secondary: Khodr thesis, Nilsson 2021, ../README §D]
- **Scheme.** Claerbout [1/1] with CN in range. The terrain enters through the shift map $\eta=z-h(x)$ with piecewise-linear $h$. The first-order solution is equivalent to Donohue–Kuttler and is tridiagonal. The second-order solution keeps $h'^2$ and is pentadiagonal.
- **Stated limits.** First order: slopes 10–15° and slope jumps 15–20°. Second order: slopes 15–20° and jumps 30–40°.
- **Later criticism.** zhang2024techrxiv argues that Holm's reduced function and first-order solution "are not accurate for waves propagating at 0°". zhang2024wasm explains why reducing $\Delta x$ and $\Delta z$ made Holm/Zhou's scheme *less* accurate.

#### zhou2015greene†, guo2017greene† [abstract + secondary: guo2023forestry full text reproduces the flat-ground Greene FD scheme]
- **Scheme.** Greene rational approximation with CN in range and a fourth-order 5-point stencil in height. Holm's second-order slope expansion is redone for Greene, and the scheme is pentadiagonal.
- **Claims.** "Performs well for slopes up to about 15°, and discontinuous slope changes up to about 30°, better than split-step/Fourier." The paper also derives "general PE solutions with respect to the terrain slope for the different approximations". Earlier notes quote phase-error-based flat-ground limits of 36° for Claerbout and 47° for Greene (phase error < 0.002).
- **Absorber and grid** (from guo2023forestry, same author and code). Tukey window over the top 30 % of the maximum height. $\Delta x=\Delta z=\lambda/2$ at 300 MHz, with a 3 km × 10 km domain.

#### guo2017pade†, guo2018twoway†, guo2020roughsea2w, guo2021idpe [abstract]
- **Padé(2,2) FD.** Pentadiagonal. "Propagation angles up to about 55° from the paraxial direction." It is extended to general terrain by "the boundary shift approach", and validated against fir-forest measurements.
- **Two-way versions.** The 2018 paper is a two-way Greene FD PE ("up to about 55°"). The 2020 paper is a two-way Padé(2,2) PE over fractal rough sea. guo2021idpe applies backward FD PE marching to source localization over terrain.
- **Limits not reported.** None of them report the slope limit of the shift map at Padé(2,2) order.

#### guo2023forestry [full]
- **Why it matters here.** It is the only open-access text from this group that writes out the full FD Greene scheme: pentadiagonal CN matrices, boundary closures, and the coefficients $T=1-ik\Delta x(\chi_2-\chi_4)/(2\chi_4)$, $D$ and $d$ in eq. 15.
- **Setup.** Leontovich IBC. Tukey absorbing layer equal to 30 % of the maximum height. $\Delta x=\Delta z=\lambda/2$ (300 MHz, 3 km height, 10 km range). Validated against AREPS and measured forest data.

#### ding2016fractalsea [full]
- Claerbout FD PE written as $u_{xzz}-2iku_{zz}+k^2(n^2+3)u_x-2ik^3(n^2-1)u=0$, with CN midpoint averaging and the Leontovich condition $\partial_zu+\beta u=0$, $\beta=ik\sin\theta\,(1-\Gamma)/(1+\Gamma)$.
- Sea surface: *staircase* approximation. It cites Holm for the FD details.

#### zhang2023ibcwasm [abstract]
- **Problem.** The original IBC difference schemes for the WA-SMPE have defects that come from treating the boundary like a PEC. The paper replaces them with one-sided difference schemes.
- **Benchmark.** A horizontally spreading Levy source with 0.5° beamwidth illuminates a slope. The PE reflection angle is compared with Fresnel reflection, and "accurate" means a reflection-angle offset of less than 0.5°.
- **Slope limits under that criterion.**
  - New first-order PE: up to **14°**.
  - Second-order Claerbout: **16°**.
  - Second-order Greene: **19°**.
- This is the most quantitative recent statement of WA-SMPE slope limits.

#### zhang2024wasm† [abstract]
- **Idea.** Uses the Taylor expansion of the forward equation to turn $\partial_x$ into $\partial_z$. Range truncation errors become a $\partial_z$ series, which is then compensated in a high-accuracy difference scheme.
- **Claims.** For a 15° terrain and propagation angles of −10° to 10°, it is at least 10× faster than the "second-order approximation of the WA equation for terrain" (zhou2015greene/guo2017greene). It also explains why refining $\Delta x$ and $\Delta z$ *decreases* accuracy in the original scheme.

#### zhang2024techrxiv† [abstract; archived page saved]
- **Claim.** A new reduced function removes the defect of Holm's 1st/2nd-order solutions near 0°. With $\Delta x=128\lambda$ it is more accurate than Holm's solution with $\Delta x=1\lambda$.

#### zhang2026qoperator [title only]
- Error analysis of the SMPE based on the $Q$-operator (the square-root operator), applied to long-range propagation. Check it before submission, since it is the newest TAP-family item on the same question.

#### wu2023fdpade3d, wu2023ihaadi [abstract]
- **4FDHP (3-D, flat).** Fourth-order FD in the transverse directions plus higher-order Padé in range, with Leontovich IBC. The authors state a larger propagation angle than 2FDHP or Mitchell–Fairweather ADI.
- **IHA-ADI.** An unconditionally stable iterative ADI that is more accurate than MF-ADI.
- **Relevance.** These are the closest "higher-order Padé + FD" works in radio, but with no terrain map and no TBC.

#### wang2012twoway†, ozgun2014comments, wangk2014reply [abstract]
- 2W-SSPE with DMFT, using the "boundary shift (BS)" method for terrain.
- Ozgun, Apaydin, Kuzuoglu and Sevgi questioned how the backward wave is handled with BS. The reply defends the BS. This is useful context for why one-way shift maps cannot produce backscatter.

### 3.2 Xi'an University of Technology group: NA vs WA over terrain at LF

#### wang2020smpe† [abstract]
- **Method.** Artificial (paraxial) dispersion analysis of the NA and WA SMPE in transformed coordinates. "Artificial dispersion" here is the PE-vs-Helmholtz dispersion, not grid dispersion.
- **Findings.**
  - The NA-SMPE dispersion is independent of terrain.
  - The WA-SMPE dispersion depends strongly on slope, and its WA capability "deteriorates dramatically".
  - For *shady* (lee-side) slopes above about 5°, the WA-SMPE is worse than the NA-SMPE.
- **Implementation.** The group's codes are SSFT plus DMFT (wang2016loranasf, zhouz2017msam). The analysis is not specific to FD.

#### zhou2017lfhighangle [abstract]
- **Comparison.** FDTD vs. one-way and two-way NA/WA PE at LF and short range.
- **Findings.**
  - The NAPE "is hard to apply where the propagation angle is above 15°".
  - The WAPE is better while the slope is "not very big".
  - Where slope changes dramatically, the two-way WAPE is more reliable.

#### zhouz2017msam [full]
- **Setup.** NAPE $u_x=\frac{ik}{2}(n^2-1+k^{-2}\partial_z^2)u$ and WAPE $u_x=ik(\sqrt{1+k^{-2}\partial_z^2}-1+n-1)u$ (Feit–Fleck), both by SSFT/DMFT, compared with cylindrical FDTD that uses PML. 100 kHz, $\Delta x=200$ m (≈λ/15), $\Delta z=100$ m (≈λ/30), domain height 35 km, range 100 km. The accuracy criterion is error < 0.02 dB.
- **Results.** Gaussian mountain $H\exp(-9(x-50)^2/2)$ km with $H$ = 0.5, 1.0 and 1.5 km. Both PEs are fine for low hills. "The WAPE method cannot provide the correct solution for the steeper mountains, whereas the NAPE performs well". The WAPE also shows range/height jitter. This is an independent numerical confirmation of wang2020smpe.

#### wang2016loranasf, wang2018pulseasf, wang2019hybrid† [abstract]
- SSFT PE with the shift map ("conformal mapping model", CMM, in Levy's terminology) for Loran-C ASF over terrain, validated against IE and FDTD.
- The hybrid 2W-PE combines a staircase for the backward wave with CMM for the forward wave.

### 3.3 Other radio shift-map work

#### zhangr2024maxangle† [abstract; full text blocked]
- WA shift-map model. The maximum propagation angle is $\theta_{\max}=\theta^{\rm flat}_{\max}+2\max|\text{slope}|$, with criteria for hybrid terrain models. It points out a flaw in an earlier NA formula.

#### gao2019roughsea† [full]
- **Method.** WA SSF with the piecewise-linear shift map over a JONSWAP/double-superposition rough sea, using eq. 30 (the formula given in §2 above) and the SSF solution in eq. 33.
- **Phase factor.** $\theta=kz'T'+f(x)$ with $K_0=k/\sqrt{1+T'^2}$. The paper calls PLSM "the most precise terrain processing method in PE".

#### huang2026gass [full, 1-page extended abstract]
- **Method.** 3-D NA PE in a shift-map frame over a piecewise-planar relief ($T(x,y)$). Unlike Khodr, the lateral cross term ($\propto T_y\partial_{y\zeta}$) is **kept**. The standard PE is recovered in transformed coordinates, with a separately computed phase-corrector term. Solved with SSF.
- **Validation.** Against FEKO MLFMM for a PEC pyramid. Keeping the cross term raises the field behind the peak and matches MLFMM better.

#### ligny2021phasecorr [abstract]
- Adds a phase correction to Donohue–Kuttler's shift map so that the field *phase* (not just the amplitude) at the receiver is right. The difference from a classical SSF reference is below −35 dBV/m. Used with SSW in non-rectangular domains (radio occultation).
- **Relevance.** The gauge factor $e^{ikT'\zeta}$ must be undone consistently, and the same issue arises for any NLBC at the top.

#### qin2024sfft2w, qin2025sspeterrain, huang2025pgcnn, an2026apce [abstract]
- University of Alberta group (X. Zhang, formerly with Sarris).
- 2W-SSPE over irregular terrain (sparse FFT), validated with Danish terrain measurements.
- The 2025 PIERS paper adds "terrain-following transformations and adaptive spatial discretization ... to ensure accuracy in steep terrain transitions" with an IBC.
- The CNN and PCE papers use these PE solvers as the reference model. This shows that the community still treats SSPE/2W-PE as the "truth" for irregular terrain.

#### magno2011imoc [abstract]
- NA PE solved by implicit CN FD ("propagation in 15° to the paraxial direction"), and WA PE by DMFT-SSF ("up to 90°"), for a mixed foliated path (900 MHz and 1.8 GHz).

### 3.4 Beilis–Tappert PE in atmospheric acoustics (same mathematics, CN FD, the most careful slope studies)

#### gilbert2022wabt (+ gilbert2018exactbt) [abstract]
- **Exact WA BT.** An exact wide-angle BT formulation follows from replacing $\partial_z\to\partial_z+ik_0\tan\varphi$, where $\varphi$ is the slope angle.
- **Why existing schemes fail on steep slopes.** The vertical wavenumber in BT coordinates is **not physical**, so a slope-dependent $k$-space vertical-wavenumber filter ("k-control") is required. For large slopes much of the field is evanescent in range. "Existing finite difference methods and all narrow-angle methods fail to properly account for the exponential decay ... of the non-propagating components."
- **Steep vs. shallow hills.** For a steep hill, k-control matters as much as the WA extension. For a shallow hill the original NA BT is acceptable.
- **For the TAP comparison.** This is the strongest physical argument for why the slope error of shift-map WA PEs does not converge with Padé order.

#### parakkal2011jasaabs / parakkal2012beilis† [abstract]
- BTPE with an exact porous-ground impedance condition, tested on hills with slopes of 5–22°. The flat-ground impedance condition is adequate below about 20°. Above about 20° "the limiting factor ... becomes the narrow-angle approximation used in the BT method". Benchmark: polar-coordinate PE.

#### khodr2020iterative† (+ khodr2018jasaabs, khodr2021jasaabs) [full]
- **Method.** 3-D NA BTPE, CN in range, with the iterative tridiagonal (fixed-point) scheme. All $h$-derivative terms in the transverse operator are dropped ("valid to 20°").
- **Absorber.** Top layer from $z_a=2$ km to $z_{\max}=3$ km (one third of the domain height) and side layers 1 km wide, using $\tilde k=k_{\rm eff}+i\kappa_0((\eta-\eta_a)/(\eta_{\max}-\eta_a))^2$.
- **Grid.** About $\lambda/10$ in both directions at 5 Hz (437 × 583 transverse points, 1450 range steps over 10 km).
- **Validation.** Against COMSOL BEM on a Gaussian hill.

#### nyborg2023† [abstract]
- BTPE vs. GTPE (Sack–West) vs. measurements for wind-turbine noise. Both are CN FD WA (Padé(1,1)) codes. Full text blocked.

### 3.5 Domain truncation for FD (CN) PEs

#### ozbayat2011abc (+ ozbayat2019diss) [abstract]
- **Problem.** The discrete TBC (spatial convolution) for the FD CN PE is costly, so the paper builds two cheaper local approximations of it.
- **PML.** A PML adapted to Crank–Nicolson is implemented, with stability analyses of the approaches compared on a 2-D model problem.
- **Relevance.** This is the radio-PE reference for "PML vs. DTBC with CN FD". The dissertation uses these for "propagation over deterministic terrain".

#### zhangp2014window, zhangp2016review [abstract]
- **Window equivalence.** Tukey and Hanning windows on the top absorbing layer are mathematically the same configuration. The paper proposes conditions a window must satisfy.
- **Review.** A survey of PE frameworks, algorithms, boundary conditions (absorbing layers, NLBC, PML) and terrain models in tropospheric groundwave propagation.

#### wangr2017dnlbc3d [full]
- **Method.** 3-D WA (Claerbout, "up to 40°") FD CN decomposition model, with a diffraction NLBC accelerated by recursive convolution (vector fitting). One grid layer absorbs completely.
- **Comparison.** The SSF reference needs Hanning windows on four sides.
- **Grid.** 3 GHz, $\Delta x=0.5$ m (5λ), $\Delta y=\Delta z=0.01$ m (λ/10).
- **Terrain.** Knife-edges only. No shift map.

#### li2016troposcatter [abstract]
- How to choose the maximum calculation height (i.e. where to put the absorber) and the maximum propagation angle of a WA SSF PE from the troposcatter common-volume geometry.

### 3.6 Grid-dispersion analysis of FD PEs

#### zhang_sarris2015†, zhangx2014adi, zhangx2014masc [thesis full]
- **Analysis.** Von Neumann/dispersion analysis of CN, PR-ADI, LOD and Mitchell–Fairweather ADI for the NA (vector) PE. The parameters are $N_\lambda=\lambda/\Delta x_\perp$, the step ratio $\alpha$ and the angle $\theta$.
- **Results.** CN has the largest phase error per wavelength, and MF-ADI the smallest. The analysis gives criteria for choosing discretization parameters and separates the PE-vs-Maxwell error from the discretization error.
- **Scope.** NA, tunnels, no terrain map. This is the standard TAP citation for "grid selection by discrete dispersion" in FD PE.

### 3.7 Lytaev (terrain-related, not yet in ../refs.bib)
- **lytaev2019icumt.** Higher-order PE vs. Longley–Rice over real terrain profiles.
- **lytaev2020electronics.** Two-way PE in an urban (staircase) geometry.
- **vavilov2020knifeedge.** Modeling equation for multiple knife-edge diffraction.
- **lytaev2024piersuav.** UAV relay over terrain.
- **None of these uses a shift map.** The existing refs (lytaev2018rtuwo, lytaev2021iccsa, lytaev2020awpl, lytaev2018awpl, lytaev2022radarconf, lytaev2026nmp) use staircase/immersed terrain with Padé + discrete NLBC.

---

## 4. Summary of published limits

| quantity | value | source |
|---|---|---|
| NA PE usable propagation angle | ≈15° | zhou2017lfhighangle, magno2011imoc, Levy |
| NA BT/SMPE slope limit | ≈20° | parakkal2011jasaabs, khodr2020iterative |
| WA flat-ground angle, Claerbout | 36° (phase error < 0.002); "up to 40°" | guo2017greene; wangr2017dnlbc3d |
| WA flat-ground angle, Greene | 47° | guo2017greene |
| WA flat-ground angle, Padé(2,2) | ≈55° | guo2017pade, guo2018twoway |
| WA-SMPE slope, first order (Holm/DK) | 10–15° (jumps 15–20°); new first-order IBC: 14° | holm2007wasm; zhang2023ibcwasm |
| WA-SMPE slope, second order | Claerbout 15–20° (16° by reflection test), Greene 15–19° (jumps ≈30°) | holm2007wasm, guo2017greene, zhang2023ibcwasm |
| WA-SMPE vs NA-SMPE | WA worse for shady slopes > 5°; WAPE fails on steep mountains | wang2020smpe; zhouz2017msam |
| required PE angle over terrain | $\theta^{\rm flat}+2\max|\text{slope}|$ | zhangr2024maxangle |
| typical FD grids | $\Delta x=\Delta z=\lambda/2$ (Greene FD, 4th-order $z$); $\Delta x\approx5\lambda$, $\Delta z=\lambda/10$ (Claerbout FD); $\approx\lambda/10$ both (acoustic BT CN); improved WA-SMPE up to $\Delta x=128\lambda$ | guo2023forestry; wangr2017dnlbc3d; khodr2020iterative; zhang2024techrxiv |
| absorber | Tukey top 30 %; Hanning top layer; complex-$k$ quadratic top 1/3; CN-PML; flat-ground NLBC only | guo2023forestry; Levy/Nilsson; khodr2020iterative; ozbayat2011abc; wangr2017dnlbc3d |
