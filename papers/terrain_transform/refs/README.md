# Terrain and bathymetry transforms for the higher-order (Padé) parabolic equation: literature survey

Compiled 2026-10-09 for pywaveprop. Context: the split-step Padé / rational propagator of arbitrary order (product of $(1+a_jL)/(1+b_jL)$), Numerov vertical discretisation, discrete NLBC at the top. Today terrain and seabed are staircases. The goal is a change of variables that maps the irregular domain onto a rectangle **without** limiting the propagation angle, works for **any** rational order, and stays compatible with the **upper discrete NLBC**.

* `refs.bib` holds BibTeX for every entry below, grouped by topic and deduplicated by key and DOI.
* PDFs are in this directory. Lytaev's own papers already in the repo are cited as `../../AWPL2018.pdf`, `../../awpl2020.pdf` and `../../radarconf22.pdf`.
* Downloaded files are untrusted third-party data. Text was extracted with pypdf only, and nothing in them was executed.
* Where a paper was not available (IEEE, JASA, Elsevier, SIAM and World Scientific paywalls; MDPI, Hindawi, AMS and TechRxiv blocked scripted downloads; DTIC was offline on the survey date), the notes are based on the abstract (OpenAlex/Crossref) and on how later papers that *were* read restate it. Such notes are marked "[abstract]", "[abstract/secondary]" or "(closed; reconstructed …)".
* Correction to the task statement: Beilis & Tappert (1979) is in **J. Acoust. Soc. Am. 66(3):811–826**, not J. Math. Phys.

---

## 0. Executive summary

**What each family of methods turns the Helmholtz equation $\psi_{xx}+\psi_{zz}+k^2n^2\psi=0$ into**

| family | transformed equation | exact before one-way step? | practical angle/slope limit (as published) | arbitrary-order Padé? | top discrete NLBC? |
|---|---|---|---|---|---|
| Staircase (current pywaveprop, PETOOL, SSW, ECPE, single scattering) | unchanged; geometry approximated | geometry is not exact | slope-dependent errors; ECPE and single scattering fix the interface conditions only | yes | yes (unaffected) |
| Beilis–Tappert shift $z'=z-T(x)$ + phase gauge (TPEM, TEMPER, APM, BTPE, GT-PE) | $\psi_{x'x'}-2T'\psi_{x'z'}-T''\psi_{z'}+(1+T'^2)\psi_{z'z'}+k^2n^2\psi=0$ → narrow angle: $2ikv_x+v_{z'z'}+k^2(n^2-1-2z'T'')v=0$ | yes, but non-conformal (cross terms), and they are removed only at the paraxial level | NA: ≈15–20°; WA (Donohue–Kuttler PLSM, Holm, Guo–Long): slopes 10–20°, jumps 15–40°; Wang 2020: WA worse than NA for slopes > 5° | formally yes, but the slope coupling is first/second order in $T'$ → **not** angle-unrestricted | exterior gets $-2zT''(x)$ and a tilt phase $e^{ikT'z}$ → range-dependent kernel; no published NLBC |
| Rotated PE (Collins 1990; Outing et al. 2006) | unchanged Helmholtz in rotated axes (rotation invariance of $\Delta$) | exact | piecewise-constant slopes only; aperture shifted by the slope; interpolation at slope breaks | yes | only if the exterior is homogeneous (a stratified exterior becomes tilted) |
| Depth scaling / shift after paraxialisation (Abrahamsson–Kreiss, Akrivis–Dougalis–Zouraris, Antonopoulou, Collins–Dacol, Moran/Metzler, Janaswamy 1994/98, Kampanis) | narrow-angle or (1,1) PE in $y=z/s(r)$, with $y\dot s/s\,u_y$ terms, dynamical BCs, extra BCs | no (metric terms dropped or kept only to PE order) | a few degrees (Moran: ≤4° tested, 1–5 dB errors); (1,1)-Padé aperture | no (each order needs a new derivation and extra BCs) | yes **if the map is the identity near the top** (Kampanis 2002/2005/2013: "NLBC left unaffected by the transformation") |
| Sigma / Gal-Chen map $\eta=H(z-h)/(H-h)$ (Janaswamy 1998, 2025; WAPEtopo) | anisotropic medium $\nabla\cdot(A\nabla u)+k^2n^2J u=0$ with mixed derivatives (exact; Janaswamy 2025 gives the 3-D Maxwell tensor) | yes (as anisotropic medium) | WAPEtopo ≈30°; Janaswamy 1998 narrow angle | no published one-way factorisation of the anisotropic operator | **yes**: identity above $H$ |
| Transformation-EM (Ozgun–Kuzuoglu) | $\varepsilon'=\mu'=\Lambda\Lambda^T/\det\Lambda$ | yes | full-wave FEM/FDTD only; no PE | n/a | n/a |
| **Conformal map** $z=\Phi(w)$ (Dozier 1984, Donohue–Kuttler 1997, Parakkal 2006–10, Novikov–Akulinichev 2011, Oba 2010; Earth flattening; Heiblum–Harris bends; Kirby 1994 water waves) | **exact Helmholtz** $\psi_{\xi\xi}+\psi_{\eta\eta}+k^2n^2|\Phi'|^2\psi=0$; Dirichlet/Neumann invariant; impedance $\alpha\to\alpha|\Phi'(\xi,0)|$; divergence form (V-pol, density) preserved | **yes** | published limits (≈6°, "$|w'|\approx1$") came from the *narrow-angle SSF* solver, not from the map | **yes**: it is a range-dependent medium, which split-step Padé already handles | only asymptotically: $|\Phi'|\ne1$ at any finite height (identity theorem); the residual decays like $e^{-2\pi\eta/L}$ (strip/periodic maps) or algebraically (localised hill) |
| Local orthogonal transform + DtN (Lu 1999–2004) | $\partial_\xi(\tfrac{h_2}{h_1}\partial_\xi u)+\partial_\eta(\tfrac{h_1}{h_2}\partial_\eta u)+h_1h_2k^2u=0$ | yes | wide-angle in principle | compatible | bounded ducts, no NLBC studied |

**Genuinely angle-unrestricted and Padé-compatible.** Only two kinds of method transform the *Helmholtz* equation exactly into a form the existing split-step Padé machinery can march without new derivations:

1. **Conformal maps.** These give an isotropic medium $n^2|\Phi'|^2$.
2. **Rotations.** These apply to a constant slope only.

Everything else either drops metric terms or maps an already-paraxial equation. That holds for the shift map, depth scaling and the non-conformal curvilinear grids. For those methods the slope error does not decrease with Padé order (Wang et al. 2020 showed this numerically for the wide-angle shift map). The maximum angle that a terrain PE must resolve is $\theta^{\rm flat}_{\max}+2\max|\text{slope}|$ (Zhang et al. 2024; Khodr 2020). This sets the Padé order and grid once the map has removed the geometric error.

**Gaps that are still open.** No paper found, in radio or underwater acoustics up to 2026, combines all three of the following:

* (a) a conformal or other Helmholtz-exact terrain/bathymetry map;
* (b) an arbitrary-order split-step Padé/rational propagator;
* (c) an exact discrete NLBC.

The closest work covers two of the three:

* **Earth flattening + Lytaev 2018** have (a)+(b)+(c), but the map is global and range-independent.
* **Kampanis 2002/2005/2013** has a terrain-fitting map with an NLBC that the map does not affect, but uses a narrow-angle FEM.
* **Janaswamy 1994** has curvilinear coordinates with a continuous TBC, but is narrow-angle.
* **Donohue–Kuttler 1997** tried global conformal maps, but with narrow-angle SSF.
* **Oba 2010** uses a numerical conformal map with an existing PE, but for the sea surface and without an NLBC.
* **Kirby et al. 1994** derive wide-angle rational PEs in conformal coordinates for water waves, with no TBC.
* **Khodr 2020** uses the BT map with higher-order Padé, but the slope coupling is first order.

No one has used a numerical (Schwarz–Christoffel or rational/AAA) conformal map of real DEM terrain or bathymetry inside a PE. No one has quantified the NLBC error caused by the residual $|\Phi'|^2-1$ at the artificial boundary.

The interaction between a flattening map and a nonlocal *lower* boundary (Lytaev 2022 Fresnel NLBC) is also untreated. The map distorts the ground half-space by $|\Phi'|^2$ as well.

Design options suggested by the literature:

* **Strip conformal map.** Use a strip map (Schwarz–Christoffel / AAA) whose upper edge is a straight line, with the NLBC boundary placed where $|\Phi'|-1$ is below tolerance. Then quantify the error, or derive an NLBC for a slowly varying exterior.
* **Hybrid map.** Use a conformal map in the lower layer, blended to the identity below $z_b$, and accept small cross terms only in the blending layer.
* **PML instead of NLBC.** A PML (Levy 2001) composes with any map.

The mesh-optimisation and discrete-dispersion theory (Lytaev 2020–2026) carries over to the conformal case: replace the range of $n^2$ with the range of $n^2|\Phi'|^2$.

---

## 1. Master table

Columns: key (as in `refs.bib`) | authors | year | title | venue | DOI | local file. The topic groups match the sections below.

**A. Beilis–Tappert / shift-map terrain PE (radio: TPEM, TEMPER, APM) and early conformal PE**

| key | authors | year | title | venue | DOI | local file |
|---|---|---|---|---|---|---|
| beilis1979coupled | A. Beilis, F. D. Tappert | 1979 | Coupled mode analysis of multiple rough surface scattering | J. Acoust. Soc. Am. 66(3):811–826 | 10.1121/1.383685 | not downloaded |
| dozier1984peruse | L. B. Dozier | 1984 | PERUSE: A numerical treatment of rough surface scattering for the parabolic wave equation | J. Acoust. Soc. Am. 75(5):1415–1432 | 10.1121/1.390849 | not downloaded |
| levy1990terrain | M. F. Levy | 1990 | Parabolic equation modelling of propagation over irregular terrain | Electronics Letters 26(15):1153–1155 | 10.1049/el:19900746 | not downloaded |
| kuttler1991ssf | J. R. Kuttler, G. D. Dockery | 1991 | Theoretical description of the parabolic approximation/Fourier split-step method of representing electromagnetic propagation in the troposphere | Radio Science 26(2):381–393 | 10.1029/91RS00109 | not downloaded |
| marcus1992hybrid | S. W. Marcus | 1992 | A hybrid (finite difference–surface Green's function) method for computing transmission losses in an inhomogeneous atmosphere over irregular terrain | IEEE TAP 40(12):1451–1458 | 10.1109/8.204735 | not downloaded |
| ursi1993barrios | A. E. Barrios | 1993 | A terrain parabolic equation model (TPEM) — URSI National Radio Science Meeting abstract F2-3 (p. 115 of program book) | URSI NRSM, Boulder, Jan. 1993 | — | ursi_1993_program_barrios_tpem_abstract_p115.pdf |
| barrios1993tpemreport | A. E. Barrios | 1993 | A Terrain Parabolic Equation Model (TPEM) | NRaD Technical Document (DTIC ADA264672) | 10.21236/ADA264672 | not downloaded (DTIC offline) |
| kuttler1993rough | J. R. Kuttler, J. D. Huffaker | 1993 | Solving the parabolic wave equation with a rough surface boundary condition | J. Acoust. Soc. Am. 94(4):2451–2453 | 10.1121/1.407469 | not downloaded |
| barrios1994tpem | A. E. Barrios | 1994 | A terrain parabolic equation model for propagation in the troposphere | IEEE TAP 42(1):90–98 | 10.1109/8.272306 | not downloaded |
| barrios1994roughssf | A. E. Barrios, K. H. Craig | 1994 | Rough surface models implemented within the split-step parabolic equation algorithm | NRaD Technical Report (DTIC ADA280843) | 10.21236/ADA280843 | barrios_craig_1994_rough_surface_ssf.pdf |
| donohue1997apl | D. J. Donohue, J. R. Kuttler | 1997 | Modeling radar propagation over terrain | Johns Hopkins APL Tech. Digest 18(2):279–287 | — | donohue_kuttler_1997_apl_terrain.pdf |
| donohue2000gtpe | D. J. Donohue, J. R. Kuttler | 2000 | Propagation modeling over terrain using the parabolic wave equation | IEEE TAP 48(2):260–277 | 10.1109/8.833076 | not downloaded (closed; content reconstructed from donohue1997apl, nilsson2021msc, guan2018dem) |
| levy2000book | M. Levy | 2000 | Parabolic Equation Methods for Electromagnetic Wave Propagation (Ch. 7 "Irregular terrain modelling") | IEE Electromagnetic Waves Series 45 | 10.1049/PBEW045E | not downloaded |
| barrios2003apm | A. E. Barrios | 2003 | Considerations in the development of the Advanced Propagation Model (APM) for U.S. Navy applications | Proc. Int. Conf. Radar 2003, pp. 77–82 | 10.1109/RADAR.2003.1278714 | not downloaded |
| awadallah2004apl | R. S. Awadallah, J. Z. Gehman, J. R. Kuttler, M. H. Newkirk | 2004 | Modeling radar propagation in three-dimensional environments | Johns Hopkins APL Tech. Digest 25(2):101–111 | — | awadallah_2004_apl_techdigest_terrain.pdf |
| awadallah2005lateral | R. S. Awadallah, J. Z. Gehman, J. R. Kuttler, M. H. Newkirk | 2005 | Effects of lateral terrain variations on tropospheric radar propagation | IEEE TAP 53(1):420–434 | 10.1109/TAP.2004.840853 | not downloaded |
| parakkal2012beilis | S. Parakkal, K. E. Gilbert, X. Di | 2012 | Application of the Beilis–Tappert parabolic equation method to sound propagation over irregular terrain | J. Acoust. Soc. Am. 131(2):1039–1046 | 10.1121/1.3676693 | not downloaded |
| guan2018dem | X.-W. Guan, L.-X. Guo, Y.-J. Wang, Q.-L. Li | 2018 | Parabolic equation modeling of propagation over terrain using digital elevation model | Int. J. Antennas Propag. 2018:1878307 (OA) | 10.1155/2018/1878307 | guan_2018_pe_terrain_dem.pdf |
| nilsson2021msc | M. Nilsson | 2021 | Radio-wave propagation modelling over rough sea surfaces and inhomogeneous atmosphere (MSc thesis; reproduces the Donohue–Kuttler PLSM derivation, Sec. 2.7.1) | Karlstad University (DiVA diva2:1567840) | — | nilsson_2021_msc_karlstad_pe_rough_sea_terrain.pdf |

**B. Underwater acoustics: sloping bottoms, rotated PE, mapped depth coordinates**

| key | authors | year | title | venue | DOI | local file |
|---|---|---|---|---|---|---|
| collins1990rotated | M. D. Collins | 1990 | The rotated parabolic equation and sloping ocean bottoms | JASA 87(3):1035–1037 | 10.1121/1.398829 | not downloaded |
| outing2006variable | D. A. Outing, W. L. Siegmann, M. D. Collins, E. K. Westwood | 2006 | Generalization of the rotated parabolic equation to variable slopes | JASA 120(6):3534–3538 | 10.1121/1.2372590 | not downloaded |
| collins1991energy | M. D. Collins, E. K. Westwood | 1991 | A higher-order energy-conserving parabolic equation for range-dependent ocean depth, sound speed, and density | JASA 89(3):1068–1075 | 10.1121/1.400526 | not downloaded |
| porter1991energy | M. B. Porter, F. B. Jensen, C. M. Ferla | 1991 | The problem of energy conservation in one-way models | JASA 89(3):1058–1067 | 10.1121/1.400525 | not downloaded |
| collins1993splitstep | M. D. Collins | 1993 | A split-step Padé solution for the parabolic equation method | JASA 93(4):1736–1742 | 10.1121/1.406739 | not downloaded |
| collins2000mapping | M. D. Collins, D. K. Dacol | 2000 | A mapping approach for handling sloping interfaces | JASA 107(4):1937–1942 | 10.1121/1.428476 | not downloaded |
| moran2012scaled | D. Moran | 2012 | A scaled mapping approach for treating sloping interfaces in range-dependent parabolic equation solutions (MSc thesis) | Colorado School of Mines | — (repository.mines.edu) | moran_2012_scaled_mapping_thesis.pdf |
| metzler2014scaled | A. M. Metzler, D. Moran, J. M. Collis, P. A. Martin, W. L. Siegmann | 2014 | A scaled mapping parabolic equation for sloping range-dependent environments | JASA 135(3):EL172–EL178 | 10.1121/1.4865265 | not downloaded |
| kusel2007single | E. T. Küsel, W. L. Siegmann, M. D. Collins | 2007 | A single-scattering correction for large contrasts in elastic layers | JASA 121(2):808–813 | 10.1121/1.2404627 | not downloaded |
| collins2012single | M. D. Collins | 2012 | A single-scattering correction for the seismo-acoustic parabolic equation | JASA 131(4):2638–2642 | 10.1121/1.3689557 | not downloaded |
| collins2015sloping | M. D. Collins, W. L. Siegmann | 2015 | Treatment of a sloping fluid-solid interface and sediment layering with the seismo-acoustic parabolic equation | JASA 137(1):492–497 | 10.1121/1.4904526 | not downloaded |
| woolfe2016benchmark | K. Woolfe, M. D. Collins, D. C. Calvo, W. L. Siegmann | 2016 | Seismo-acoustic benchmark problems involving sloping fluid–solid interfaces | J. Comput. Acoust. 24(4):1650022 | 10.1142/S0218396X16500223 | not downloaded |
| collins2020sloping | M. D. Collins, A. Ramamurti | 2020 | Approaches for handling sloping fluid-solid interfaces with the parabolic equation method | J. Theor. Comput. Acoust. (arXiv:2005.10748) | arXiv:2005.10748 | collins_ramamurti_2020_sloping_fluid_solid_arxiv.pdf |
| collins2019book | M. D. Collins, W. L. Siegmann | 2019 | Parabolic Wave Equations with Applications | Springer | 10.1007/978-1-4939-9934-7 | not downloaded |
| abrahamsson1990bc | L. Abrahamsson, H.-O. Kreiss | 1990 | Boundary conditions for the parabolic equation in a range-dependent duct | JASA 87(6):2438–2441 | 10.1121/1.399089 | not downloaded |
| abrahamsson1990ibvp | L. Abrahamsson, H.-O. Kreiss | 1990 | The initial boundary value problem for the Schrödinger equation | Math. Meth. Appl. Sci. 13(5):385–390 | 10.1002/mma.1670130503 | not downloaded |
| abrahamsson1991grid | L. Abrahamsson | 1991 | Orthogonal grid generation for two-dimensional ducts | J. Comput. Appl. Math. 34(3):305–314 | 10.1016/0377-0427(91)90091-W | not downloaded (Elsevier open archive, blocked from script) |
| abrahamsson1994coupled | L. Abrahamsson, H.-O. Kreiss | 1994 | Numerical solution of the coupled mode equations in duct acoustics | J. Comput. Phys. 111(1):1–14 | 10.1006/jcph.1994.1038 | not downloaded |
| akrivis1990variable | G. D. Akrivis, V. A. Dougalis | 1990 | Finite difference discretization with variable mesh of the Schrödinger equation in a variable domain | Bull. Greek Math. Soc. 31:19–28 | — | akrivis_dougalis_1990_schrodinger_variable_domain.pdf |
| akrivis1994galerkin | G. D. Akrivis, V. A. Dougalis, N. A. Kampanis | 1994 | On Galerkin methods for the wide-angle parabolic equation | J. Comput. Acoust. 2(2):99–112 | 10.1142/S0218396X94000087 | akrivis_dougalis_kampanis_1994_galerkin_wide_angle.pdf |
| akrivis1994fem | G. D. Akrivis, V. A. Dougalis, N. A. Kampanis | 1994 | Error estimates for finite element methods for a wide-angle parabolic equation | Appl. Numer. Math. 16(1–2):81–100 | 10.1016/0168-9274(94)00046-8 | akrivis_dougalis_kampanis_1994_fem_wide_angle.pdf |
| akrivis1996fd | G. D. Akrivis, V. A. Dougalis, G. E. Zouraris | 1996 | Error estimates for finite difference methods for a wide-angle "parabolic" equation | SIAM J. Numer. Anal. 33(6):2488–2509 | 10.1137/S0036142994266352 | akrivis_dougalis_zouraris_1996_fd_wide_angle.pdf |
| akrivis2001rigid | G. D. Akrivis, V. A. Dougalis, G. E. Zouraris | 2001 | Finite difference schemes for the "parabolic" equation in a variable depth environment with a rigid bottom boundary condition | SIAM J. Numer. Anal. 39(2):539–565 | 10.1137/S0036142999367460 | akrivis_dougalis_zouraris_2001_fd_variable_depth_rigid.pdf |
| dougalis1996fem | V. A. Dougalis, N. A. Kampanis | 1996 | Finite element methods for the parabolic equation with interfaces | J. Comput. Acoust. 4(1):55–88 | 10.1142/S0218396X96000222 | not downloaded |
| kampanis1999helmholtz | N. A. Kampanis, V. A. Dougalis | 1999 | A finite element code for the numerical solution of the Helmholtz equation in axially symmetric waveguides with interfaces | J. Comput. Acoust. 7(2):83–110 | 10.1142/S0218396X99000084 | not downloaded |
| sturm2007sloping | F. Sturm, N. A. Kampanis | 2007 | Accurate treatment of a general sloping interface in a finite-element 3D narrow-angle PE model | J. Comput. Acoust. 15(3):285–318 | 10.1142/S0218396X07003366 | not downloaded |
| dougalis2008chapter | V. A. Dougalis, N. A. Kampanis, F. Sturm, G. E. Zouraris | 2008 | Numerical solution of the parabolic equation in range-dependent waveguides | In: Effective Computational Methods for Wave Propagation, CRC, pp. 175–207 | 10.1201/9781420010879.ch6 | not downloaded |
| dougalis2009wa | V. A. Dougalis, F. Sturm, G. E. Zouraris | 2009 | On an initial-boundary value problem for a wide-angle parabolic equation in a waveguide with a variable bottom | Math. Meth. Appl. Sci. 32(12):1519–1540 | 10.1002/mma.1097 | not downloaded |
| antonopoulou2009dynamical | D. C. Antonopoulou, V. A. Dougalis, G. E. Zouraris | 2009 | Galerkin methods for parabolic and Schrödinger equations with dynamical boundary conditions and applications to underwater acoustics | SIAM J. Numer. Anal. 47(4):2752–2781 | 10.1137/070710858 | antonopoulou_dougalis_zouraris_2009_dynamical_bc.pdf (arXiv:0904.3900) |
| antonopoulou2013downslope | D. C. Antonopoulou, V. A. Dougalis, G. E. Zouraris | 2013 | A finite difference method for the wide-angle "parabolic" equation in a waveguide with downsloping bottom | Numer. Meth. PDE 29(4):1416–1440 | 10.1002/num.21762 | antonopoulou_dougalis_zouraris_2013_downsloping_bottom.pdf (arXiv:1110.3438) |
| lee1983irregular | D. Lee, S. T. McDaniel | 1983 | A finite-difference treatment of interface conditions for the parabolic wave equation: the irregular interface | JASA 73(5):1441–1447 | 10.1121/1.389431 | not downloaded |
| evans1983couple | R. B. Evans | 1983 | A coupled mode solution for acoustic propagation in a waveguide with stepwise depth variations of a penetrable bottom | JASA 74(1):188–195 | 10.1121/1.389707 | not downloaded |
| jensen1990benchmark | F. B. Jensen, C. M. Ferla | 1990 | Numerical solutions of range-dependent benchmark problems in ocean acoustics | JASA 87(4):1499–1510 | 10.1121/1.399448 | not downloaded |
| greene1985sloping | R. R. Greene | 1985 | A high-angle one-way wave equation for seismic wave propagation along rough and sloping interfaces | JASA 77(6):1991–1998 | 10.1121/1.391770 | not downloaded |
| godin1999reciprocity | O. A. Godin | 1999 | Reciprocity and energy conservation within the parabolic approximation | Wave Motion 29(2):175–194 | 10.1016/S0165-2125(98)00025-0 | not downloaded |
| lu1999exact | Y. Y. Lu | 1999 | Exact one-way methods for acoustic waveguides | Math. Comput. Simul. 50(5–6):377–391 | 10.1016/S0378-4754(99)00111-1 | not downloaded |
| lu2001lot | Y. Y. Lu, J. Huang, J. R. McLaughlin | 2001 | Local orthogonal transformation and one-way methods for acoustic waveguides | Wave Motion 34(2):193–207 | 10.1016/S0165-2125(00)00083-4 | not downloaded |
| lu2004lot | Y. Y. Lu, J. Zhu | 2004 | A local orthogonal transform for acoustic waveguides with an internal interface | J. Comput. Acoust. 12(1):37–53 | 10.1142/S0218396X04002183 | not downloaded |
| oba2010conformal | R. M. Oba | 2010 | Global boundary flattening transforms for acoustic propagation under rough sea surfaces | JASA 128(1):39–49 | 10.1121/1.3445783 | not downloaded |
| lin2019boundary | Y.-T. Lin | 2019 | Three-dimensional boundary fitted parabolic-equation model of underwater sound propagation | JASA 146(3):2058–2067 | 10.1121/1.5126011 | not downloaded (WHOI author copy sits behind a captcha) |

**C. Conformal maps, Earth flattening, curvilinear and sigma coordinates**

| key | authors | year | title | venue | DOI | local file |
|---|---|---|---|---|---|---|
| heiblum1975conformal | M. Heiblum, J. H. Harris | 1975 | Analysis of curved optical waveguides by conformal transformation | IEEE J. Quantum Electron. 11(2):75–83 (corr. 12(5):313, 1976) | 10.1109/JQE.1975.1068563 | not downloaded |
| pekeris1946earthflattening | C. L. Pekeris | 1946 | Accuracy of the earth-flattening approximation in the theory of microwave propagation | Phys. Rev. 70:518–522 | 10.1103/PhysRev.70.518 | not downloaded |
| koo1960exact | B. Y.-C. Koo, M. Katzin | 1960 | An exact earth-flattening procedure in propagation around a sphere | J. Res. NBS 64D(1):61–64 | 10.6028/jres.064D.011 | koo_katzin_1960_exact_earth_flattening.pdf |
| janaswamy1994report | R. Janaswamy | 1994 | Efficient parabolic equation solution of radiowave propagation in an inhomogeneous atmosphere and over irregular terrain: formulation | NPS Tech. Rep. NPS-EC-94-004 | hdl:10945/28720 | janaswamy_1994_nps_curvilinear_pe_report.pdf |
| janaswamy1998curvilinear | R. Janaswamy | 1998 | A curvilinear coordinate-based split-step parabolic equation method for propagation predictions over terrain | IEEE Trans. Antennas Propag. 46(7):1089–1097 | 10.1109/8.704813 | not downloaded |
| sack1995gtpe | R. A. Sack, M. West | 1995 | A parabolic equation for sound propagation in two dimensions over any smooth terrain profile: the generalised terrain parabolic equation (GT-PE) | Applied Acoustics 45(2):113–129 | 10.1016/0003-682X(94)00039-X | not downloaded |
| salomons2001book | E. M. Salomons | 2001 | Computational Atmospheric Acoustics | Kluwer (book) | 10.1007/978-94-010-0660-6 | not downloaded |
| digilbertyou1992grid | X. Di, K. E. Gilbert, C. You | 1992 | Application of numerical grid generation to sound propagation over irregular terrain | JASA 92(4 Suppl.):2431 (abstract) | 10.1121/1.404634 | not downloaded |
| you1993polar | C. You, H. E. Bass, K. E. Gilbert | 1993 | Comparisons of experimental measurements of sound propagation over a hill and the polar parabolic equation method | JASA 94(3 Suppl.):1875 (abstract; see also You, PhD thesis, U. Mississippi 1993) | 10.1121/1.407585 | not downloaded |
| digilbert1994turb | X. Di, K. E. Gilbert | 1994 | Atmospheric sound propagation in a turbulent atmosphere over irregular terrain | JASA 96(5 Suppl.):3276 (abstract) | 10.1121/1.410970 | not downloaded |
| digilbert2005mapping | X. Di, K. E. Gilbert | 2005 | Comparison of two mapping methods for computing sound propagation over irregular terrain | JASA 117(4 Suppl.):2594 (abstract) | 10.1121/1.4777407 | not downloaded |
| parakkal2006cascaded | S. Parakkal, K. E. Gilbert, X. Di | 2006 | A continuously cascaded conformal mapping for sound propagation over irregular terrain | JASA 119(5 Suppl.):3263 (abstract) | 10.1121/1.4786109 | not downloaded |
| parakkal2009three | S. Parakkal, K. E. Gilbert, X. Di | 2009 | Comparison of three coordinate mapping methods for sound propagation over irregular terrain | JASA 126(4 Suppl.):2160 (abstract) | 10.1121/1.3248395 | not downloaded |
| parakkal2010polar | S. Parakkal, K. E. Gilbert, X. Di, H. E. Bass | 2010 | A generalized polar coordinate method for sound propagation over large-scale irregular terrain | JASA 128(5):2573–2580 | 10.1121/1.3495941 | not downloaded |
| khodr2020iterative | C. Khodr, M. Azarpeyvand, D. N. Green | 2020 | An iterative three-dimensional parabolic equation solver for propagation above irregular boundaries | JASA 148(2):1089–1100 | 10.1121/10.0001766 | khodr_2020_3dpe_terrain_bristol.pdf |
| colas2023wapetopo | J. Colas, A. Emmanuelli, D. Dragna, P. Blanc-Benon, B. Cotté, R. Stevens | 2023 | Wind turbine sound propagation: comparison of a linearized Euler equations model with parabolic equation methods | arXiv:2309.09669 (later JASA) | 10.48550/arXiv.2309.09669 | colas_etal_2023_wapetopo_wind_turbine_arxiv.pdf |
| galchen1975coord | T. Gal-Chen, R. C. J. Somerville | 1975 | On the use of a coordinate transformation for the solution of the Navier–Stokes equations | J. Comput. Phys. 17(2):209–228 | 10.1016/0021-9991(75)90037-6 | not downloaded |
| novikov2011conformal | A. V. Novikov, Yu. P. Akulinichev | 2011 | Conformal mapping as a method for taking into account small-slope terrain irregularities in the numerical solution of the parabolic equation (in Russian) | Doklady TUSUR 1(23):55–59 | — | novikov_akulinichev_2011_conformal_pe_terrain.pdf |
| guo2017pade | Q. Guo, Y. Long | 2017 | Padé second-order parabolic equation modeling for propagation over irregular terrain | IEEE AWPL 16 (2017; pages unverified) | 10.1109/LAWP.2017.2749603 | not downloaded |
| kirby1988nonorth | J. T. Kirby | 1988 | Parabolic wave computations in non-orthogonal coordinate systems | J. Waterway Port Coastal Ocean Eng. 114(6):673–685 | 10.1061/(ASCE)0733-950X(1988)114:6(673) | not downloaded |
| kirby1994conformal | J. T. Kirby, R. A. Dalrymple, H. Kaku | 1994 | Parabolic approximations for water waves in conformal coordinate systems | Coastal Engineering 23(3–4):185–213 | 10.1016/0378-3839(94)90001-9 | not downloaded |
| baets1983curved | R. Baets, P. E. Lagasse | 1983 | Loss calculation and design of arbitrarily curved integrated-optic waveguides | JOSA 73(2):177–182 | 10.1364/JOSA.73.000177 | not downloaded |
| rivera1995bent | M. Rivera | 1995 | A finite difference BPM analysis of bent dielectric waveguides | J. Lightwave Technol. 13(2):233–238 | 10.1109/50.365211 | not downloaded |
| lu2002bent | Y. Y. Lu, P. L. Ho | 2002 | Beam propagation modeling of arbitrarily bent waveguides | IEEE Photon. Technol. Lett. 14(12):1698–1700 | 10.1109/LPT.2002.804670 | not downloaded |
| shyroki2006exact | D. M. Shyroki | 2006 | Exact equivalent-profile formulation for bent optical waveguides | arXiv:physics/0605002 | 10.48550/arXiv.physics/0605002 | shyroki_2006_exact_equivalent_profile_bent.pdf |
| shyroki2008exact | D. M. Shyroki | 2008 | Exact equivalent straight waveguide model for bent and twisted waveguides | IEEE Trans. Microw. Theory Tech. 56(2):414–419 | 10.1109/TMTT.2007.914637 | not downloaded |
| hadley1992pade | G. R. Hadley | 1992 | Wide-angle beam propagation using Padé approximant operators | Optics Letters 17(20):1426–1428 | 10.1364/OL.17.001426 | not downloaded |
| ilic1996pade | I. Ilić, R. Scarmozzino, R. M. Osgood | 1996 | Investigation of the Padé approximant-based wide-angle beam propagation method for accurate modeling of waveguiding circuits | J. Lightwave Technol. 14(12):2813–2822 | 10.1109/50.545802 | not downloaded |
| desanto1977relation | J. A. DeSanto | 1977 | Relation between the solutions of the Helmholtz and parabolic equations for sound propagation | JASA 62(2):295–297 | 10.1121/1.381527 | not downloaded |
| driscoll1996sc | T. A. Driscoll | 1996 | Algorithm 756: a MATLAB toolbox for Schwarz–Christoffel mapping | ACM Trans. Math. Softw. 22(2):168–186 | 10.1145/229473.229475 | not downloaded |
| driscoll2002scbook | T. A. Driscoll, L. N. Trefethen | 2002 | Schwarz–Christoffel Mapping | Cambridge Univ. Press | 10.1017/CBO9780511546808 | not downloaded |
| trefethen2020ncm | L. N. Trefethen | 2020 | Numerical conformal mapping with rational functions | Comput. Methods Funct. Theory 20:369–387 | 10.1007/s40315-020-00325-w | trefethen_2020_numerical_conformal_mapping_rational.pdf |

**D. Transformation-EM and recent (2005–2026) terrain PE; Lytaev works**

| key | authors | year | title | venue | DOI | local file |
|---|---|---|---|---|---|---|
| janaswamy2025rsl | R. Janaswamy | 2025 | Rigorous Formulation of Vector Wave Propagation Over 3D Irregular Terrain | URSI Radio Science Letters 7 | 10.46620/25-0039 | janaswamy_2025_rsl_vector_terrain_sigma_transform.pdf |
| ozgun2010domain | O. Ozgun, M. Kuzuoglu | 2010 | Domain compression via anisotropic metamaterials designed by coordinate transformations | J. Comput. Phys. 229(3) 921–932 | 10.1016/j.jcp.2009.10.023 | not downloaded |
| ozgun2013tmtt | O. Ozgun, M. Kuzuoglu | 2013 | Transformation Electromagnetics Based Analysis of Waveguides With Random Rough or Periodic Grooved Surfaces | IEEE Trans. MTT 61(2) 709–719 | 10.1109/TMTT.2012.2231428 | not downloaded |
| ozgun2013softmeta | O. Ozgun, M. Kuzuoglu | 2013 | Software metamaterials: Transformation media based multiscale techniques for computational electromagnetics | J. Comput. Phys. 236 203–219 | 10.1016/j.jcp.2012.11.015 | not downloaded |
| kuzuoglu2014perturb | M. Kuzuoglu, O. Ozgun | 2014 | Combining perturbation theory and transformation electromagnetics for finite element solution of Helmholtz-type scattering problems | J. Comput. Phys. 274 883–897 | 10.1016/j.jcp.2014.06.057 | not downloaded |
| ozgun2009recursive | O. Ozgun | 2009 | Recursive Two-Way Parabolic Equation Approach for Modeling Terrain Effects in Tropospheric Propagation | IEEE TAP 57(9) 2706–2714 | 10.1109/TAP.2009.2027166 | not downloaded |
| ozgun2011petool | O. Ozgun, G. Apaydin, M. Kuzuoglu, L. Sevgi | 2011 | PETOOL: MATLAB-based one-way and two-way split-step parabolic equation tool for radiowave propagation over variable terrain | Comput. Phys. Commun. 182(12) 2638–2654 | 10.1016/j.cpc.2011.07.017 | not downloaded |
| ozgun2020petool2 | O. Ozgun, V. Sahin, M. E. Erguden, G. Apaydin, A. E. Yilmaz, M. Kuzuoglu | 2020 | PETOOL v2.0: Parabolic Equation Toolbox with evaporation duct models and real environment data | Comput. Phys. Commun. 256 107454 | 10.1016/j.cpc.2020.107454 | not downloaded |
| apaydin2011fempe | G. Apaydin, O. Ozgun, M. Kuzuoglu, L. Sevgi | 2011 | A Novel Two-Way Finite-Element Parabolic Equation Groundwave Propagation Tool | IEEE TGRS 49(8) 2887–2899 | 10.1109/TGRS.2011.2114889 | not downloaded |
| apaydin2010map | G. Apaydin, L. Sevgi | 2010 | A Novel Split-Step Parabolic-Equation Package for Surface-Wave Propagation Prediction Along Multiple Mixed Irregular-Terrain Paths | IEEE AP Magazine 52(4) 90–97 | 10.1109/MAP.2010.5638238 | not downloaded |
| apaydin2017book | G. Apaydin, L. Sevgi | 2017 | Radio Wave Propagation and Parabolic Equation Modeling (book, ch. 5 "PE and Terrain Modeling") | Wiley-IEEE Press | 10.1002/9781119432166 | not downloaded |
| holm2007wasm | P. D. Holm | 2007 | Wide-Angle Shift-Map PE for a Piecewise Linear Terrain—A Finite-Difference Approach | IEEE TAP 55(10) 2773–2789 | 10.1109/TAP.2007.905865 | not downloaded |
| holm2009aip | P. D. Holm, B. Nilsson, L. Fishman, A. Karlsson, S. Nordebo | 2009 | Wide-Angle Shift-Map PE for a Piecewise Linear Terrain | AIP Conf. Proc. 1106, 56–65 | 10.1063/1.3117113 | not downloaded |
| zhou2015greene | C. Zhou, Q. Zeng, W. Ding, Y. Long | 2015 | Greene second-order approximation of the wide-angle equation for terrain | APMC 2015 | 10.1109/APMC.2015.7413341 | not downloaded |
| guo2017greene | Q. Guo, C. Zhou, Y. Long | 2017 | Greene Approximation Wide-Angle Parabolic Equation for Radio Propagation | IEEE TAP 65(11) 6048–6056 | 10.1109/TAP.2017.2748222 | not downloaded |
| guo2018twoway | Q. Guo, Z. Li, Y. Long | 2018 | Two-Way Parabolic Equation Modeling for Radio Propagation Over Irregular Terrain | ICMMT 2018 | 10.1109/ICMMT.2018.8563284 | not downloaded |
| wang2012twoway | K. Wang, Y. Long | 2012 | Propagation Modeling Over Irregular Terrain by the Improved Two-Way Parabolic Equation Method | IEEE TAP 60(9) 4467–4471 | 10.1109/TAP.2012.2207063 | not downloaded |
| wang2019hybrid | D.-D. Wang, X.-L. Xi, Y.-R. Pu, J.-S. Zhang, L.-L. Zhou | 2019 | LF Groundwave Propagation Modeling Over Irregular Terrain by the Hybrid Two-Way Parabolic Equation Method | IEEE TAP 67(10) 6596–6601 | 10.1109/TAP.2019.2925155 | not downloaded |
| wang2020smpe | D.-D. Wang, Y.-R. Pu, X.-L. Xi, L.-L. Zhou | 2020 | An Analysis of Narrow-Angle and Wide-Angle Shift-Map Parabolic Equations | IEEE TAP 68(5) 3911–3918 | 10.1109/TAP.2020.2963903 | not downloaded |
| zhang2024wasm | N. Zhang, Z. Liang, Y. Long | 2024 | An Efficient Difference Scheme for Wide-Angle Shift-Map Parabolic Equation | IEEE TAP 72(4) 3545–3552 | 10.1109/TAP.2024.3363496 | not downloaded |
| zhang2024techrxiv | N. Zhang, Y. Long | 2024 | New Shift-Map Parabolic Equations Based on Finite Difference Method | TechRxiv preprint | 10.36227/techrxiv.170619205.59357076/v1 | not downloaded (TechRxiv blocks scripted download) |
| zhangr2024maxangle | R. Zhang, H. Wang, Z. Wu, L. Lin, Z. Zhao | 2024 | Prediction Method of Maximum Propagation Angle in Parabolic Equation Model over Irregular Terrain | Int. J. Antennas Propag. 2024, 4988555 | 10.1155/2024/4988555 | not downloaded (open access; Wiley blocked the scripted download) |
| gao2019roughsea | Y. Gao, Q. Shao, B. Yan, Q. Li, S. Guo | 2019 | Parabolic Equation Modeling of Electromagnetic Wave Propagation over Rough Sea Surfaces | Sensors 19(5) 1252 | 10.3390/s19051252 | not downloaded (open access; read online via PMC6427321) |
| silva2012_3dpe | M. A. N. Silva, E. Costa, M. Liniger | 2012 | Analysis of the Effects of Irregular Terrain on Radio Wave Propagation Based on a Three-Dimensional Parabolic Equation | IEEE TAP 60(4) 2138–2143 | 10.1109/TAP.2012.2186227 | not downloaded |
| eibert2002 | T. F. Eibert | 2002 | Irregular terrain wave propagation by a Fourier split-step wide-angle parabolic wave equation technique for linearly bridged knife-edges | Radio Sci. 37(1) | 10.1029/2000RS002560 | not downloaded |
| kuttler2002 | J. R. Kuttler, R. Janaswamy | 2002 | Improved Fourier transform methods for solving the parabolic wave equation | Radio Sci. 37(2) | 10.1029/2001RS002488 | not downloaded |
| bonnafont2021lssw | T. Bonnafont, R. Douvenot, A. Chabory | 2021 | A Local Split-Step Wavelet Method for the Long Range Propagation Simulation in 2D | Radio Sci. 56(2) e2020RS007114 | 10.1029/2020RS007114 | bonnafont_2021_local_ssw.pdf (HAL hal-03121383) |
| bonnafont2020thesis | T. Bonnafont | 2020 | Modeling the atmospheric long-range electromagnetic waves propagation in 3D using the wavelet transform (PhD thesis, Univ. Toulouse) | HAL tel-04028199 | — (NNT 2020TOU30173) | bonnafont_2020_phd_thesis_ssw_3d.pdf |
| bonnafont2022twoway | T. Bonnafont, O. Benhmammouch, A. Khenchaf | 2022 | A Two-Way Split-Step Wavelet Scheme for Tropospheric Long-Range Propagation in Various Environments | Remote Sensing 14(11) 2686 | 10.3390/rs14112686 | bonnafont_2022_two_way_ssw.pdf (HAL hal-03693962) |
| khodr2020thesis | C. Khodr | 2020 | Three-dimensional infrasonic wave propagation above irregular boundaries (PhD thesis, Univ. Bristol) | Univ. Bristol | — | khodr_2020_phd_thesis_bristol.pdf |
| khodr2022gji | C. Khodr, D. N. Green, M. Azarpeyvand | 2022 | Three-dimensional topographic effects on infrasound propagation across Ascension Island | Geophys. J. Int. 231(3) 1558–1572 | 10.1093/gji/ggac230 | not downloaded |
| nyborg2023 | C. M. Nyborg, K. Bolin, I. Karasalo, A. Fischer | 2023 | An inter-model comparison of parabolic equation methods for sound propagation from wind turbines | JASA 154(2) 1299–1314 | 10.1121/10.0020562 | not downloaded (open access, AIP blocked scripted download) |
| kampanis2002 | N. A. Kampanis | 2002 | Numerical simulation of low-frequency aeroacoustics over irregular terrain using a finite element discretization of the parabolic equation | J. Comput. Acoust. 10(1) 97–111 | 10.1142/S0218396X02001413 | not downloaded |
| kampanis2005 | N. A. Kampanis | 2005 | A finite element method for the parabolic equation in aeroacoustics coupled with a nonlocal boundary condition for an inhomogeneous atmosphere | J. Comput. Acoust. 13(4) 569–584 | 10.1142/S0218396X05002864 | not downloaded |
| kampanis2013 | N. A. Kampanis, A. I. Delis, D. C. Antonopoulou, G. Kozyrakis | 2013 | A finite element discretization of the standard parabolic equation in generalized boundary fitting coordinates | Appl. Numer. Math. 67 152–166 | 10.1016/j.apnum.2011.05.005 | not downloaded |
| swearingen2017 | M. E. Swearingen, M. J. White, M. H. McKenna | 2017 | A wide-angle topography-capable parabolic equation using a non-transformed coordinate system (abstract) | JASA 141(5 Suppl.) | 10.1121/1.4988705 | not downloaded (abstract only) |
| lytaev2018awpl | M. S. Lytaev | 2018 | Nonlocal Boundary Conditions for Split-Step Padé Approximations of the Helmholtz Equation With Modified Refractive Index | IEEE AWPL 17(8) 1561–1565 | 10.1109/LAWP.2018.2855086 | ../../AWPL2018.pdf (existing local copy) |
| lytaev2018rtuwo | M. S. Lytaev, A. G. Vladyko | 2018 | Split-step Padé Approximations of the Helmholtz Equation for Radio Coverage Prediction over Irregular Terrain | RTUWO 2018, 179–184 | 10.1109/RTUWO.2018.8587886 | not downloaded |
| lytaev2020awpl | M. S. Lytaev | 2020 | Numerov-Padé Scheme for the One-Way Helmholtz Equation in Tropospheric Radio-Wave Propagation | IEEE AWPL 19(12) 2167–2171 | 10.1109/LAWP.2020.3026626 | ../../awpl2020.pdf (existing local copy) |
| lytaev2021iccsa | M. S. Lytaev | 2021 | An Improved Accuracy Split-Step Padé Parabolic Equation for Tropospheric Radio-Wave Propagation | ICCSA 2021, LNCS 12949, 418–433 | 10.1007/978-3-030-86653-2_31 | not downloaded |
| lytaev2022jocs | M. S. Lytaev | 2022 | Rational interpolation of the one-way Helmholtz propagator | J. Comput. Sci. 58 101536 | 10.1016/j.jocs.2021.101536 | not downloaded |
| lytaev2022radarconf | M. S. Lytaev | 2022 | Fresnel Reflection Modeling Within the Higher-order Parabolic Equation and Discrete Nonlocal Boundary Conditions | IEEE RadarConf22 | 10.1109/RadarConf2248738.2022.9764174 | ../../radarconf22.pdf (existing local copy) |
| lytaev2023jmse | M. Lytaev | 2023 | Mesh Optimization for the Acoustic Parabolic Equation | J. Mar. Sci. Eng. 11(3) 496 | 10.3390/jmse11030496 | not downloaded (open access; MDPI blocked scripted download) |
| lytaev2023jocs | M. S. Lytaev | 2023 | Reducing the numerical dispersion of the one-way Helmholtz equation via the differential evolution method | J. Comput. Sci. 71 102057 | 10.1016/j.jocs.2023.102057 | not downloaded |
| lytaev2026nmp | M. S. Lytaev | 2026 | Mesh optimization for rational approximations of the one-way Helmholtz equation in inhomogeneous media | Numer. Methods Program. 27(2) 302–316 | 10.26089/NumMet.v27r220 | lytaev_2026_mesh_optimization_inhomogeneous.pdf |

**E. Transparent / nonlocal boundary conditions (topic 5) and discrete dispersion analysis (topic 6)**

| key | authors | year | title | venue | DOI | local file |
|---|---|---|---|---|---|---|
| arnold1998vlsi | A. Arnold | 1998 | Numerically absorbing boundary conditions for quantum evolution equations | VLSI Design 6:313–319 | 10.1155/1998/38298 | not downloaded (Hindawi 403) |
| arnold_ehrhardt1998jcp | A. Arnold, M. Ehrhardt | 1998 | Discrete transparent boundary conditions for wide angle parabolic equations in underwater acoustics | J. Comput. Phys. 145:611–638 | 10.1006/jcph.1998.6043 | not downloaded |
| ehrhardt1999vlsi | M. Ehrhardt | 1999 | Discrete transparent boundary conditions for general Schrödinger-type equations | VLSI Design 9:325–338 | 10.1155/1999/72984 | not downloaded (Hindawi 403) |
| arnold_ehrhardt_sofronov2003 | A. Arnold, M. Ehrhardt, I. Sofronov | 2003 | Discrete transparent boundary conditions for the Schrödinger equation: fast calculation, approximation, and stability | Commun. Math. Sci. 1:501–556 | 10.4310/cms.2003.v1.n3.a7 | arnold_ehrhardt_sofronov_2003_dtbc_fast.pdf |
| ehrhardt_mickens2004 | M. Ehrhardt, R.E. Mickens | 2004 | Solutions to the discrete Airy equation: application to parabolic equation calculations | J. Comput. Appl. Math. 172:183–206 | 10.1016/j.cam.2004.02.011 | ehrhardt_mickens_2004_discrete_airy_tbc_preprint.pdf (preprint) |
| zisowsky_ehrhardt2006 | A. Zisowsky, M. Ehrhardt | 2006 | Discrete transparent boundary conditions for parabolic systems | Math. Comput. Modelling 43:294–309 | 10.1016/j.mcm.2005.08.011 | zisowsky_ehrhardt_dtbc_parabolic_systems_preprint.pdf (preprint) |
| ehrhardt_zisowsky2006pamm | M. Ehrhardt, A. Zisowsky | 2006 | Nonlocal boundary conditions for higher-order parabolic equations | PAMM 6:733–734 | 10.1002/pamm.200610347 | ehrhardt_zisowsky_2006_nonlocal_bc_proc.pdf |
| ehrhardt_zisowsky2007 | M. Ehrhardt, A. Zisowsky | 2007 | Discrete non-local boundary conditions for split-step Padé approximations of the one-way Helmholtz equation | J. Comput. Appl. Math. 200:471–490 | 10.1016/j.cam.2006.01.001 | ehrhardt_zisowsky_2007_dnlbc_split_step_pade_preprint.pdf (preprint) |
| antoine2008review | X. Antoine, A. Arnold, C. Besse, M. Ehrhardt, A. Schädle | 2008 | A review of transparent and artificial boundary conditions techniques for linear and nonlinear Schrödinger equations | Commun. Comput. Phys. 4:729–796 | 10.4208/cicp.2008.v4.p729 | antoine_etal_2008_review_tbc_schrodinger_zib_preprint.pdf (ZIB report 07-34) |
| schulte_arnold2008 | M. Schulte, A. Arnold | 2008 | Discrete transparent boundary conditions for the Schrödinger equation — a compact higher order scheme | Kinet. Relat. Models 1:101–125 | 10.3934/krm.2008.1.101 | not downloaded |
| arnold2012circular | A. Arnold, M. Ehrhardt, M. Schulte, I. Sofronov | 2012 | Discrete transparent boundary conditions for the Schrödinger equation on circular domains | Commun. Math. Sci. 10:889–916 | 10.4310/cms.2012.v10.n3.a9 | arnold_etal_2012_dtbc_circular.pdf |
| zlotnik_romanova2015 | A. Zlotnik, A. Romanova | 2015 | On a Numerov–Crank–Nicolson–Strang scheme with discrete transparent boundary conditions for the Schrödinger equation on a semi-infinite strip | Appl. Numer. Math. 93:279–294 | 10.1016/j.apnum.2014.05.003 | zlotnik_romanova_2015_numerov_cn_dtbc.pdf (arXiv 1307.5398) |
| antoine_besse_mouysset2004 | X. Antoine, C. Besse, V. Mouysset | 2004 | Numerical schemes for the simulation of the two-dimensional Schrödinger equation using non-reflecting boundary conditions | Math. Comp. 73:1779–1799 | 10.1090/s0025-5718-04-01631-x | not downloaded |
| antoine_besse_klein2013 | X. Antoine, C. Besse, P. Klein | 2013 | Absorbing boundary conditions for the two-dimensional Schrödinger equation with an exterior potential | Numer. Math. 125:191–223 | 10.1007/s00211-013-0542-8 | not downloaded |
| baskakov_popov1991 | V.A. Baskakov, A.V. Popov | 1991 | Implementation of transparent boundaries for numerical solution of the Schrödinger equation | Wave Motion 14:123–128 | 10.1016/0165-2125(91)90053-q | not downloaded |
| marcus1991 | S.W. Marcus | 1991 | A generalized impedance method for application of the parabolic approximation to underwater acoustics | J. Acoust. Soc. Am. 90:391–398 | 10.1121/1.401263 | not downloaded |
| papadakis1992 | J.S. Papadakis, M.I. Taroudakis, P.J. Papadakis, B. Mayfield | 1992 | A new method for a realistic treatment of the sea bottom in the parabolic approximation | J. Acoust. Soc. Am. 92:2030–2038 | 10.1121/1.405255 | not downloaded |
| papadakis1994 | J.S. Papadakis | 1994 | Exact, nonreflecting boundary conditions for parabolic-type approximations in underwater acoustics | J. Comput. Acoust. 2:83–98 | 10.1142/s0218396x94000075 | not downloaded |
| popov1996 | A.V. Popov | 1996 | Accurate modeling of transparent boundaries in quasi-optics | Radio Sci. 31:1781–1790 | 10.1029/96rs02538 | not downloaded |
| levy1997 | M.F. Levy | 1997 | Transparent boundary conditions for parabolic equation solutions of radiowave propagation problems | IEEE Trans. Antennas Propag. 45:66–72 | 10.1109/8.554242 | not downloaded |
| yevick_thomson1999 | D. Yevick, D.J. Thomson | 1999 | Nonlocal boundary conditions for finite-difference parabolic equation solvers | J. Acoust. Soc. Am. 106:143–150 | 10.1121/1.427043 | not downloaded |
| brooke_thomson2000 | G.H. Brooke, D.J. Thomson | 2000 | Non-local boundary conditions for high-order parabolic equation algorithms | Wave Motion 31:117–129 | 10.1016/s0165-2125(99)00039-6 | not downloaded |
| schmidt_deuflhard1995 | F. Schmidt, P. Deuflhard | 1995 | Discrete transparent boundary conditions for the numerical solution of Fresnel's equation | Comput. Math. Appl. 29:53–76 | 10.1016/0898-1221(95)00037-y | not downloaded |
| schmidt_friese_yevick2001 | F. Schmidt, T. Friese, D. Yevick | 2001 | Transparent boundary conditions for split-step Padé approximations of the one-way Helmholtz equation | J. Comput. Phys. 170:696–719 | 10.1006/jcph.2001.6757 | not downloaded |
| zebic2001 | A. Zebic-Le Hyaric | 2001 | Wide-angle nonlocal boundary conditions for the parabolic wave equation | IEEE Trans. Antennas Propag. 49:916–922 | 10.1109/8.931149 | not downloaded |
| levy2001pml | M.F. Levy | 2001 | Perfectly matched layer truncation for parabolic wave equation models | Proc. R. Soc. A 457:2609–2624 | 10.1098/rspa.2001.0848 | not downloaded |
| mikhin2004 | D. Mikhin | 2004 | Exact discrete nonlocal boundary conditions for high-order Padé parabolic equations | J. Acoust. Soc. Am. 116:2864–2875 | 10.1121/1.1802811 | not downloaded |
| mikhin2008 | D. Mikhin | 2008 | Analytic discrete transparent boundary conditions for high-order Padé parabolic equations | Wave Motion 45:881–894 | 10.1016/j.wavemoti.2008.03.006 | not downloaded |
| feshchenko_popov2017 | R.M. Feshchenko, A.V. Popov | 2017 | Exact transparent boundary conditions for the parabolic wave equations with linear and quadratic potentials | Wave Motion 68:202–209 | 10.1016/j.wavemoti.2016.10.003 | feshchenko_popov_2017_exact_tbc_linear_quadratic.pdf (arXiv 1610.08967) |
| tyshchenko2019 | A.G. Tyshchenko, P.S. Petrov, M. Ehrhardt | 2019 | Wide-angle mode parabolic equation with transparent boundary conditions and its applications in shallow water acoustics | Days on Diffraction 2019, 221–225 | 10.1109/dd46733.2019.9016598 | tyshchenko_petrov_ehrhardt_2019_wide_angle_mode_pe_tbc.pdf (preprint) |
| bamberger1988a | A. Bamberger, B. Engquist, L. Halpern, P. Joly | 1988 | Parabolic wave equation approximations in heterogeneous media | SIAM J. Appl. Math. 48:99–128 | 10.1137/0148005 | not downloaded |
| bamberger1988b | A. Bamberger, B. Engquist, L. Halpern, P. Joly | 1988 | Higher order paraxial wave equation approximations in heterogeneous media | SIAM J. Appl. Math. 48:129–154 | 10.1137/0148006 | not downloaded |
| trefethen_halpern1986 | L.N. Trefethen, L. Halpern | 1986 | Well-posedness of one-way wave equations and absorbing boundary conditions | Math. Comp. 47:421–435 | 10.1090/s0025-5718-1986-0856695-2 | not downloaded (AMS 403 from this host; AMS back-issue is free online) |
| milinazzo1997 | F.A. Milinazzo, C.A. Zala, G.H. Brooke | 1997 | Rational square-root approximations for parabolic equation algorithms | J. Acoust. Soc. Am. 101:760–766 | 10.1121/1.418038 | not downloaded |
| yevick_thomson2000 | D. Yevick, D.J. Thomson | 2000 | Complex Padé approximants for wide-angle acoustic propagators | J. Acoust. Soc. Am. 108:2784–2790 | 10.1121/1.1322571 | not downloaded |
| vassallo1996 | C. Vassallo | 1996 | Limitations of the wide-angle beam propagation method in nonuniform systems | J. Opt. Soc. Am. A 13:761–770 | 10.1364/josaa.13.000761 | not downloaded |
| zhang_sarris2015 | X. Zhang, C.D. Sarris | 2015 | Error analysis and comparative study of numerical methods for the parabolic equation applied to tunnel propagation modeling | IEEE Trans. Antennas Propag. 63:3025–3034 | 10.1109/tap.2015.2421974 | not downloaded |
| lytaev2020iccsa | M.S. Lytaev | 2020 | Automated selection of the computational parameters for the higher-order parabolic equation numerical methods | ICCSA 2020, LNCS 12249:296–311 | 10.1007/978-3-030-58799-4_22 | not downloaded |
| lytaev2020nmp | M.S. Lytaev | 2020 | On application of the finite-difference Padé approximation of the pseudo-differential parabolic equation to the tropospheric radio wave propagation problem | Numer. Methods Program. 21:405–419 | 10.26089/NumMet.v21r433 | lytaev_2020_fd_pade_pseudodiff_pe_nmp.pdf |
| lytaev2021iccs | M.S. Lytaev | 2021 | Chebyshev-type rational approximations of the one-way Helmholtz equation for solving a class of wave propagation problems | ICCS 2021, LNCS, 422–435 | 10.1007/978-3-030-77961-0_35 | lytaev_2021_chebyshev_rational_iccs.pdf |
| lytaev2022iccs | M.S. Lytaev | 2022 | Numerical approximation of the one-way Helmholtz equation using the differential evolution method | ICCS 2022, LNCS, 205–218 | 10.1007/978-3-031-08751-6_15 | lytaev_2022_differential_evolution_iccs.pdf |
| lytaev2023iccsa | M.S. Lytaev | 2023 | Computational grid optimization for the 3D higher-order parabolic equation | ICCSA 2023, LNCS 13956:170–185 | 10.1007/978-3-031-36805-9_12 | not downloaded |
| ramamurti2025 | A. Ramamurti, J.F. Lingevitch, J.C. Lighthall, M.D. Collins | 2025 | Stable rational approximations for parabolic equation methods | arXiv:2510.18622 | 10.48550/arXiv.2510.18622 | ramamurti_etal_2025_stable_rational_approx_pe.pdf |

---

# Per-paper notes

Dropped duplicates: a paper that appears in several clusters is described once and cross-referenced elsewhere. Alias keys used by the clusters are normalised to the `refs.bib` keys.

## A. Beilis–Tappert / shift-map terrain PE (radio: TPEM, TEMPER, APM) and early conformal PE

#### beilis1979coupled — Beilis–Tappert terrain-following shift
- **Method.** Ocean rough-surface scattering via normal-mode coupling, starting from the narrow-angle PE. The part later reused for terrain PE is the change of variables $x'=x,\ z'=z-T(x)$ (here $T$ is the boundary profile; in BT the free surface $\eta(r)$), which flattens the boundary, together with a phase "gauge" factor that removes the first-order $\partial_{z'}$ term the shift creates. The paper itself is about coupled-mode master equations (wind-driven rough sea, mode equipartition). The terrain-PE reading of it comes from later papers (Barrios 1993/94, Donohue–Kuttler 1997/2000, Parakkal et al. 2012).
- **Key equations (reconstructed, consistent with donohue1997apl Eq. 4 and the text below it).** Exact Helmholtz in shifted coordinates:
  $$\psi_{x'x'}-2T'\psi_{x'z'}-T''\psi_{z'}+(1+T'^2)\psi_{z'z'}+k^2n^2\psi=0.$$
  With $\psi=e^{ikx}u$, $u=v\,e^{i\theta}$, $\theta=kT'(x)z'+\tfrac{k}{2}\int^x T'^2$, and after dropping $u_{xx}$, $T'u_{xz}$, $T'^2u_{zz}$ and $T''v_{z}$:
  $$2ik\,v_{x}+v_{z'z'}+k^2\big(n^2-1-2z'T''(x)\big)v=0 .$$
  The terrain shows up only through the extra "refractivity" $-2z'T''$, a linear-in-height term proportional to the terrain curvature (like the Earth-flattening term $2z/a$). The Dirichlet BC $v(x,0)=0$ carries over unchanged. Neumann and impedance BCs pick up $T'$-dependent terms.
- **Angle limitations.** Narrow-angle (paraxial) in the sheared frame. It also neglects $O(T'^2)$ metric terms and the cross derivative, so it is small-slope as well. Parakkal et al. (2012) find that above about 20° slope "the limiting factor … is the narrow-angle approximation required by the Beilis–Tappert method".
- **Padé.** The shear $z'=z-T(x)$ is not conformal. The transformed operator has mixed derivatives and an anisotropic metric, $(\partial_{x'}-T'\partial_{z'})^2+\partial_{z'}^2$. The phase trick removes the $\partial_{z'}$ coupling only at narrow-angle order. A Padé expansion of $\sqrt{1+L}$ therefore does not make the BT model wide-angle: the error is in the operator itself, not in how the square root is approximated.
- **Transparent BC.** Above the terrain the map is a rigid vertical shift. The extra potential $-2z'T''$ is linear in $z'$, so NLBCs for linear $m(z)$ profiles (Lytaev 2018 AWPL) could in principle be used. But $T''(x)$ changes the slope of the exterior profile at every range step, and the phase factor $e^{ikT'z'}$ is unbounded in $z'$ (a tilt of the field), so the discrete exterior problem is no longer range-invariant. Nobody has published this combination.

#### dozier1984peruse — PERUSE (piecewise conformal maps)
- **Method.** The surface is taken as piecewise linear and frozen in time. A sequence of conformal maps flattens successive surface segments. "Each conformal mapping preserves the elliptic form of the reduced wave equation", so the SSF narrow-angle PE is applied in each transformed strip and the field is handed over from strip to strip. According to Donohue–Kuttler (1997), Dozier's maps send rectangular strips to strips with straight angled ends and need elliptic integrals, which is expensive.
- **Key equations.** For $w=w(\zeta)$ conformal, $\Delta_{\zeta}\psi+k^2n^2(w(\zeta))\,|w'(\zeta)|^2\psi=0$.
- **Limitations.** SSF narrow-angle. Validated with Bragg peaks from sinusoidal surfaces (small slopes). Interpolation is needed at strip hand-offs.
- **Padé / TBC.** The transformed equation is an exact Helmholtz, so any Padé PE could be used inside each strip. The open problems are the hand-off between strips and the fact that $|w'|\ne 1$ at the top boundary unless the map is the identity there. This is historically the first conformal-map PE (underwater acoustics, rough sea surface).

#### levy1990terrain / levy2000book — FD PE over irregular terrain (Levy)
- **Method.** Crank–Nicolson finite-difference narrow/wide-angle PE. The terrain is a staircase with the ground BC (impedance or Dirichlet) applied at the local ground node. Ch. 7 of the 2000 book reviews terrain methods: staircase, Beilis–Tappert/shift map, and the limits of each. Not read in full (closed).
- **Limitations.** Staircase artefacts at steep slopes. Levy (1997) is the radio reference for the continuous TBC (see cluster E).

#### kuttler1991ssf — Kuttler & Dockery, SSF theory and mixed Fourier transform (TEMPER)
- Background for TEMPER/APM. The mixed Fourier transform (sine/cosine plus an extra term) imposes a Leontovich impedance BC on a flat boundary. Donohue–Kuttler (2000) extend it to the shift-mapped boundary. Narrow- or wide-angle (Feit–Fleck) SSF. Not downloaded.

#### marcus1992hybrid — Marcus, FD + surface Green's function over terrain
- Narrow-angle FD PE with a staircase/terrain-following grid, coupled to a surface Green's function treatment of the upper boundary (an early nonlocal upper BC used together with terrain). Not downloaded. Listed because it is an early example of terrain and a nonlocal top BC in one radio PE model.

#### barrios1994tpem (+ ursi1993barrios, barrios1993tpemreport) — TPEM
- **Method** (from the 1993 URSI abstract and the TAP abstract). Narrow-angle SSF PE (Hardin–Tappert), horizontal polarization (Dirichlet). "The method … is based on 'transforming' the PE through a change of variables and determining a new 'modified' PE with a new range-independent boundary condition… equivalent to defining a new modified refractivity that includes a term based on the second derivative of the terrain." That is, it is exactly the Beilis–Tappert shift: $M\to M-2zT''\cdot10^6/2$, i.e. $m=n^2-1+2z/a-2zT''$, with piecewise-linear terrain. At slope breaks this becomes a delta function in $T''$, which in practice is a phase kink/steering of the field.
- **Limitations.** Narrow angle (about 10–15°) with SSF. Dirichlet only (V-pol discussed but not implemented). Validated against SEKE, GTD, FDPEM and Arizona measurements. Later absorbed into APM (barrios2003apm), which uses a wide-angle SSF with the Donohue–Kuttler shift map and a hybrid ray-optics region.
- **Padé / TBC.** SSF with an absorbing layer (filter) at the top, no transparent BC. The same structural limitation as BT applies.

#### barrios1994roughssf — Barrios & Craig: rough-surface models in SSF (read)
- Compares three rough-sea treatments in SSF: (1) the mixed-transform Miller–Brown roughness reduction factor, (2) a surface-loss-vs-angle function, and (3) Tappert's "approximate solution". In (3) a Taylor expansion of the BC on the displaced surface $z=\eta(r)$ about $z=0$ gives an impedance-like condition that is folded into an effective index $10^{-6}M(z)-\tfrac{i\eta(r)}{2k^2}\,\delta''(z)$ (a smoothed $\delta''$ with a mixing length $L_s$ estimated from the field spectrum). This is the small-roughness (Rayleigh) limit of the BT idea. Not usable for large terrain, but a useful contrast: a perturbative boundary perturbation instead of a mapping.

#### kuttler1993rough — Kuttler & Huffaker
- The narrow-angle SSF PE with the BT shift for a rough (time-frozen) boundary, cited by Donohue–Kuttler as the precursor of their shift map. Not downloaded.

#### donohue1997apl — Donohue & Kuttler, "Modeling radar propagation over terrain" (read in full)
- **Mappings compared** (all chosen so that "the Helmholtz equation on S is still a Helmholtz equation in the coordinates of R; the only modification is to the refractive index"):
  1. **Global conformal map** $w=w(\zeta)$: $n\to n|w'|$. It works "as long as $|w'|$ does not differ too far from unity", i.e. $m=n^2|w'|^2-1\ll1$ for SSF. Demonstrated on a sea-wave train (wavelength 344 m, amplitude 4 m, 25.6 MHz, Bragg lobes). For steps and ramps the conformal maps have such steep gradients that $|w'|$ becomes too large, "and the angle restrictions on the parabolic equation are violated". The authors read this as a failure of the narrow-angle SSF propagator, not of the map.
  2. **Piecewise conformal maps** (Dozier's elliptic-integral maps; DK use $w=\log(c+e^{ib\zeta})$, which maps strips to strips with S-curve ends and has an inverse of the same form). Hand-off between regions by forward map, interpolation and inverse map. Accurate up to about 5.7° (S-ramp) and 6.5° (S-pyramid).
  3. **Continuous shift map** (Beilis–Tappert): $x'=x,\ z'=z-T(x)$ and $m=n^2-1-2zT''$. Described as "mathematically rigorous in that no additional approximations are made beyond those required in the derivation of the PE". Note: this holds only for the narrow-angle PE.
  4. **Piecewise linear shift map (PLSM)**: tracks slope changes discretely at segment boundaries. Recommends 1 terrain sample per 10 range steps. Warns that the $T''$ term needs careful range quadrature.
  5. **Wide-angle PLSM**: acceptable up to about 12.9° (pyramid). Judged "the most robust mapping" and adopted in TEMPER, with knife-edge masking for steeper slopes.
- Shows the exact two-rectangle trick for a vertical step (pad and trim) as a benchmark. Notes that the tilted boundary couples range and height derivatives in the impedance BC; a small-slope impedance approximation was used.
- **Key point for this project.** DK explicitly found conformal maps exact at the Helmholtz level. They rejected them only because the narrow-angle SSF cannot handle large $|n^2|w'|^2-1|$. A wide-angle Padé PE removes exactly that obstacle.

#### donohue2000gtpe — Donohue & Kuttler, TAP 2000 (closed; content from abstract and reproductions)
- **Abstract.** It generalizes the BT shift map to a wide-angle form, derives a new impedance BC for a finitely conducting sloped surface (used with the mixed Fourier transform), and develops a piecewise-linear shift map for sampled terrain. Validity: "slopes as large as 10–15° and discontinuous slope changes on the order of 15–20°". Recommends a hybrid with terrain masking for larger slopes.
- **Wide-angle PLSM equations** (as reproduced in nilsson2021msc Eqs. 141–151 and guan2018dem Eqs. 21–23). With $T'=\tan\beta$ on each linear segment, the shear $\zeta=z-T(x)$ gives
  $$\Big[\big(\partial_x+i\theta_x-T'(\partial_z+i\theta_z)\big)-i\sqrt{(\partial_z+i\theta_z)^2+k^2n^2}\Big]u=0,$$
  with $\theta=k_0T'z+f(x)$. The square root is split as $\sqrt{1+a+b+c}\approx\sqrt{1+a}+\sqrt{1+b}+\tfrac12c-1$ (a Feit–Fleck-type splitting; $c$ is the first-order cross term). Choosing $k_0=k/\sqrt{1+T'^2}=k\cos\beta$ gives
  $$\partial_x u=i\Big(\sqrt{k^2\cos^2\beta+\partial_z^2}\Big)u+ik\Big(\sqrt{n^2-\sin^2\beta}\Big)u\quad(\text{minus the reference phase}).$$
  At a slope break $u_2=u_1\,e^{ikz(\sin\beta_1-\sin\beta_2)}$ (steering by the angle difference).
- **Approximations.** The splitting $\sqrt{1+a+b+c}$ is an operator splitting with $O(ac)$ error. $T''=0$ inside segments, and curvature is concentrated in the phase jumps. Local range contraction $k\cos\beta$. The result is wide-angle in a Feit–Fleck sense (about 30°) but limited to slopes of about 15° and slope jumps of about 20°.
- **Padé.** The square-root symbol $\sqrt{k^2\cos^2\beta+\partial_z^2}$ could be Padé-approximated. But the sheared Laplacian is not reproduced exactly, so the slope error remains whatever the propagator order. Rotation (Collins) is the exact analogue for a straight slope.
- **TBC.** No transparent BC (Hanning absorbing filter). The steering factor $e^{ikz\sin\beta}$ is incompatible with a range-invariant discrete TBC unless it is folded into the exterior solution.

#### barrios2003apm — APM
- Hybrid model: ray optics (flat-earth/extended optics) region + wide-angle SSF PE with the Donohue–Kuttler shift map over terrain, plus knife-edge masking for steep slopes, and a mixed Fourier transform for V-pol. Not downloaded (DTIC down).

#### awadallah2004apl / awadallah2005lateral — TEMPER 3D (read the digest)
- TEMPER uses "the linear-shift map model proposed by Donohue and Kuttler … accurate for terrain slopes less than about 15°", with knife-edge masking for steeper terrain. The 3D vector PE uses a narrow-angle propagator for terrain plus a wide-angle step. Mentions SSF needs a flat boundary in transform space, hence the shift map, whereas FD handles terrain "in a more straightforward fashion" (staircase/terrain nodes). This confirms the about 15° practical limit of the shift map in a production code.

#### parakkal2012beilis — Parakkal, Gilbert & Di, Beilis–Tappert PE for outdoor sound (abstract)
- Beilis–Tappert "treats surface variations in terms of a simple multiplicative factor ('phase screen')". Derives the exact sloping-surface impedance condition and compares it with the flat-ground approximation on hills with 5–22° maximum slopes. Below about 15–20° the flat-ground impedance BC is enough. Above about 20° "the limiting factor … is … the narrow-angle approximation required by the Beilis–Tappert method". This is the clearest published statement of the BT angle limit. The same group's generalized polar coordinate method (parakkal2010polar, cluster C) was proposed to go beyond it.

#### guan2018dem — Guan, Guo, Wang, Li (Xidian), DEM-based PE (read)
- WAPE (Feit–Fleck/Thomson–Chapman type) with SSF, applying the Donohue–Kuttler PLSM (Eqs. 21–23 above) to terrain extracted from a DEM. Comparison against the continuous shift map on two Gaussian hills. Notes that results are "not that satisfactory when the propagation angle is too large". The review paragraph traces Ayasli (masking) → Beilis–Tappert (shift) → Barrios (TPEM) → Janaswamy 1998 (grid interpolation) → Donohue–Kuttler (wide-angle, PLSM). No Padé, no TBC.

#### nilsson2021msc — Nilsson MSc thesis (read Sec. 2.7)
- The most complete open-access reproduction of the Donohue–Kuttler wide-angle PLSM derivation (Eqs. 141–151). It also derives the Earth-flattening map as the conformal map $\xi=ia\log(\zeta/ia)$ (Eqs. 93–97), the standard proof that the modified refractive index $n(1+z/a)$ is $n|w'|$ of a conformal map.

## B. Underwater acoustics: sloping bottoms, rotated PE, mapped depth coordinates

#### collins1990rotated: Rotated PE [abstract/secondary: Moran 2012 §2, Collins & Ramamurti 2020]
- **Method:** rotate the (r,z) frame by the local slope angle φ so that the new range axis $\hat r$ runs parallel to a constant-slope bottom. March a standard (split-step Padé) PE along $\hat r$. In the rotated frame the sloping bottom and its interface conditions are horizontal and exact. The sea surface becomes a sloping pressure-release boundary and is handled by stair steps. A pressure-release boundary is easy to stair-step because its condition is a scalar with no normal derivative.
- **Key equations:** $(\hat r,\hat z)^T = R(\varphi)(r,z)^T$. The Laplacian is rotation invariant, so $\hat p_{\hat r\hat r}+\hat p_{\hat z\hat z}+k^2\hat p=0$ (no extra terms), and $\partial_{\hat r}\hat p = ik_0\sqrt{1+\hat X}\,\hat p$ with $\hat X=k_0^{-2}(\partial_{\hat z}^2+\hat k^2-k_0^2)$.
- **Approximations / limits:** there is no slope-term approximation. Two limits remain. (i) The PE aperture must cover the propagation angles measured from the rotated axis, which are shifted by φ: up-going rays gain φ, so a given Padé order covers roughly φ less of the physical aperture. (ii) The approach applies only to a single constant-slope region. The abstract states that the standard stair-step PE gives large errors for slopes of only a few degrees, and the rotated PE fixes this.
- **Padé:** fully compatible with any order, since the operator is unchanged.
- **Transparent BCs:** in the rotated frame the far boundary (the bottom half-space or an upper exterior) is still a horizontal line in $\hat z$. A rotated environment is range-dependent in $\hat r$ (for example a stratified atmosphere becomes tilted), so an NLBC derived for a $\hat r$-independent exterior is exact only if the exterior is homogeneous.

#### outing2006variable: Variable rotated PE [abstract/secondary]
- **Method:** approximate a variable slope by piecewise-constant slope regions and apply the rotated PE in each one. At each change of slope, march slightly past the junction, then build the starting field for the next rotated frame by **interpolation and extrapolation**. In the elastic case the dependent-variable vector is rotated as well.
- **Limits:** Collins & Ramamurti (2020) report that the method "is not effective for problems involving frequent or continuous changes in slope" but is very accurate for regions of constant slope. It handles Scholte waves.
- **Padé:** any order. **TBC:** the interpolation and extrapolation at the frame changes is ad hoc, and no discrete transparent boundary condition (TBC) has been formulated in this setting.

#### collins1991energy: Energy-conserving PE (ECPE) [abstract/secondary]
- **Method:** keep the stair-step geometry, but at the vertical interfaces between range-independent regions conserve energy flux (following Porter et al. 1991) instead of pressure. The conserved quantity is $\rho^{-1/2}(1+X)^{1/4}p$. Collins & Ramamurti eq. (8) gives the elastic analogue $\rho^{1/2}S(L^{-1}M)^{-1/2}$. It is implemented with rational approximations, and complex Padé coefficients suppress Gibbs oscillations.
- **Limits:** this is a staircase method and conserves energy only. Errors grow with slope, though they are much smaller than with pressure continuity. Its main value for pywaveprop is as the established baseline for the staircase approach the project currently uses.
- **Padé:** any order (rational approximations of $(1+X)^{\pm1/4}$). **TBC:** unaffected, since the exterior is unchanged.

#### porter1991energy: Energy conservation in one-way models [abstract]
- Shows that the standard stair-step PE does not conserve energy because the vertical-interface conditions are treated approximately, and proposes improved interface conditions. This is the motivation for the ECPE and for change-of-variable methods.

#### collins1993splitstep: Split-step Padé [abstract; well known]
- Split-step Padé product $\prod_j (1+a_jX)/(1+b_jX)$ for $\exp(ik_0\Delta r(\sqrt{1+X}-1))$. Range dependence is handled by stair steps (piecewise range-independent regions, field continuity). This is the basis of RAM and the scheme pywaveprop uses.

#### collins2000mapping: Mapping approach, translation $\tilde z = z - d(r)$ [abstract + Moran 2012 §2 restatement]
- **Method:** at each range, rigidly translate the medium vertically so that the sloping bottom interface becomes horizontal. The pressure-release sea surface then becomes a sloping boundary, handled by stair steps.
- **Key equations:** $\tilde r=r,\ \tilde z=z-d(r)$. With constant density the far-field wave equation becomes
  $$\tilde p_{\tilde r\tilde r}-2d'\tilde p_{\tilde r\tilde z}-d''\tilde p_{\tilde z}+(1+d'^2)\tilde p_{\tilde z\tilde z}+k^2\tilde p=0.$$
  The slope terms ($d'$, $d''$) are **neglected**, which gives the standard PE $\partial_{\tilde r}\tilde p=ik_0\sqrt{1+\tilde X}\,\tilde p$. To leading order the neglected terms are restored by a phase correction applied at each change of slope δ: $p\to e^{-ik_0\tilde z\tan\delta}p$. Neglecting the slope terms amounts to physically distorting the waveguide, and the scheme conserves energy.
- **Limits:** small slopes only. Moran (2012) cites a practical 7° limit, and the tests use slopes up to about 4°. Collins & Ramamurti (2020) report the corrected mapping is "fairly accurate", including for high-contrast elastic cases, and that it suits continuous slope variation.
- **Padé:** any order (the operator is the standard one). **TBC:** the translation moves the upper exterior too, so an exterior that is range-independent in z becomes range-dependent in $\tilde z$. A purely translational map cannot keep both the boundary and a stratified exterior range-independent. The translation is, however, harmless for a homogeneous exterior.
- **Relevance:** for z/D(x) maps this is the closest underwater analogue of the Beilis–Tappert / Barrios approach. The phase factor $e^{-ik_0 \tilde z\tan\delta}$ plays the same role as the Beilis–Tappert gauge factor $\exp(-ik_0 z h'(x) - \ldots)$, applied only at slope breaks.

#### moran2012scaled / metzler2014scaled: Scaled mapping, $\tilde z = z/d(r)$ [thesis read in full; Metzler abstract]
- **Method:** scale depth by the relative bottom depth $d(r)$ (bottom at $h\,d(r)$), so that **both** the surface and the bottom are horizontal.
- **Key equations (Moran eq. 3.2):**
  $$\tilde p_{\tilde r\tilde r}-\frac{2zd'}{d^2}\tilde p_{\tilde r\tilde z}-\frac{zd''}{d^2}\tilde p_{\tilde z}+\frac{2zd'^2}{d^3}\tilde p_{\tilde z}+\frac{z^2d'^2}{d^4}\tilde p_{\tilde z\tilde z}+\frac{1}{d^2}\tilde p_{\tilde z\tilde z}+k^2\tilde p=0.$$
  Keeping only the $1/d^2$ term gives $\partial_{\tilde r}\tilde p=ik_0\sqrt{1+\tilde X}\tilde p$ with $\tilde X=k_0^{-2}(d^{-2}\partial_{\tilde z}^2+\tilde k^2-k_0^2)$.
- **Approximations / limits:** all cross and slope terms are dropped, which again amounts to "physically distorting the waveguide". Upslope and downslope tests from 0.23° to 4° against RAM show errors of about 1–5 dB, growing with slope. A phase correction like Collins–Dacol's is said to be needed and was not developed. Moran notes explicitly that it would be "interesting to apply the mapping first to the Helmholtz equation and then introduce a parabolic approximation". In other words, nobody in this line of work carried the full metric terms into the square-root operator.
- **Padé:** any order (the operator has the same form). **TBC:** the scaling acts on the whole depth column, so a top or bottom exterior would see a range-dependent metric. The map must be the identity, or a pure translation, in the exterior.

#### kusel2007single, collins2012single, collins2015sloping, woolfe2016benchmark, collins2020sloping: Single-scattering corrections [Collins & Ramamurti read in full; others abstract]
- **Method:** stair-step geometry. At each vertical interface the transmitted field is computed from an approximate single-scattering solution that conserves the interface conditions "in an average sense". Collins & Ramamurti eqs. (3)–(5):
  $$\begin{pmatrix}u_x\\ w\end{pmatrix}_t=\tfrac i2 (L_B^{-1}M_B)^{1/2}S_B^{-1}\begin{pmatrix}u\\-\sigma_{xz}\end{pmatrix}_i+\tfrac12R_B^{-1}\begin{pmatrix}\sigma_{xx}\\ w\end{pmatrix}_i,$$
  where R and S are depth operators. A fluid–solid variant replaces σ_xz by ∂σ_xz/∂x (eqs. 6–7).
- **Limits:** for elastic sloping fluid–solid interfaces at the benchmark slope of 2.86°, the single-scattering approach breaks down for high contrast and for Scholte waves. ECPE fails for Scholte waves. The rotated PE is best for constant slopes. The mapping approach is "fairly accurate" and applies to continuous slope. Collins & Ramamurti conclude that "it is also possible that a different approach that renders existing approaches obsolete will be discovered", which confirms that no definitive wide-angle sloping-boundary method exists in ocean acoustics.
- **Padé:** any order (rotated rational approximations, Milinazzo et al. 1997). **TBC:** unaffected, since the geometry stays a staircase.

#### abrahamsson1990bc, abrahamsson1990ibvp: Abrahamsson–Kreiss bottom conditions [abstract + ADZ 2001 and ADZ 2009 restatements]
- **Method:** narrow-angle PE in a duct with a rigid range-dependent bottom $z=\ell(r)$. The exact Neumann condition $p_z-\ell'(r)p_r=0$, written for the PE envelope, becomes the "dynamical" condition $\psi_z-\ell'\psi_r-ik_0\ell'\psi=0$. It contains $\psi_r$, so under the change of variable $y=z/\ell(r)$ it turns into a condition containing $u_t$ at the boundary.
- **Key results:** for some downsloping ducts, the IBVP with the dynamical BC shows strong growth (ill-conditioning). Amplitude errors grow with local mode angle and slope. Their replacement, the "paraxialized" AK condition $\psi_z-ik_0\ell'(r)\psi=0$, is well posed for any bottom. After the change of variables $x=y/s(t)$ and the gauge transform $u=e^{-i\delta(t)x^2}w$ with $\delta=\dot s s/2$, it reduces to the simple Neumann condition $u_x(t,1)=0$ (ADZ 2009, eqs. 1.12–1.13).
- **Relevance:** this shows explicitly that a terrain-following map applied **after** the paraxial approximation produces dynamical boundary conditions and well-posedness problems. The Beilis–Tappert-type gauge factor $e^{-i\delta x^2}$ removes the first-order term.
- **Padé / TBC:** narrow-angle only, with no NLBC.

#### abrahamsson1991grid, abrahamsson1994coupled [bibliographic and abstract only]
- Abrahamsson (1991) constructs **orthogonal** curvilinear grids for 2D ducts with varying walls. This is the numerical-orthogonal-map ingredient that later LOT-type methods use. Abrahamsson & Kreiss (1994) solve coupled-mode equations in such ducts. Neither paper was read.

#### akrivis1990variable: Crank–Nicolson on a variable mesh for the Schrödinger PE in a variable domain [read]
- At each range level the depth interval $[0,s(r_n)]$ is divided into J+1 equal cells ($h_n=s(r_n)/(J+1)$), giving a moving uniform mesh, which is equivalent to $y=z/s(r)$. The paper proves ℓ² stability and second-order convergence, with a Dirichlet bottom only. Narrow-angle PE.

#### akrivis2001rigid: FD schemes, variable depth, rigid bottom [read]
- **Key equations:** with $y=x/s(t)$ the PE $u_t=i\alpha u_{xx}+i\beta u$ becomes
  $$w_t=\frac{i}{\xi(t)}w_{yy}+y\mu(t)w_y+i\gamma w,\quad \xi=s^2/\alpha,\ \mu=\dot s/s,$$
  with the dynamical BC $w_y(1,t)=S_1w_t+S_2w$, where $S_1=\dot s s/(1+\dot s^2)$.
- **Results:** H¹ estimates under a sign condition on $S_1(2-\mu S_1)$, optimal-order error bounds for Crank–Nicolson, and the same results for the AK condition.
- **Padé / TBC:** narrow-angle. The surface is pressure-release and the bottom is local, with no NLBC.

#### akrivis1994galerkin, akrivis1994fem, akrivis1996fd: Wide-angle (Claerbout (1,1)-Padé) PE analysis [read; horizontal interfaces]
- Galerkin/FEM and FD discretizations with Crank–Nicolson and implicit Runge–Kutta range stepping, with stability and error estimates, for the third-order WA equation $(1+q\beta)v_r+\alpha^2qv_{zzr}=i\lambda(\alpha^2v_{zz}+\beta v)$. These cover horizontal interfaces only and serve as background for the variable-bottom papers below.

#### dougalis1996fem: FEM for PE with interfaces [abstract]
- Narrow-angle PE with fluid layers. A **change of variable** extends the schemes to interfaces with range-dependent topography, and they are **coupled with a nonlocal ("impedance") bottom boundary condition**. This is the only paper found in this cluster that combines a terrain-following change of variables with a nonlocal BC, and it is narrow-angle and continuous (not a discrete TBC).

#### kampanis1999helmholtz: Helmholtz FEM [abstract]
- Full Helmholtz FEM on an unstructured mesh for range-dependent topography in the interior, with a DtN nonlocal BC at the far-field artificial boundary. It is a reference solver rather than a marching scheme, so it gives no change-of-variables guidance.

#### sturm2007sloping: 3D narrow-angle FE PE, general sloping interface [abstract]
- **Method:** a new change-of-variable technique that does not need the "homotheticity" condition (layers scaled proportionally) used earlier, together with a parabolized interface condition "of the same aperture as the PE". Tested on the 3D wedge, it is better than stair steps.
- **Limits:** narrow-angle. The interface condition must be paraxialized consistently with the PE order, and this is the crux for higher-order Padé.

#### dougalis2009wa, antonopoulou2013downslope: Wide-angle PE in variable-bottom waveguides [2013 read in full]
- **Key equations:** for the Claerbout-type (1,1) rational PE
  $$(1+q\beta)v_r+\alpha^2q\,v_{zzr}=i\lambda(\alpha^2v_{zz}+\beta v)$$
  on $0\le z\le s(r)$, the map $y=z/s(r)$, $u(r,y)=v(r,ys(r))$, gives
  $$\Lambda(r)\Big(u_r-\tfrac{i\lambda}{q}u-\tfrac{\dot s}{s}y\,u_y\Big)=\tfrac{i\lambda s^2}{\alpha^2q^2}u,\qquad \Lambda(r)v=-v''-\tfrac{(1+q\gamma)s^2}{\alpha^2q}v.$$
  The transformed equation is **third order in y** because of the term $\frac{\dot s}{s}yu_y$ inside $\Lambda$. Dirichlet data alone then give an IBVP that may be ill posed for downsloping bottoms, so an extra BC is required: $v_r(r,s(r))=0$, equivalently $v_z=0$ (2013), or $u_{yy}=\frac{2i}{\alpha}s\dot s\,u_y$ (2009, which assumes the narrow-angle PE holds on the bottom). With these, the problem is L²-conservative, $s(r)\|u\|^2$ is conserved, and Crank–Nicolson FD is unconditionally stable and second order. The 2.86° downslope benchmark agrees with a staircase WA-PE code.
- **Approximations / limits:** the mapping is applied to the **already-paraxialized** WA equation. It is exact for that PE, but the aperture is that of the (1,1) Padé PE, not of the Helmholtz equation. Higher-order Padé would add more spatial order, and more extra boundary conditions would be needed.
- **TBC:** a Dirichlet (soft) bottom is used. There is no NLBC.

#### antonopoulou2009dynamical: Galerkin FEM with dynamical BCs [read]
- **Method:** narrow-angle PE with a rigid variable bottom. Map $x=y/s(t)$ and apply the gauge $u=e^{-i\delta(t)x^2}w$ with $\delta=\dot ss/2$, which removes the $u_x$ term. The result is $u_t=ia(t)u_{xx}+i\beta(t,x)u$ with $a=1/(2s^2)$ and $\beta_R=\mathrm{Re}\gamma-\ddot s s x^2/2$, $\beta_I=\mathrm{Im}\gamma+\dot s/(2s)$.
- **Results:** dynamical BC $u_x(t,1)=s_1u_t+mu$. The paper proves optimal L²/H¹ estimates for upsloping bottoms. With the Neumann BC the downsloping ASA wedge is **not convergent**, with violent growth after a bathymetry inflection point. The AK condition gives $u_x=0$, converges for any profile, and agrees with IFD.
- **Relevance:** this is an exact transformed narrow-angle PE on a rectangle. Note the $-\ddot s s x^2/2$ "curvature potential" and the $\dot s/(2s)$ pseudo-attenuation, which are the analogues of the Beilis–Tappert extra terms.

#### lee1983irregular: IFD irregular interface [abstract]
- Implicit FD for the Claerbout WA PE. At an irregular interface it imposes continuity of p and of the normal particle velocity $\rho^{-1}\partial_n p$ on a fixed grid (no coordinate change). The interface is tilted relative to the grid. Narrow or wide angle (1,1) only.

#### evans1983couple: COUPLE [abstract]
- Two-way coupled modes over stepwise depth changes. It is the standard reference solution for the slope benchmarks and is not itself a transformation method.

#### jensen1990benchmark: ASA wedge benchmarks [abstract]
- Upslope and downslope wedges of 2.86°. These are the standard tests used by every paper above.

#### greene1985sloping: High-angle one-way equation along rough and sloping interfaces [abstract]
- Padé-based one-way elastic equation. Rough or sloping interfaces are replaced by an "equivalent reflector" of two thin layers whose properties vary in range. This is a property-smearing method, not a coordinate map.

#### godin1999reciprocity: Reciprocity and energy conservation in PE [bibliographic]
- Gives the conditions under which PE (including WA) solutions are reciprocal and energy-conserving with range-dependent density. Godin's analysis shows that the WA PE over a downsloping boundary need not be energy-conserving, as cited by ADZ 2013.

#### lu1999exact, lu2001lot, lu2004lot: Local orthogonal transform (LOT) plus exact one-way (DtN) reformulation [abstract; details from general knowledge, unverified]
- **Method:** construct an orthogonal curvilinear coordinate system $(\xi,\eta)$ that flattens a curved bottom (2001) or a curved internal interface (2004). In orthogonal coordinates with scale factors $h_1,h_2$, the Helmholtz equation keeps divergence form, $\partial_\xi(\tfrac{h_2}{h_1}\partial_\xi u)+\partial_\eta(\tfrac{h_1}{h_2}\partial_\eta u)+h_1h_2k^2u=0$, with no cross derivatives. The **full Helmholtz** problem is then reformulated as an exact one-way problem through the DtN operator (an operator Riccati equation). Large range steps use local eigenfunction expansions or rational approximations.
- **Limits:** "local" means orthogonality holds only near the boundary or interface (or approximately). The method suits slowly varying waveguides but involves no paraxial or slope truncation, so it is **genuinely wide-angle in principle**.
- **Padé:** Lu's large-range-step methods use rational approximants of the propagator, so the approach is compatible with them. **TBC:** the ducts studied are bounded (pressure-release top, rigid or penetrable bottom), and no NLBC was studied.
- **Relevance:** this is the closest underwater analogue of the conformal-map idea. Conformal maps are the special case $h_1=h_2=|F'|$, where the equation reduces to $\Delta u+|F'|^2k^2u=0$.

#### oba2010conformal: Conformal boundary-flattening transform (rough sea surface) [abstract]
- **Method:** a numerically computed **conformal map** (Bergström's inverse Riemann map via an integral equation) takes the domain under a 1D rough sea surface, many hundreds of wavelengths long, to a flat-topped domain. The Helmholtz equation is preserved with a **transformed sound speed**, $c\to c/|F'|$, i.e. $n^2\to n^2|F'|^2$, so **any existing Helmholtz or PE model** can be applied unchanged. The field is mapped back by interpolation. It was verified with a PE model on rough-surface forward-scatter tests.
- **Limits:** the transform is non-perturbative and the method is exact as a Helmholtz problem. The only approximation is the PE used afterwards, plus neglect of surface motion. A conformal map of a strip also distorts the far field ($|F'|\to$ const at depth, but not exactly 1), which matters for a TBC.
- **Relevance:** this is a **direct precedent** in underwater acoustics for the "conformal map plus modified refractive index plus standard PE" route. It is applied to the surface rather than to bathymetry, and it uses no NLBC.

#### lin2019boundary: 3D boundary-fitted split-step Padé PE [abstract]
- **Method:** a higher-order split-step Padé in 3D with ADI. At each marching step a non-uniform 1D Galerkin mesh is fitted so that the boundary (mainly the pressure-release surface) lies on nodes. A sloping interface is handled by **rotating the grid** to align with it, in the manner of the rotated PE. Validated on a semi-circular waveguide and a tilted wedge.
- **Limits:** the boundary-fitted grid is a mesh adaptation, not a metric-carrying coordinate transform (an inference from the abstract). It works best for pressure-release boundaries, and range dependence is still stepwise.

#### collins2019book: Parabolic Wave Equations with Applications [bibliographic]
- Collins & Siegmann's monograph. Chapter 2 covers range dependence via energy conservation, single scattering, rotated and mapped coordinates. It is the authoritative summary of the NRL line of work.

### Cluster synthesis
1. **Staircase-based methods** (ECPE, single scattering, film or low-shear layers) keep the operator and any Padé order unchanged, and they leave a top NLBC untouched. They fix only the interface conditions, and their accuracy degrades with slope and contrast.
2. **Rotation** (rotated and variable-rotated PE) is exact, because the Laplacian is invariant under rotation. It is Padé-compatible, but it handles piecewise-constant slopes only, needs interpolation and extrapolation at slope breaks, and shifts the aperture by the slope angle.
3. **Terrain-following maps.** These include translation $z-d(r)$ (Collins–Dacol), scaling $z/d(r)$ (Moran and Metzler; Akrivis, Dougalis, Zouraris; Abrahamsson–Kreiss; Antonopoulou et al.) and Sturm's non-homothetic map. Every one of them either **drops the metric (slope) terms** before factoring, which is limited to a few degrees and partly restored by phase corrections, or **applies the map to an already-paraxial (narrow or (1,1)) PE**, which produces dynamical or higher-order boundary conditions and well-posedness issues for downslopes. **None carries the exact metric terms into a higher-order Padé square-root operator.**
4. **Orthogonal and conformal maps** (Lu's LOT plus DtN; Oba's conformal flattening) transform the **Helmholtz** equation exactly, before any one-way approximation. In the conformal case the result is $\Delta u+k^2n^2|F'|^2u=0$. These are the only angle-unrestricted options found. Neither has been combined with a discrete NLBC or with Collins-type arbitrary-order split-step Padé marching on a terrain or bathymetry bottom. Oba used an existing PE but with a standard (non-transparent) bottom treatment. The one combination of a change of variables with a nonlocal BC (Dougalis & Kampanis 1996) is narrow-angle and continuous.
5. **Design implication for pywaveprop:** if the map is the identity (or a rigid translation) above a height $z_*$, and the exterior above $z_*$ is the range-independent region used by the discrete NLBC, then the existing Lytaev NLBC still applies unchanged. Conformal maps of a half-strip do not satisfy this exactly, since $|F'|\ne1$ at finite height, so either (a) the $|F'|^2$ perturbation must decay below the NLBC level, or (b) the NLBC must be derived for $n^2|F'|^2$ with linear-in-height behaviour, as in Lytaev 2018. The underwater literature offers no ready answer here. This is a gap.

## C. Conformal maps, Earth flattening, curvilinear and sigma coordinates

(Also relevant but belonging to other clusters: Dozier 1984 PERUSE used *piecewise* conformal maps of strips with angled ends via elliptic integrals; Collins & Dacol 2000 "A mapping approach for handling sloping interfaces", JASA 107:1937, 10.1121/1.428476.)

### General fact used throughout (conformal invariance of the 2-D Helmholtz operator)

Let $z = x + i y = \Phi(w)$, $w = \xi + i\eta$, be analytic and one-to-one from a computational domain $R$ (rectangle/strip/half-plane, ground at $\eta=0$) onto the physical domain $S$. Then $\Delta_{x,y} = |\Phi'(w)|^{-2}\Delta_{\xi,\eta}$, so

$$\psi_{xx}+\psi_{yy}+k^2n^2\psi=0 \;\Longrightarrow\; \psi_{\xi\xi}+\psi_{\eta\eta}+k^2\,n^2(\Phi(w))\,|\Phi'(w)|^2\,\psi=0 .$$

The equation remains an *exact* Helmholtz equation with effective index $n_{\rm eff}^2=n^2|\Phi'|^2$ (no metric/cross-derivative terms). The divergence form is also invariant: $\nabla\cdot(\rho^{-1}\nabla p)+k^2\rho^{-1}n^2p=0 \to \nabla_w\cdot(\rho^{-1}\nabla_w p)+k^2\rho^{-1}n^2|\Phi'|^2p=0$, so V-pol ($\nabla\cdot\varepsilon^{-1}\nabla H$) and density jumps transform the same way. Boundary conditions: tangential derivatives scale by $|\Phi'|^{-1}$, so a Leontovich/impedance condition $\partial_n\psi+ik\alpha\psi=0$ on the ground becomes $\partial_\eta\psi+ik\alpha|\Phi'(\xi,0)|\psi=0$ (the impedance is multiplied by the local $|\Phi'|$; Dirichlet/Neumann are invariant). None of the cluster-C papers except Parakkal 2010 (polar, curvature term) and Janaswamy (general curvilinear) treat the IBC transformation carefully; Novikov–Akulinichev 2011 state it is unchanged, which is only true where $|\Phi'|=1$.

#### heiblum1975conformal — Curved optical waveguides by conformal transformation
- **Method:** Map $w = R\ln(z/R)$ ($u=R\ln(r/R)$, $v=R\theta$) turns a circular bend of radius $R$ into a straight guide; 2-D scalar wave equation in $w$ has index $n(r)\,e^{u/R}$.
- **Key equations:** $n_{\rm eff}(u)=n\big(Re^{u/R}\big)e^{u/R}\approx n(1+u/R)$; transformed equation $\psi_{uu}+\psi_{vv}+k^2n_{\rm eff}^2\psi=0$ exactly.
- **Approximations/angle limits:** None from the map itself (exact 2-D); they linearize $e^{u/R}$ for analytic/WKB treatment. Basis of all "equivalent straight waveguide" BPM bend models.
- **Padé applicability:** Exact Helmholtz in $w$, so any one-way/Padé BPM applies (used later with Padé BPM, e.g. Hadley 1992, Ilić 1996).
- **TBC:** Outer region acquires an exponentially growing index — the radiation-loss mechanism; a transparent BC there must handle a non-constant ($\propto e^{2u/R}$) index (cf. Lytaev 2018 NLBC for linear $M$-profile, which is the first-order version).

#### pekeris1946earthflattening / koo1960exact — Earth-flattening as a conformal (log) map
- **Method:** Pekeris quantified errors of the approximate flattening $n^2\to n^2(1+2h/a)$. Koo & Katzin (after Pryce, Copson) use $\eta=a\ln(R/a)$, $r=a\theta$ and get an *exact* plane-earth-form equation for the height-gain function.
- **Key equations (Koo–Katzin):** modified index $N^2 = n^2(R/a)^2 = n^2e^{2\eta/a}$; exact height-gain $U=(a/R)^{1/2}V$ with $V''+k^2(N^2-\sigma^2)V=0$, separation constant shifted by $1/(2ka)^2$ (3-D sphere). In 2-D (cylindrical earth) the map $w=a\ln(z/a)$ is exactly conformal and gives $n_{\rm eff}=n\,e^{\eta/a}$ with no extra terms. The usual $M=n(1+h/a)$ is its first-order version.
- **Approximations:** Pekeris: the error grows like $(r/a)^2$ with distance (about 8% at $r/a=1$) and like $f\,h^{5/2}$ with height.
- **Relevance:** The standard modified-refractivity PE (pywaveprop, Lytaev AWPL 2018 NLBC with linear $M$ above the domain) is already a conformal-map PE with a discrete NLBC. This is the only conformal-map + Padé + discrete NLBC combination in the literature, and it uses a *global, range-independent* map.

#### janaswamy1994report — NPS-EC-94-004 (formulation report)
- **Method:** Starts from the exact 2-D equation $\nabla\cdot(a\nabla\psi)+k^2 n^2 a\psi=0$ ($a=1/\varepsilon$ for V-pol). Derives narrow-angle PE, then a wide-angle Padé(1,1) (Claerbout) PE (stated valid to about 20°). Adds a continuous transparent ("tropospheric") BC from the quarter-plane Schrödinger solution, discretized with Fresnel integrals. Then transforms **the narrow-angle PE + IBC + TBC** to a numerically generated *non-orthogonal* body-fitted grid $x=x(\xi),\ y=y(\xi,\eta)$ (linear interpolation between ground and flat top), with Crank–Nicolson.
- **Key equations:** metrics $g_{11}=x_\xi^2+y_\xi^2$, $\sqrt g=x_\xi y_\eta$. The transformed PE is $u_\xi=b_1u+b_2u_\eta+b_3u_{\eta\eta}$ with $b_i$ containing $y_\xi/y_\eta$ etc. The IBC becomes $u_\eta-(\ldots)\sin\theta\cos\theta\,u_\xi+c_1y_\eta\cos\theta\,u=0$ (contains a range derivative because the grid is not orthogonal). Top: $u_\eta+r\,y_\eta u=s\,y_\eta$ (the discretized convolution TBC).
- **Approximations/angle limits:** The transformation is applied *after* the paraxial approximation, so the method stays narrow-angle in the transformed variables (about 10–15°); mixed derivatives and range variation of the metrics are kept to first order only. Grid derivatives are discontinuous at terrain breakpoints.
- **Padé:** Only a (1,1) wide-angle form is derived, and only for a flat ground. Not carried into curvilinear coordinates.
- **TBC:** Notably one of the very few terrain-PE works that combine a curvilinear terrain transform with a transparent (nonlocal, convolution-in-range) upper BC. The top is flat, so the TBC is untouched by the map.

#### janaswamy1998curvilinear — Curvilinear split-step PE over terrain (IEEE TAP 1998)
- **Method (from abstract / secondary sources; full text not obtained):** piecewise-continuous terrain-following coordinate that conforms to the ground at the bottom and becomes flat at a maximum height $H$ (sigma-type, $\eta = H\,(y-T)/(H-T)$-like), plus a phase/amplitude substitution so the equation keeps split-step Fourier (SSF) form. The terrain enters as a refractive-index correction whose *gradient decreases with height*, reducing vertical sampling requirements compared with the Beilis–Tappert map (where the correction $-2zT''$ grows linearly with height).
- **Limitations:** narrow-angle SSPE. Non-orthogonal coordinates, so cross terms are neglected; small-slope (about 10–15°).
- **Padé/TBC:** Not addressed. The flat top in the transformed frame is attractive for a TBC because the map is the identity there.

#### sack1995gtpe — Generalised terrain PE (GT-PE), atmospheric acoustics
- **Method:** Terrain-following transformation (Beilis–Tappert shear $z'=z-h(x)$ family, according to Khodr 2020 and Salomons 2001) applied to the Helmholtz equation, with first and second derivatives of $h$ retained in a wide-angle (Padé(1,1)-type) PE solved by Crank–Nicolson. "Any smooth terrain profile".
- **Limitations:** Salomons (2001) quotes validity for slopes up to about 30°. Accuracy is sensitive to $h',h''$ and the IBC evaluated at the ground (numerical errors accumulate). Requires smooth terrain. Non-conformal, so mixed-derivative terms appear and some are neglected.
- **Descendants:** WAPEtopo (Blanc-Benon/Cotté group, Colas et al. 2023), which uses the Gal-Chen–Somerville sigma map $z=h(\xi)+\eta\,(z_{\max}-h)/z_{\max}$ instead of the pure shear, with the same 30° limit.
- **Padé/TBC:** Only a low-order rational form. The upper boundary is an absorbing layer, not a TBC.

#### Parakkal / Di / Gilbert / You series (Univ. Mississippi, NCPA) — conformal, polar, B–T maps for outdoor sound
- **digilbertyou1992grid, you1993polar:** numerical grid generation and the "polar PE": locally circular terrain, PE in polar coordinates (You's PhD thesis, 1993), validated against hill measurements.
- **digilbert1994turb / digilbert2005mapping:** *cascaded conformal mapping* (sequence of hills of constant curvature, each an earth-flattening-type log map) combined with GFPE and phase screens. The 2005 abstract: a piecewise linear (non-conformal) map does not preserve the Helmholtz form but *does* preserve the narrow-angle PE form, and needs only a phase multiplication per range step.
- **parakkal2006cascaded:** a *continuously* cascaded conformal map for continuously varying curvature. In mapped coordinates the ground is flat and the sound speed is $c/\sqrt J$, $J=|\Phi'|^2$ (exactly the $n^2|\Phi'|^2$ rule). Checked against a non-conformal map.
- **parakkal2009three (abstract):** conformal map and polar transformation are "essentially exact". The Beilis–Tappert map is "in principle applicable only to small slopes and low propagation angles". The mapping cost can equal the propagation cost.
- **parakkal2010polar:** the polar-coordinate method is rederived in differential form and generalized with a *local continuously varying radius of curvature* $R(x)$ (terrain curvature radii much larger than a wavelength). The IBC in the mapped frame becomes $\partial_\eta w = (ik_0Y_g+\chi/2)\,w$ with $\chi$ the local ground curvature (as quoted by Khodr 2020). Agrees with hill measurements.
- **parakkal2012beilis:** B–T PE for porous ground; exact IBC derived. A flat-ground impedance is adequate below about 20° slope. Above that the narrow-angle B–T approximation fails.
- **Padé/TBC:** All narrow-angle (Crank–Nicolson or GFPE). No high-order Padé, no discrete TBC. The conformal variants are the clearest acoustic precedent for "flatten terrain via conformal map then apply a standard flat-ground PE with $c/|\Phi'|$".

#### khodr2020iterative — 3-D B–T PE (read)
- **Method:** 3-D B–T shear $(\xi,\nu,\eta)=(x,y,z-h(x,y))$ applied to the narrow-angle 3-D PE. Transverse Laplacian terms with $h_\nu$ are neglected ($\Delta^*_\perp\approx\partial_\eta^2+\partial_\nu^2$). The 2-D reduction recovers Parakkal's BTPE: $\phi_\xi=\frac{i}{2k_0}(\phi_{\eta\eta}+\delta k^2_{\rm eff}\phi)-ik_0\eta h''\phi$ with $\phi=\psi e^{-i\theta}$, $\theta=k_0\eta h'+\frac{k_0}{2}\int (h')^2d\xi$.
- **Limitations:** about 20° slopes (narrow-angle). The non-orthogonal grid makes the IBC contain $\xi,\nu$ derivatives, and these are dropped (locally flat ground). The paper explicitly calls a wide-angle extension with non-flat IBC "highly desirable". Contains a good literature map (GT-PE, BTPE, Kampanis et al. 2013 FE, Silva et al. 2012 3-D EM B–T, Lin 2019).

#### colas2023wapetopo — WAPEtopo description (read relevant parts)
- **Method:** Sigma map $x=\xi$, $z=h(\xi)+\eta\,[z_{\max}-h(\xi)]/z_{\max}$ (Gal-Chen & Somerville 1975) inside a Sack–West-style wide-angle PE (WAPEtopo). The same map is used for curvilinear LEE.
- **Limitations stated:** slopes up to about 30° (Salomons 2001). Accuracy depends strongly on ground quantities ($h$, derivatives, impedance). Errors accumulate. A hill-induced effect was missed by WAPEtopo compared with LEE.

#### novikov2011conformal — Conformal map for terrain in radio PE (Doklady TUSUR; read)
- **Method:** Builds an *explicit* global conformal map $w=f(z)=z+\sum_{n=1}^N ia_n e^{-iK_nz}$, $K_n=2\pi n/D$, so that $u=x+\sum A_ne^{-K_ny}\sin(K_nx+\varphi_n)$ and $v=y-\sum A_ne^{-K_ny}\cos(K_nx+\varphi_n)$. The curve $v=0$ is the terrain, and $A_n,\varphi_n$ come from the FFT of the (zero-mean) terrain profile.
- **Key equations:** Jacobian $J=(1+U_c)^2+U_s^2\approx1+2U_c$ ($U_c=\sum A_nK_ne^{-K_ny}\cos(\cdot)$). Transformed Helmholtz is $E_{uu}+E_{vv}+k^2\varepsilon/J\,E=0$, and the PE is $2ikU_u+U_{vv}+k^2(\varepsilon/J-1)U=0$ (narrow-angle, CN or SSF).
- **Approximations/limits:** Small slopes $t=\sum A_nK_n\ll1$ are needed both for the first-order inversion $y(x)\approx\sum A_n\cos(K_nx+\varphi_n)$ and for $J\approx1+2U_c$. Practical only for terrain well represented by a few Fourier harmonics. Results agree with piecewise-linear shift map to within 2 dB. The Leontovich BC is (incorrectly in general) left unchanged.
- **Important observation for NLBC:** the index perturbation $1/J-1\sim\sum 2A_nK_ne^{-K_n v}$ decays with height on the scale of the *terrain horizontal wavelength* $D/n$. Long hills perturb the index up to km heights, so a top NLBC assuming range-independent (or linear) index above the domain is violated unless the domain is tall or the map is designed to become the identity at the top (strip map).
- **Padé/TBC:** Not addressed, but the transformed equation is exact Helmholtz, so a split-step Padé of any order applies directly.

#### guo2017pade — Padé-2 PE over irregular terrain (abstract)
- Padé(2) FD PE (pentadiagonal), propagation angles up to about 55°, terrain handled by a "boundary shift" (B–T-type shift map). Secondary sources say the paper derives effective-earth-radius / modified-index formulas from conformal transformations. Terrain is still a shift map (small-slope). DOI only.

#### levy2000book — Levy, Ch. 7 "Irregular terrain modelling"
- Textbook survey: staircase, Beilis–Tappert/shift map, conformal maps, and the finite-difference terrain-following approaches with impedance BCs. Levy also authored the transparent BC for radio PE. Reference text, not downloaded.

#### kirby1988nonorth / kirby1994conformal — Water-wave PE in non-orthogonal and conformal coordinates
- **Method:** Mild-slope equation $\nabla\cdot(CC_g\nabla\phi)+k^2CC_g\phi=0$ (a Helmholtz-type divergence-form equation). Under a conformal map it keeps its form with $k^2\to k^2|\Phi'|^2$. Kirby et al. derive *small- and large-angle* (rational, Kirby-1986 minimax/Padé-type) parabolic approximations directly in conformal coordinates (polar examples: diverging breakwaters, circular channel bend) and compare with full Helmholtz solutions.
- **Relevance:** The closest published precedent for "conformal map + *wide-angle rational* one-way operator" for a Helmholtz-type equation with variable boundaries (outside EM/acoustics). The 1988 paper treats non-orthogonal grids, where cross terms appear and wide-angle forms are harder.
- **TBC:** Lateral boundaries are reflective/absorbing walls. No transparent BC. DOI only (Ifremer copy blocked).

#### Optics BPM line — baets1983curved, rivera1995bent, lu2002bent, shyroki2006/2008, hadley1992pade, ilic1996pade
- **Baets & Lagasse 1983:** arbitrarily curved guides via *local* conformal transformation (local curvature radius) inside BPM. Optics analogue of the cascaded conformal map.
- **Rivera 1995:** FD-BPM for bends in a conformally transformed frame.
- **Lu & Ho 2002:** BPM in *curvilinear* coordinates following an arbitrary bend. Claimed accurate for any curvature (avoids small-curvature linearization) and more flexible than conformal mapping for arbitrary shapes.
- **Shyroki 2006/2008 (read 2006):** exact "equivalent straight waveguide" via the covariant (transformation-optics) form of Maxwell's equations. In cylindrical coordinates (polar, non-conformal) $\varepsilon_{xx}=\varepsilon_{yy}=\varepsilon_C(1+y/R)$, $\varepsilon_{zz}=\varepsilon_C(1+y/R)^{-1}$, so the scalar $n^2(1+y/R)^2$ rule is only approximate for vector fields. In 2-D scalar problems the conformal log map is exact. Bridges to transformation-electromagnetics PE (Ozgun et al., another cluster).
- **Hadley 1992; Ilić et al. 1996:** Padé wide-angle BPM of arbitrary order (multistep factorization identical in structure to Collins' split-step Padé) and its accuracy limits. These are the operators routinely combined with conformally transformed bends in photonics, so conformal map + arbitrary-order Padé is standard practice in optics BPM (with TBC/PML lateral boundaries such as Hadley's TBC), though with the $u$-independent map of a constant-radius bend.

#### desanto1977relation — exact Helmholtz/PE relation
- An exact integral (Fourier–Mellin-type) relation between Helmholtz and PE solutions, obtained with a conformal map in a rectangular waveguide. Theory only, no terrain.

#### galchen1975coord — sigma (terrain-following) coordinate
- Origin of the $\eta=H(z-h)/(H-h)$ coordinate used in WAPEtopo, curvilinear LEE and (probably) Janaswamy 1998. Non-orthogonal and non-conformal. Advantage: identity at the top (flat lid), which suits an upper TBC.

#### Numerical conformal mapping tools — driscoll1996sc, driscoll2002scbook, trefethen2020ncm
- **SC Toolbox:** Schwarz–Christoffel maps for polygonal domains, including strips and half-planes with polygonal boundaries. Piecewise-linear terrain profiles (DEM samples joined by segments) are polygons, so the map of $\{0<z-T(x)<H\}$ onto a rectangle/strip is directly an SC strip map ("stripmap" / "crrectmap"). $|\Phi'|$ has integrable singularities at vertices ($|\Phi'|\sim|w-w_k|^{\alpha_k-1}$), which correspond to diffracting wedge edges (exact geometry, but the singular effective index needs care in Padé discretization, e.g. local refinement or smoothing).
- **Trefethen 2020 (downloaded):** lightning/AAA rational approximations give fast, accurate numerical conformal maps (and their inverses) for domains with corners. Practical for long terrain strips and cheap evaluation of $|\Phi'|$ on a PE grid.
- **No paper found** that uses SC or other *numerical* conformal maps of real terrain/bathymetry in a PE solver. Dozier (1984) and Donohue–Kuttler (1997) used analytic piecewise strip maps, and Novikov–Akulinichev (2011) an explicit Fourier-series map. This is an open gap.

### Cluster synthesis

1. **Exactness.** Conformal maps are the only coordinate change that turns the 2-D Helmholtz equation into an *exact* Helmholtz equation, $n^2\to n^2|\Phi'|^2$, with no cross terms. Range-dependent coefficients are then the only complication, and split-step Padé of any order already handles those (frozen per range step). No angle restriction is introduced by the map itself. Reported "angle limits" (Donohue–Kuttler about 6°, Novikov small slopes) came from narrow-angle SSF/CN with large contrast $|\Phi'|^2-1$, or from approximate analytic map inversion.
2. **Curvilinear/sigma/shear maps** (Beilis–Tappert, GT-PE, WAPEtopo, Janaswamy, polar) are non-conformal. They produce mixed-derivative and metric terms that are dropped or treated paraxially, giving limits of about 20° (BTPE), about 30° (GT-PE/WAPEtopo) and narrow-angle (Janaswamy). The exception is the polar/log map for constant curvature, which is conformal.
3. **TBC compatibility.** The only existing "conformal map + discrete NLBC + Padé" instance is the earth-flattening (log) map (Lytaev 2018). Terrain-induced index perturbations from a half-plane conformal map decay as $e^{-2\pi\eta/L}$ (with $L$ the terrain horizontal scale), so the NLBC top must be high, or the map must be built as a strip map whose top boundary is a straight line with $|\Phi'|\to$ const there. A further issue is that $\xi$ (marching variable) differs from $x$ at the top unless the map is near-identity there. The transformed impedance gets a factor $|\Phi'(\xi,0)|$.
4. **Gap.** No published work combines (numerical) conformal terrain/bathymetry maps, arbitrary-order Padé and a discrete NLBC. The closest precedents are Kirby et al. 1994 (water waves, wide-angle rational PE in conformal coordinates, no TBC), Parakkal 2006/2010 (conformal/polar, narrow-angle), Donohue–Kuttler 1997 (global/piecewise conformal + narrow-angle SSF), and Janaswamy 1994 (curvilinear + continuous TBC, narrow-angle).

## D. Transformation-EM and recent (2005–2026) terrain PE; Lytaev works


#### janaswamy2025rsl — Exact terrain-flattening sigma transform of Maxwell's equations (read in full)
- **Method.** Covariant (tensor) form of Maxwell's equations in 3D under the "σ-transformation" (Phillips 1957, meteorology):
  $x'=x,\ y'=y,\ z'=(z-\zeta(x,y))\,h,\ h=\dfrac{H}{H-\zeta(x,y)}$. It maps the terrain $z=\zeta$ to $z'=0$ and the plane $z=H$ to $z'=H$. Unlike Beilis–Tappert ($z'=z-\zeta$), the terrain's influence decays linearly to zero at $z=H$, and the map is the identity on and above $z=H$. Janaswamy notes that a 2D version was already used in his 1998 TAP curvilinear SSPE.
- **Key equations.** With $g(z')=(H-z')/H$, the Jacobian is $J'=h$. The transformed medium has $\varepsilon=\mu$ (this is exact):
  $$\varepsilon^{m'n'}=\begin{pmatrix}h^{-1}&0&-g\zeta_x\\0&h^{-1}&-g\zeta_y\\-g\zeta_x&-g\zeta_y&h\,[g^2(\zeta_x^2+\zeta_y^2)+1]\end{pmatrix}.$$
  PEC terrain becomes $E_{1'}=E_{2'}=0$ on $z'=0$. The transformed Gauss law is $P_{1'}E_{1'}+P_{2'}E_{2'}+P_{3'}E_{3'}=0$, with $P_{1'}=h^{-1}\partial_{1'}-\zeta_{1'}g\partial_{3'}$, etc. The vector wave equations ((38)–(39) in the paper) are exact. A Higdon ABC is used at the top.
- **Approximations / angle limits.** None in the formulation, which is exact. The paper suggests that PE versions could be derived by dropping higher terrain derivatives and assuming forward propagation, valid "if the slopes … are shallow". No PE is actually derived.
- **Padé applicability.** Not addressed. In 2D, a scalar Helmholtz equation in an anisotropic medium has the form $\nabla'\cdot(A\nabla' u)+k^2 n^2 J^{-1}u=0$, and it contains mixed $\partial_{x'}\partial_{z'}$ terms. A one-way factorization of that operator is not a simple square root of a transverse operator. This is the main obstacle for an angle-unrestricted Padé scheme.
- **Transparent BC.** This is the attractive property: for $z'\ge H$ the map is the identity, so a flat-medium NLBC or TBC at $z'=H$ is unchanged. The transformation reaches the top boundary only through the field values.

#### ozgun2010domain / ozgun2013tmtt / ozgun2013softmeta / kuzuoglu2014perturb — Transformation-electromagnetics (TE) "software metamaterials" (abstracts only)
- **Method.** A coordinate transformation $x\to x'$ is encoded as an anisotropic medium $\varepsilon'=\mu'=\dfrac{\Lambda\Lambda^T}{\det\Lambda}$ inside a fixed FEM/FDTD mesh. This lets them (i) compress the domain or mesh, (ii) remove staircasing in FDTD (Ozgun & Kuzuoglu, IJRFMiCAE 2012, DOI 10.1002/mmce.20642), and (iii) analyse rough or grooved waveguide walls and Monte Carlo rough-surface scattering on a single mesh. Only the material tensor is recomputed for each surface realization. In 2D scalar form this is $\nabla\cdot(\mu'^{-1}\nabla u)+k^2\varepsilon'_{zz}u=0$.
- **PE?** No. I found no Ozgun/Kuzuoglu paper that applies TE to the parabolic equation. All their TE work is full-wave FEM or FDTD (Helmholtz). Their PE work (PETOOL, two-way SSPE, FEM-PE) handles terrain by staircase (PETOOL v1/v2 abstracts state "staircase-approximated terrain") or by knife-edges (2009 recursive 2W-PE). Apaydin & Sevgi 2010 mention "coordinate transformation", i.e. a shift map, for smooth islands.
- **Angle limits.** Not applicable for FEM. PETOOL uses the SSF wide-angle (Feit–Fleck) propagator plus staircase.
- **Padé / TBC.** Not addressed. The TE viewpoint is still useful conceptually: any terrain-flattening map gives an anisotropic $\varepsilon=\mu$ tensor (Janaswamy 2025). The conformal map is the unique case in which the tensor is isotropic, $\varepsilon'=\mu'\propto I$ in 2D. That case gives the Helmholtz equation with $n^2|F'|^2$.

#### ozgun2009recursive, ozgun2011petool, ozgun2020petool2, apaydin2011fempe, apaydin2010map, apaydin2017book — Turkish PE toolboxes (abstracts)
- **Method.** One-way and two-way (recursive forward–backward) SSPE with the Feit–Fleck wide-angle propagator, DMFT impedance ground, and staircase terrain. The 2W scheme reflects at the vertical facets of the staircase. FEM-PE (2011) uses FEM in height with a two-way scheme. The 2010 MAP package uses a coordinate transformation (shift map) for terrain.
- **Angle limits.** Staircase plus SSF. The backward wave is the stated motivation for steep facets.
- **Padé / TBC.** Neither is used: SSF with an absorbing window. The book (2017, ch. 5) is a tutorial summary of staircase vs. shift map.

#### holm2007wasm / holm2009aip — Wide-angle shift-map PE for piecewise-linear terrain, FD (abstract plus Khodr's description)
- **Method.** Shift map $\eta=z-h(x)$, piecewise-linear $h$, combined with the Claerbout (Padé(1,1)) wide-angle FD PE. The "first-order in slope" version reproduces the Donohue–Kuttler SSF wide-angle shift map and gives a tridiagonal system. A "second-order in slope" solution keeps $O(h'^2)$ terms and gives a pentadiagonal system.
- **Key equation** (shift map applied to Helmholtz). Exact:
  $$u_{\xi\xi}-2h'u_{\xi\eta}+(1+h'^2)u_{\eta\eta}-h''u_\eta+k^2n^2u=0 .$$
  The first-order approach applies the phase $u=e^{ik\xi}\,e^{ik h'\eta+\frac{ik}{2}\int h'^2}\psi$ and uses the flat-ground one-way operator.
- **Angle limits (stated).** First order: slopes 10–15°, slope discontinuities 15–20°. Second order: slopes about 15–20°, slope discontinuities 30–40°.
- **Padé.** Only Padé(1,1) (Claerbout). The slope expansion is mixed with the angle expansion, so there is no arbitrary-order generalization.
- **TBC.** Not addressed. Because of the phase factor $e^{ikh'\eta}$, the field in the transformed variables carries a tilt-dependent plane-wave phase up to the top boundary.

#### zhou2015greene / guo2017greene / guo2017pade / guo2018twoway — Long-group FD wide-angle shift-map PEs (abstracts)
- **Method.** Holm's FD shift-map second-order solution is generalized from Claerbout to the Greene approximation (Guo, Zhou, Long, TAP 2017) and to "Padé second-order" (AWPL 2017, $\sqrt{1+Z}$ by a Padé(2,2)-type rational form, pentadiagonal). Terrain is handled by a "boundary shift" (shift map). A two-way Greene FD version appeared in 2018.
- **Stated limits.** Greene: slopes up to about 15°, discontinuities up to about 30°. Phase error below 0.002 gives a flat-ground propagation angle of 36° for Claerbout and 47° for Greene. Padé-2 claims about 55° "from the paraxial direction" on flat ground, but the terrain slope limit still comes from the shift-map expansion.
- **Padé.** Fixed low orders (1 or 2). The slope terms are derived separately for each order, so the method does not generalize to arbitrary order.
- **TBC.** Not used (absorbing layer).

#### wang2020smpe — Dispersion analysis of NA vs. WA shift-map PEs (abstract)
- **Finding.** This is the key negative result. The artificial (numerical) dispersion of the narrow-angle SMPE is independent of terrain, but for the wide-angle SMPE it depends strongly on slope. Wide-angle capability "deteriorates dramatically" as slope grows, and for a "shady" slope above about 5° the WA-SMPE is less accurate than the NA-SMPE.
- **Interpretation.** The shift-map phase trick is exact only for the narrow-angle (Schrödinger) equation, because a Galilean boost commutes with $\partial_\eta^2$. For any wide-angle $\sqrt{1+Z}$, the boost does not commute, so wide-angle shift-map PEs are inconsistent beyond $O(h')$.
- **Padé / TBC.** Shows that "shift map + high-order Padé" does not give an angle-unrestricted terrain PE.

#### zhang2024wasm / zhang2024techrxiv — New FD schemes for WA shift-map PE (abstracts)
- **Method.** TAP 2024 converts $\partial_x$ into $\partial_z$ using a Taylor expansion of the forward wave equation, which turns truncation errors into a $\partial_z$ series and gives a high-accuracy difference scheme. For a 15° terrain it is more than 10 times faster than the Zhou/Guo second-order scheme for propagation angles of ±10°. The TechRxiv paper shows that Holm's/Guo's first-order solutions have a defect for waves near 0° and proposes a new reduced function, allowing a range step of 128λ.
- **Limits.** Still a shift-map, slope-expanded scheme, tested for angles of about ±10°. Not arbitrary-order Padé, no TBC.

#### zhangr2024maxangle — Maximum propagation angle with terrain (abstract)
- The required maximum angle is $\theta_{\max}^{\rm terrain}=\theta_{\max}^{\rm flat}+2\max|\text{terrain slope}|$, because a reflection from a facet of slope $\beta$ turns a grazing ray by $2\beta$. The same geometric argument is in Khodr's thesis: the narrow-angle limit of about 17.5° restricts slopes to about 8.75°. This sets the angle any terrain PE must support, and it is a useful rule for choosing the Padé order and mesh in pywaveprop.

#### wang2019hybrid — Hybrid staircase + "conformal mapping model" 2W-PE (abstract)
- The paper uses the term "conformal mapping model (CMM)" for the one-way shift-map transform (as in Levy's book), not a true conformal map. The staircase handles the backward wave, and CMM gives one-way accuracy on sloping facets. Calibrated against FDTD for LF groundwave. Not Padé, no TBC.

#### silva2012_3dpe — 3D PE with a Beilis–Tappert-like transform for radio (DOI only)
- A narrow-angle 3D PE in a terrain-following $z-h(x,y)$ frame, solved with a direct sparse solver. Cited by Khodr as a precursor. Narrow-angle only.

#### eibert2002 — Linearly bridged knife-edges, spectral-domain terrain BC (abstract)
- Not a coordinate transform. Terrain is a sequence of linearly bridged knife-edges with a spectral-domain boundary condition. Claims very wide angular spectrum "as long as the dominant portion of energy propagates above the terrain". SSF wide-angle.

#### kuttler2002 — Improved DMFT (abstract)
- A robust and efficient implementation of the mixed Fourier transform for impedance boundaries in SSF, used together with the shift map in later DK-type terrain codes. Relevant only for impedance BCs with SSF, not for terrain mapping.

#### bonnafont2021lssw / bonnafont2020thesis / bonnafont2022twoway — Split-step wavelet (read relevant sections)
- **Method.** Wavelet-domain propagation with local, translation-invariant propagators, DMFT or local image method for the ground. Relief is a **staircase model** ("The relief is modeled with a staircase model … the field is translated for increasing relief … set to 0 inside the relief"; operator $L$). The thesis (§1.2.2.f) cites Janaswamy's curvilinear transform as a possible improvement, but it is not implemented. The two-way SSW (2022) adds backscatter from staircase facets.
- **Padé / TBC.** SSF-type propagator (wide-angle Feit–Fleck in the wavelet basis) with an apodization window. Neither is used.

#### khodr2020iterative / khodr2020thesis / khodr2022gji — 3D and wide-angle Beilis–Tappert PE (read)
- **3D (JASA 2020).** Map $\xi=x,\ \nu=y,\ \eta=z-h(x,y)$. The transformed transverse Laplacian is
  $\Delta^*_\perp=\partial_\eta^2+h_\nu^2\partial_\eta^2+\partial_\nu^2-h_{\nu\nu}\partial_\eta-2h_\nu\partial_{\eta\nu}$.
  All $h$-derivative terms in it are dropped ("consistent with the narrow-angle approximation … limits accuracy to sloping angles of 20°"). This gives $\partial_\xi w=\frac{ik_0}{2}(Z^*+Y^*)w$ with $Z^*=k_0^{-2}\partial_\eta^2+\delta k^2/k_0^2-\frac{2i}{k_0}(h_\xi+M_yh_\nu-M_z)\partial_\eta$. The ground uses an approximate (parabolized, after Abrahamsson–Kreiss) impedance BC, the top an absorbing layer, and the solver is iterative tridiagonal.
- **Wide-angle 2D (thesis §3.2.3, eqs. 3.54–3.59).** Applying $p=\psi e^{iE}$ gives $\{(\partial_\xi+iE_\xi)-h'(\partial_\eta+iE_\eta)-ik_0(-1+\sqrt{1+Z+Q})\}\psi=0$. The square root is then split as $\sqrt{1+Z+Q}\approx\sqrt{1+Z}+Q/2$ ("valid for boundaries with a sloping angle of about 20°"). The choice $E=k_0h'\eta+C(\xi)$ gives
  $$\partial_\xi\psi=ik_0(-1+\sqrt{1+Z})\psi-ik_0\eta h''\psi,$$
  i.e. the flat-ground wide-angle PE plus a linear-in-$\eta$ potential. $\sqrt{1+Z}$ is then approximated by higher-order Padé with Crank–Nicolson FD. This reproduces Donohue–Kuttler/Holm, and for piecewise-linear $h$ the extra term vanishes. Stated practical validity is for slopes within about 30°.
- **Padé.** Arbitrary order formally, but the terrain coupling is only first order in $h'$. The wide-angle gain is real only for the free-space part. This matches Wang 2020.
- **TBC.** Absorbing layer only. The 3D model is also restricted to small slopes.

#### nyborg2023 — BTPE vs GTPE comparison for wind-turbine noise (abstract)
- Compares the Beilis–Tappert PE and the Sack–West Generalized Terrain PE against two measurement campaigns. Both are terrain-following transforms with slope-limited accuracy. Useful for benchmarks.

#### kampanis2002 / kampanis2005 / kampanis2013 — FEM SPE in boundary-fitting coordinates with a nonlocal top BC (abstracts)
- **Method.** Standard (narrow-angle) PE in a refracting atmosphere over mild topography. A curvilinear or "generalized boundary-fitting" coordinate system maps the domain onto a rectangle. P1 FEM in height and conservative Crank–Nicolson in range.
- **TBC.** **This is the closest existing combination of a terrain transform with an NLBC.** In 2002 there is an NLBC for a homogeneous upper half-space, and "the underlying transformation of coordinates allows the effective coupling with the nonlocal boundary condition". In 2005 it is the Dawson–Brooke–Thomson NLBC for a linear $n^2$ upper layer, and "a discrete form of the nonlocal boundary condition, which is left unaffected by the transformation of coordinates, is employed". The paper evidently uses a map that is the identity near the artificial top, like Janaswamy's σ-map.
- **Angle limits / Padé.** Narrow-angle (SPE) only. No Padé, no wide-angle. Full texts were not obtained; the exact map should be checked in kampanis2013 (Appl. Numer. Math.).

#### lin2019boundary — 3D boundary-fitted split-step Padé PE (abstract)
- Higher-order split-step Padé with ADI and a non-uniform Galerkin discretization on a boundary-fitted grid. It is not a coordinate transform of the equation: the mesh nodes follow the boundary. Sloping interfaces are handled by rotating the grid (as in the rotated PE). Pressure-release surfaces only.

#### swearingen2017 — Wide-angle topography PE without coordinate transform (abstract)
- Represents topography as range-dependent density and wavenumber jumps (immersed-ground approach) in a wide-angle FD PE. The authors argue that transform methods need interpolation or extrapolation near the top boundary, which causes errors at long range. This is the same top-boundary issue the σ-type maps avoid.

#### lytaev2018rtuwo — SSP over irregular terrain (abstract)
- Arbitrary-order SSP with NLBC on both boundaries, ground and vegetation as dielectric media, and terrain of arbitrary profile. Judging by the abstract and the current pywaveprop design, the terrain is staircase or immersed. This is the starting point the new work wants to improve.

#### lytaev2020awpl / lytaev2021iccsa / lytaev2022jocs / lytaev2023jocs / lytaev2023jmse / lytaev2026nmp — Numerov-Padé, rational interpolation, mesh optimization
- **Method.** 4th-order Numerov transverse discretization with arbitrary-order Padé or rational interpolation of $\exp(ik\Delta x(\sqrt{1+L}-1))$. The discrete dispersion relation is used to optimize $\Delta x$, $\Delta z$ and order. The 2026 paper extends mesh optimization to inhomogeneous media: the admissible grid depends on the range of the wavenumber $k^2(z)$. Ignoring that range leads to grids that are too coarse.
- **Relevance.** A conformal map gives the Helmholtz equation with $k^2n^2|F'(w)|^2$. The mesh-selection theory of lytaev2026nmp applies directly if the range of $n^2|F'|^2$ over the domain is used in place of the range of $n^2$. Grazing angles in the mapped frame are bounded by the physical angle plus the local boundary slope (see zhangr2024maxangle).
- **TBC.** Discrete NLBC (2018) is compatible if $|F'|\to$ const near the top (see the gap note below).

### Cluster synthesis
1. **Shift map / Beilis–Tappert family** (Holm, Long group, Wang 2020, Zhang 2024, Khodr, Parakkal, Guan, Gao). The transform is exact only for the narrow-angle PE. All wide-angle variants split $\sqrt{1+Z+Q}\approx\sqrt{1+Z}+Q/2$ or expand in slope, so they are valid for slopes of about 15–20° (discontinuities up to 30–40°). Wang 2020 shows that the WA-SMPE degrades below the NA-SMPE for slopes above about 5°. Padé is used only at orders 1–2 and is not angle-unrestricted.
2. **TE / anisotropic exact transforms** (Ozgun–Kuzuoglu FEM, Janaswamy 2025). These are exact, but they give an anisotropic $\varepsilon=\mu$ tensor with mixed derivatives. Nobody has derived a one-way or Padé PE from them. Ozgun's PE tools use staircase terrain.
3. **NLBC + terrain transform.** Only Kampanis (2002, 2005, 2013), with narrow-angle FEM, has coupled a terrain-fitting transform with a nonlocal top BC. He relies on the transform not affecting the top boundary. Janaswamy's σ-map has the same property.
4. **SSW (Bonnafont)** and the PETOOL toolboxes use staircase terrain.
5. **Gap.** I found no paper (radio or acoustics, 2005–2026) that combines (a) a Helmholtz-preserving (conformal) map, (b) arbitrary-order Padé or rational one-way propagation, and (c) a discrete NLBC. The closest pieces are Kampanis (transform + NLBC, narrow-angle), Khodr (BT + higher-order Padé, slope-limited) and Lytaev (arbitrary Padé + discrete NLBC, staircase).

## E. Transparent / nonlocal boundary conditions (topic 5) and discrete dispersion analysis (topic 6)

### Common thread (read this first)

All *fully discrete* TBCs relevant to pywaveprop (Arnold–Ehrhardt 1998; Ehrhardt–Zisowsky 2007; Mikhin 2004/2008; Lytaev 2018/2020/2022) are built the same way: write the exact marching scheme in the exterior strip $j\ge J$, apply the $\mathcal Z$-transform in the **range index** $n$, solve the resulting $p\times p$ system of second-order difference equations in $j$ in closed form (geometric sequence $\mu^j$ for a constant exterior, Bessel functions $J_{\nu+j}$ = discrete Airy for a linear exterior profile, continued fraction for Numerov+linear), keep the decaying branch, and invert $\mathcal Z$ numerically. This imposes hard requirements that any change of variables must respect:

1. **Range-invariant exterior scheme**: the difference equations for $j\ge J$ must have coefficients independent of $n$. This means: the same Padé coefficients $a_l,b_l$ at every step, the same $\Delta x$ (uniform range step in the *computational* variable), the same $\Delta z$, and an exterior potential $V(z)$ that does not depend on $x$.
2. **Exterior profile of special form**: constant (all papers) or linear in $z$ (Levy 1997 continuous; Ehrhardt–Mickens 2004 for CN/Schrödinger; Lytaev 2018 for 2nd-order Padé; Lytaev 2020 for Numerov–Padé; Feshchenko–Popov 2017 continuous, also quadratic). Constant density in the exterior.
3. **Starting field (and sources) supported inside** the computational domain (relaxations: Mikhin 2004, Levy 1997 incoming-energy term, Ehrhardt 2005 preprint "non-vanishing starting fields").
4. The convolution coefficients depend only on $(\Delta x,\Delta z,a_l,b_l,\text{exterior }V)$, so they can be precomputed and tabulated (Lytaev 2018).

**Consequence for a terrain-following / conformal map** $(x,z)\mapsto(\xi,\eta)$: the transformed equation must coincide, for $\eta\ge\eta_J$, with a range-independent PE whose effective $m^2$ is constant or linear in $\eta$. 
- A pure Beilis–Tappert shift $\eta=z-h(x)$ applied everywhere violates (1): the effective potential acquires $h'(x)$, $h''(x)$ terms (and a phase factor $e^{ikh'z}$ in the narrow-angle version) all the way to infinity. 
- A **blended map** (identity, or a constant shift, above some height $z_b>\max h$; e.g. $\eta=(z-h(x))\,H/(H-h(x))$ on $[h,H]$ and $\eta=z$ above $H$) keeps the exterior untouched, which is what Kampanis (2002) and Kampanis et al. (2013) do ("the underlying transformation of coordinates allows the effective coupling with the nonlocal boundary condition"; the discrete NLBC "is left unaffected by the transformation"). Such maps are *not* conformal and introduce mixed-derivative/first-derivative terms in the interior.
- A **conformal map** cannot be the identity on an open set unless it is the identity everywhere (identity theorem for analytic functions). So with an exactly conformal map the exterior index $m^2|F'|^2$ is only asymptotically range-independent ($|F'|\to1$, typically like $O(1/r)$ for a localized hill, or exponentially in $\eta$ for a periodic/strip-type map $F(w)=w+\sum c_k e^{ikw}$, where the $k$-th Fourier mode of the boundary perturbation decays like $e^{-k\eta}$). Then the DTBC is exact only up to the residual $|F'|^2-1$ at the top; this residual must be pushed below the target accuracy by choosing $z_b$ (or by using a strip map where the decay is exponential). None of the papers found analyses this error.
- If the exterior perturbation is a **range-dependent constant** $V(x)$ (e.g. a constant vertical shift with changing reference), the Schrödinger case can be fixed by a gauge change $\psi=e^{-i\int V}u$ (Antoine et al. 2008, Remark 5), but for $\sqrt{1+L}$ and Padé products this does not factor out (the propagator is nonlinear in $V$), so it does not carry over to wide-angle Padé.
- A **range-dependent linear exterior slope** (bent waveguide of variable curvature, i.e. exactly what Heiblum–Harris-type conformal bend maps produce) has an exact continuous TBC only for the narrow-angle Leontovich PE (Feshchenko–Popov 2017); no discrete/Padé version exists in the literature found.

No paper was found that combines (i) a conformal or terrain-following transform, (ii) arbitrary-order Padé/rational propagator, and (iii) an exact discrete NLBC. Closest: Kampanis 2002/2013 (terrain-fitting + NLBC/DtN, but standard narrow-angle PE, FEM, CN) and Lytaev 2018/2020 (arbitrary Padé + discrete NLBC, linear exterior, but staircase terrain).

#### arnold1998vlsi — Numerically absorbing BCs for quantum evolution equations
- **Method**: discrete TBC for the 1D Schrödinger equation discretized by Crank–Nicolson, derived from the discrete exterior problem by $\mathcal Z$-transform; discrete convolution in time.
- **Key equations**: $\psi^{n}_{J}-\psi^{n}_{J-1}=\sum_{k=0}^{n}\ell_{n-k}\psi^{k}_{J}$ with $\ell_n=\mathcal Z^{-1}\{\hat\ell(z)\}$, $\hat\ell$ = decaying root of the characteristic equation of the CN exterior recurrence.
- **Assumptions/limitations**: potential constant outside the domain; initial data compactly supported inside; uniform time step. Proven unconditionally stable; shown superior to ad-hoc discretizations of the continuous TBC (which are only conditionally stable and reflect).
- **Padé applicability**: (1,0)/(0,1) level only; it is the template for all later DTBCs.
- **Transformed coordinates**: requires constant exterior potential, i.e. untransformed exterior.

#### arnold_ehrhardt1998jcp — DTBCs for wide-angle PE (underwater acoustics)
- **Method** (from abstract-level knowledge and Ehrhardt–Zisowsky's summary): DTBC for Claerbout's (1,1) WAPE $\psi_r=ik_0\big(\frac{p_0-p_1L}{1-q_1L}-1\big)\psi$ discretized by CN in range and 2nd-order FD in depth, with careful discretization of the water–bottom density jump.
- **Assumptions**: homogeneous fluid bottom half-space (constant $c_b,\rho_b,\alpha_b$) below the artificial boundary; uniform range step; source/initial field inside. Scheme proven unconditionally stable. Irregular bottoms are handled by placing the DTBC *below the deepest point* (the bottom variation stays inside the computational domain).
- **Padé**: (1,1) only; generalized to arbitrary order by Ehrhardt–Zisowsky 2007.
- **Transformed coordinates**: not considered; exterior must be range-independent.

#### ehrhardt1999vlsi — DTBCs for general Schrödinger-type equations
- **Method**: extends DTBC construction to general Schrödinger-type equations (e.g. pseudo-differential/wide-angle) with $\mathcal Z$-transform.
- **Assumptions**: constant exterior coefficients. Not downloaded (Hindawi blocked).

#### arnold_ehrhardt_sofronov2003 — fast calculation, approximation, stability (downloaded)
- **Method**: DTBC for CN Schrödinger; the convolution kernel is approximated by a sum of exponentials $\ell_n\approx\sum_{k=1}^{L}b_kq_k^{-n}$, giving a recursive $O(L)$-per-step update instead of $O(n)$.
- **Key result**: stability proof for the approximated DTBC; error estimates.
- **Relevance**: the sum-of-exponentials trick is directly applicable to pywaveprop's NLBC convolution (cost reduction for long ranges); it does not relax the range-invariance requirement.

#### ehrhardt_mickens2004 — discrete Airy equation (downloaded preprint)
- **Method**: exact decaying solution of the discrete Airy recurrence $y_{m+1}-2y_m+y_{m-1}-(d+cm)y_m=0$ is $y_m=J_{m+(2+d)/c}(2/c)$ (Bessel function of the order index). Used to build a DTBC for CN Schrödinger with a **linear** exterior potential (earth-flattened troposphere, linear sound-speed-squared gradient).
- **Key equations**: decaying solution $y_m=J_{m+\frac{2+d}{c}}\!\left(\tfrac{2}{c}\right)$; ratio $y_{J-1}/y_J$ computed by continued fractions (modified Lentz).
- **Assumptions**: linear exterior potential, range-independent; CN (narrow-angle) only; "construction of DTBC to wide-angle PEs will be future work".
- **Padé**: extended to arbitrary-order Padé by Lytaev 2018 (2nd order in $z$) and Lytaev 2020 (Numerov).
- **Transformed coordinates**: if a map leaves a *range-independent linear* effective profile above $z_b$ (e.g. Earth flattening, or a terrain map that becomes a pure constant shift), this DTBC applies unchanged.

#### zisowsky_ehrhardt2006 — DTBCs for parabolic systems (downloaded preprint)
- **Method**: DTBCs for general systems of parabolic equations ($\theta$-method + FD), $\mathcal Z$-transform; error analysis of the numerical inverse $\mathcal Z$-transform; sum-of-exponentials approximation.
- **Relevance**: theoretical basis for the $p\times p$ system that the Padé product generates in the exterior (each Padé factor adds one auxiliary unknown).

#### ehrhardt_zisowsky2006pamm / ehrhardt_zisowsky2007 — discrete NLBC for split-step Padé (downloaded preprint, read in full)
- **Method**: two schemes: (a) CN in range + $(\ell,m)$ Padé of $\sqrt{1-L}$, factorized into $\psi^{n+1}=\frac{c_U}{c_W}\prod_{l=1}^p\frac{1-a_{l}L}{1-b_{l}L}\psi^n$; (b) Collins' split-step Padé, additive form $\psi^{n+1}=\psi^n+\sum_{l=1}^p\frac{a_lL_h}{1+b_lL_h}\psi^n$. Introduce $p-1$ intermediate fields, $\mathcal Z$-transform the exterior system, write it as a first-order $2p$ system $(\hat\psi_{j+1},\hat\eta_{j+1})^T=(A^{-1}B+I)(\hat\psi_j,\hat\eta_j)^T$, split decaying/growing Jordan blocks, obtain DtN form $\nabla^-\hat\psi_J=\hat D(z)\hat\psi_J$, invert with FFT on a circle of radius $\varrho>1$:
  $$\psi^{n+1}_J-\psi^{n+1}_{J-1}-D_0\psi^{n+1}_J=\sum_{l=1}^{n}D_{n+1-l}\psi^l_J,\qquad D_n=\frac{\varrho^n}{2\pi}\int_0^{2\pi}\hat D(\varrho e^{i\varphi})e^{in\varphi}d\varphi.$$
- **Depth discretization**: standard 2nd order, plus Collins' "Padé in $L_h$" trick $L=\theta(L_h)=-\frac{4}{\sigma^2}\log^2\!\big(\ldots\big)$, i.e. the exact relation between the continuous and discrete Laplacian, $L_h=\frac{2}{\sigma^2}(\cosh(\sigma L^{1/2})-1)$, $\sigma=hk_0$; coefficients fitted with fsolve. This is the same idea as Lytaev 2020 NMP ("Padé simultaneously in $x$ and $z$"). Authors note it is "unclear how to generalize this depth operator to the case of a non-zero potential term".
- **Assumptions/limitations**: homogeneous exterior (constant $V_b$, constant density); uniform range step; initial data inside; no stability proof (numerically stable up to (8,8)). Numerical examples only for horizontal bottoms; "irregular bottom surfaces ... can be included by simply extending the range of z" (i.e. TBC below the deepest point, staircase inside).
- **Padé**: arbitrary order, both CN-Padé and split-step Padé. Error from DTBC ~$10^{-12}$–$10^{-14}$ (round-off level), no "numerical plateaus" (unlike semi-discrete TBC of Schmidt et al.).
- **Transformed coordinates**: not considered. Works for any interior (any $V(x,z)$, any transform) as long as the exterior strip is range-invariant and homogeneous.

#### antoine2008review — review of TBC/ABC for Schrödinger (downloaded ZIB preprint)
- **Method**: comprehensive review: continuous TBCs (Baskakov–Popov–Papadakis), DTBCs (Arnold, Ehrhardt, Schmidt–Deuflhard), semi-discrete, Engquist–Majda-type ABCs, PML, 2D (straight, circular, general convex boundaries), nonlinear case, fast convolution (sum-of-exponentials).
- **Key facts used here**:
  - Remark 5: exterior potential depending only on time ($\equiv$ range) is removed by the gauge change $\psi=e^{i\mathcal V(t)}u$, $\mathcal V(t)=\int_0^tV(s)ds$; TBC $\partial_n u+e^{-i(\pi/4-\mathcal V)}\partial_t^{1/2}(e^{-i\mathcal V}u)=0$.
  - Remark 6: space- **and** time-dependent exterior potentials have no exact TBC; only approximate (pseudo-differential, Antoine–Besse–Klein).
  - §4.2.3: for a general smooth convex boundary the domain is locally mapped onto a half-space; the transformed operator has variable coefficients, Fourier analysis fails, and only asymptotic (approximate, local-in-space) ABCs via symbolic calculus are available.
- **Relevance**: makes explicit that exact TBCs require the exterior to be invariant along the evolution variable; any terrain map must leave the exterior untouched.

#### schulte_arnold2008 — DTBC for compact higher-order (9-point) scheme
- **Method**: 2D Schrödinger, compact 9-point (4th-order, Numerov-type) in space + CN in time; DTBCs derived for this scheme; perfect absorption independent of angle.
- **Relevance**: discrete TBC theory is compatible with Numerov-type transverse discretization (same as pywaveprop's). Not downloaded.

#### arnold2012circular — DTBC on circular domains (downloaded)
- **Method**: polar coordinates, Fourier in angle, DTBC per angular mode for conservative CN; sum-of-exponentials approximation; unconditional stability.
- **Assumptions**: potential constant outside $r\ge R$, initial data inside.
- **Relevance**: example of an exact DTBC in curvilinear (but *separable*) coordinates — works because the exterior operator stays separable and evolution-invariant. A terrain map is not separable in the exterior unless it reduces to identity there.

#### zlotnik_romanova2015 — Numerov–CN–Strang with DTBC (downloaded, arXiv)
- **Method**: 2D Schrödinger on semi-infinite strip; Numerov in space, CN in time, Strang splitting w.r.t. potential; DTBCs; uniqueness and uniform-in-time $L^2$ stability (conservation) proven.
- **Relevance**: rigorous stability theory for Numerov + DTBC (pywaveprop uses Numerov + Padé + DTBC); Strang splitting w.r.t. potential is one way to handle the extra variable coefficients that a transform generates while keeping the exterior DTBC exact.

#### antoine_besse_mouysset2004, antoine_besse_klein2013 — ABCs with curved boundaries / exterior potentials
- **Method**: pseudo-differential (symbolic) ABCs for 2D Schrödinger; 2013 paper: exterior potentials that vary in space (and time).
- **Relevance**: these are the only systematic treatments of variable exterior coefficients, but they are *approximate* (asymptotic order) and narrow-angle (Schrödinger). Could be a fallback if a conformal map leaves a weak residual variation in the exterior.

#### baskakov_popov1991, marcus1991, papadakis1992, papadakis1994, popov1996 — continuous NLBC (BPP condition)
- **Method**: continuous exact TBC for the narrow-angle PE $2iku_x+u_{zz}=0$ above $z_b$ (Baskakov–Popov–Papadakis, BPP), in the form given by Feshchenko–Popov (eq. 27): $\partial_z u(x,z_b)=-\sqrt{\frac{2k}{\pi i}}\,\partial_x\int_0^x\frac{u(\xi,z_b)}{\sqrt{x-\xi}}d\xi$ (constant exterior; a constant exterior potential is absorbed by a phase factor). Papadakis 1994 also gives exact NLBCs for (1,0) and (1,1) Padé; for sloping bottoms the boundary must be placed along "an artificially placed horizontal computational boundary inside the bottom" (abstract). Popov 1996 extends to quasi-stratified exteriors and "modified PEs" (earth-flattened).
- **Limitations**: continuous kernel; ad-hoc discretizations are only conditionally stable (reason for DTBCs). Horizontal boundary required.
- **Transformed coordinates**: Papadakis explicitly avoids sloping NLBC boundaries; the boundary is horizontal and the medium below homogeneous.

#### levy1997 — TBC for radiowave PE
- **Method** (abstract): nonlocal TBC as convolution over previous ranges, "kernel matched to the refractive index vertical gradient at the boundary" (constant or linear $m^2$ above $z_b$, Airy-type); includes an incoming-energy term (plane wave, sources above boundary); extension above boundary via horizontal PE. Narrow-angle (standard PE, FD/CN).
- **Limitations**: range-independent exterior gradient; narrow-angle; continuous kernel discretized.
- **Transformed coordinates**: Levy's book (levy2000book) handles terrain by staircase/shift-map; TBC requires the top boundary in a range-invariant exterior.

#### yevick_thomson1999 — NLBC for FD PE solvers
- **Method**: NLBCs obtained directly from z-space CN solvers for Tappert and Claerbout PEs by expanding the downgoing vertical wavenumber operator in powers of the range-translation operator $R=e^{-\Delta r\partial_r}$; i.e. a discrete-in-range construction (close in spirit to $\mathcal Z$-transform).
- **Limitations**: low-order (Tappert, Claerbout) PEs; homogeneous exterior.

#### brooke_thomson2000 — NLBC for high-order PE algorithms
- **Method**: implementation of Papadakis-type exact NLBC (formally unlimited wide-angle, for $\sqrt{1-L}$) within high-order Padé PE algorithms (PECan lineage).
- **Limitations**: reported computational instabilities (cited by Ehrhardt–Zisowsky as motivation for fully discrete TBCs). Not downloaded.

#### schmidt_deuflhard1995, schmidt_friese_yevick2001 — DTBC for Fresnel / semi-discrete TBC for split-step Padé
- **Method**: 1995: DTBC for the Fresnel (narrow-angle) equation. 2001: TBC for split-step Padé via Laplace transform in the *transverse* variable (semi-discrete, continuous in z).
- **Limitations**: semi-discrete approach shows "unphysical numerical plateaus" depending on $\Delta z$ (Ehrhardt–Zisowsky 2007 comparison).

#### zebic2001 — wide-angle NLBC for split-step Padé (radio)
- **Method** (abstract): NLBC for split-step Padé PE in tropospheric propagation; wide angles "without loss of accuracy", plane waves at very large angles handled.
- **Limitations**: constant exterior presumably; not downloaded.

#### levy2001pml — PML for PE
- **Method**: PE solution analytically continued to complex height; marching on a complex contour = Bérenger PML; both narrow- and wide-angle PE; efficient with split-step Fourier.
- **Relevance to transforms**: PML is itself a complex coordinate stretching $z\mapsto z+i\sigma(z)$, so it combines naturally with any real coordinate transform (compose maps) and does **not** require a range-invariant exterior. Approximate (not exact) and needs layer thickness/profile tuning; with Padé FD it can be implemented through a modified $L$ ($\partial_z\to s(z)^{-1}\partial_z$). This is the main fallback if the conformal map cannot be made identity at the top.

#### mikhin2004, mikhin2008 — exact discrete NLBC for high-order Padé
- **Method** (abstract): NLBCs for high-order FD PE obtained by $\mathcal Z$-transform of the discrete PE in a homogeneous medium; includes density jump at the NLBC interface, NLBC at an arbitrary impedance interface, and NLBCs for sources/starting field beyond the interface; "not limited by the order of Padé approximation or by the PE steps in range and depth"; robust for range steps >25 wavelengths and orders up to 10. 2008: analytic (closed-form) DTBC coefficients.
- **Relevance**: independently equivalent to Ehrhardt–Zisowsky/Lytaev; the impedance-interface NLBC is related to Lytaev 2022 (Fresnel reflection as nonlocal lower BC).

#### feshchenko_popov2017 — exact TBC with linear and quadratic potentials (downloaded arXiv, read)
- **Method**: narrow-angle PWE $2ik\,u_z+u_{xx}+k^2\alpha(x,z)u=0$ (here $z$ = evolution variable, $x$ transverse). Exterior $\alpha=a\,g(z)x$ (linear potential with *evolution-dependent* slope; $g=2/R(z)$ = curvature of a bent waveguide of variable curvature) or $\alpha=bx^2$.
- **Key equations**: gauge $u=\varphi\exp\!\big(i\frac{akx}{2}G(z)-i\frac{ka^2}{8}\int_0^zG^2\big)$, $G(z)=\int_0^zg$, gives $2ik\varphi_z+\varphi_{xx}+ikaG(z)\varphi_x=0$ (coefficients independent of $x$); Laplace transform in $x$ → exact NtD/DtN convolution TBCs (their eqs. (19)–(20)) with elementary kernels $\propto e^{i\Phi(z,\zeta)}/\sqrt{z-\zeta}$; for $g\equiv1$ reduces to the classical Airy-kernel TBC, for $a=0$ to BPP.
- **Limitations**: continuous (discretization "discussed" but not done), narrow-angle only.
- **Relevance to transformed coordinates**: the only exact TBC found for an exterior whose coefficients vary along the propagation direction — precisely the situation produced by a conformal bend map $w=R\log(\cdot)$ with variable curvature (Heiblum–Harris type) or by a terrain map whose exterior effect is a range-dependent linear tilt. No Padé/discrete analogue exists.

#### tyshchenko2019 — wide-angle mode PE with DTBC (downloaded preprint)
- **Method**: rational-linear (1,1) wide-angle mode PE from horizontal refraction equations; CN + fully discrete TBC.
- **Relevance**: recent use of DTBC with wide-angle (low order) PE; exterior homogeneous.

#### lytaev2018awpl — discrete NLBC for split-step Padé with modified refractive index (local, read)
- **Method**: $\psi_{xx}+\psi_{zz}+k^2m^2\psi=0$, $m^2=1+2N+2z/R$; $u=e^{-ikx}\psi$; $u^{n+1}=\prod_{l=1}^p\frac{1+a_lL_h}{1+b_lL_h}u^n$, $L_hu_j=k^{-2}D_h^2u_j+\alpha_ju_j$; Padé-$[n/m]$ of $\exp(ik\Delta x(\sqrt{1+L}-1))$ (Collins). Above $z_b=Jh$: $N\equiv0$, $L_h=k^{-2}D_h^2+(\beta+\gamma jh)$.
- **Key equations**: $\mathcal Z$-transformed exterior system $AD_h^2\hat{\mathbf u}_j=k^2(B-(\beta+\gamma jh)A)\hat{\mathbf u}_j$ with $A=\begin{pmatrix}\xi b_1&&\\-a_1&b_2&\\&\ddots&\ddots\\&-a_{p-1}&b_p\end{pmatrix}$-type bidiagonal, $B$ bidiagonal; generalized eigenproblem $B=APJP^{-1}$ (QZ) decouples into $D_h^2\chi_{j,l}=k^2(s_l-\beta-\gamma jh)\chi_{j,l}$. $\gamma=0$: $\chi_{j,l}=\mu_l^j$, $\mu_l-2+\mu_l^{-1}=k^2h^2(s_l-\beta)$, $|\mu_l|<1$. $\gamma\ne0$: discrete Airy, $\chi_{j,l}=J_{\sigma_{j,l}}(-2/(\gamma k^2h^3))$, $\sigma_{j,l}=j-2+(s_l-\beta)/(\gamma k^2h^3)$. $\hat D(\xi)=P\,\mathrm{diag}(\chi_{J-1,l}/\chi_{J,l})P^{-1}$; NLBC $u_{J-1}-D_0u_J^n=\sum_{m=1}^{n-1}D_mu_J^{n-m}$, $D_m$ by adaptive Filon–Clenshaw–Curtis.
- **Assumptions**: $N(x,z)\equiv0$ above $z_b$ (range-independent linear $m^2$ from Earth flattening), uniform $\Delta x,\Delta z$, Padé $[n/m]$ with $0\le m-n\le2$ for stability ($n<m$ suppresses evanescent oscillations). Coefficients independent of environment → tabulable.
- **Padé**: arbitrary order (examples [7/8]).
- **Transformed coordinates**: Earth flattening is itself a (conformal, $w=R\log(z/R)$-type) transform whose exterior effect is a range-independent linear term — the prototypical compatible case. A terrain map must reduce to identity (or a range-independent shift/linear profile) above $z_b$.

#### lytaev2020awpl — Numerov–Padé scheme + 4th-order discrete NLBC (local, read)
- **Method**: Padé-$[m/n]$ product in $x$; Numerov in $z$: $\partial_z^2\approx\Delta z^{-2}\delta^2(1+\alpha\delta^2)^{-1}$, $\alpha=1/12$; each factor gives a tridiagonal system $(\tau(1+\alpha\delta^2)+b_l\delta^2+b_l\tau h_j)v_j+b_l\tau\alpha\delta^2(h_jv_j)=(\ldots a_l\ldots)u_j$, $\tau=(k\Delta z)^2$, $h_j=m_j^2-1$.
- **Discrete dispersion relation** (key for topic 6):
  $$k_x^d=k+\frac{\ln\prod_{l=1}^p t_l}{i\Delta x},\quad t_l=\frac{1-a_l\frac{4}{k^2\Delta z^2}\big(\sin^2\frac{k\sin\theta\Delta z}{2}+4\alpha\sin^4\frac{k\sin\theta\Delta z}{2}\big)}{1-b_l\frac{4}{k^2\Delta z^2}\big(\sin^2\frac{k\sin\theta\Delta z}{2}+4\alpha\sin^4\frac{k\sin\theta\Delta z}{2}\big)}$$
  (sign convention as in paper); relative error of $k_x^d$ vs $k\cos\theta$ used to pick $(\Delta x,\Delta z,p)$ for a given $\theta_{\max}$.
- **NLBC**: exterior $L=k^{-2}\partial_z^2+\beta+\gamma z$; Numerov-discretized decoupled equations $(a_1+bj)\chi_{j+1}+(c+dj)\chi_j+(a_{-1}+bj)\chi_{j-1}=0$ with $b=\alpha\gamma\Delta z\tau$, $d=\gamma\Delta z\tau-2b$, $a_{\pm1}=1-\alpha\tau(s_l-\beta)\pm b$, $c=-2+2\alpha\tau(s_l-\beta)-\tau(s_l-\beta)$; ratio $\chi_{j-1}/\chi_j$ by generalized continued fraction (Lentz).
- **Stated limitations/claims**: "no theoretical restrictions on the maximum propagation angle, provided that the appropriate computational parameters are selected"; $\theta_{\max}$ depends on Padé order, $\Delta x$, $\Delta z$ and $z$-order. Terrain via staircase + impedance BC; with max slope 5°, $\theta_{\max}\approx5°$ assumed. [n−1/n], [n−2/n] L-stable.
- **Transformed coordinates**: a transform modifies $h_j$ (effective $m^2|F'|^2$) and possibly adds variable coefficients to $\delta^2$; the NLBC derivation is unchanged iff the exterior $h_j$ stays $\beta+\gamma z$.

#### lytaev2022radarconf — Fresnel reflection via discrete nonlocal lower BC (local, read)
- **Method**: semi-discrete dispersion $\xi=\prod_l\frac{1+a_ls}{1+b_ls}$ gives $p$ vertical wavenumbers $k_{z,l}(\xi)=k\sqrt{-s_l(\xi)}$; general solution $\chi_l=c_1e^{-ik\sqrt{-s_l}z}+c_2e^{ik\sqrt{-s_l}z}$ with $c_2=\Gamma(k_{z,l})c_1$; Numerov-discrete version $\chi_{l,j}=c_1\mu_l^{-j}+c_2\mu_l^j$, $\frac{\chi_{l,0}}{\chi_{l,1}}=\frac{1+\Gamma(k_{z,l})}{\mu_l^{-1}+\Gamma(k_{z,l})\mu_l}$; same $\mathcal Z^{-1}$ machinery. Any Fresnel coefficient (analytically continued to complex $k_z$), e.g. Brewster angle for V-pol.
- **Assumptions**: flat, range-independent lower interface with homogeneous ground half-space; $m\equiv1$ near $z=0$.
- **Transformed coordinates**: a flattening transform makes the terrain a flat lower boundary, which is exactly what this nonlocal lower BC needs — **but only if the ground half-space below is also mapped consistently and stays range-independent** (a conformal map that flattens the air–ground interface also distorts the ground medium by $|F'|^2$; the exact Fresnel NLBC then no longer holds unless $|F'|\approx1$ at the interface — i.e. the angle-dependent ground BC and the terrain map interact). Local Leontovich impedance BC transforms trivially (Neumann/Robin in the normal direction, normal derivative scaled by $|F'|$).

#### bamberger1988a, bamberger1988b — paraxial approximations in heterogeneous media
- **Method** (abstracts): general rational one-way (paraxial) approximations for variable coefficients; new "heterogeneous" PE variant with optimal reflection/transmission at interfaces (continuous dependence on interfaces); higher-order paraxial equations written as systems (one auxiliary unknown per Padé term, like the factored form); support/singular-support analysis; energy estimates ⇒ well-posedness.
- **Relevance**: after a conformal map the medium is heterogeneous ($m^2|F'|^2$ varies with $x$ and $z$); these papers give the well-posedness/energy framework for rational one-way operators with variable coefficients, and the condition on Padé coefficients (Padé coefficient positivity, as in Trefethen–Halpern) needed for stability. Not downloaded (SIAM).

#### trefethen_halpern1986 — well-posedness of one-way wave equations
- **Method**: one-way equations from rational $r(s)=p_m(s)/q_n(s)\approx\sqrt{1-s^2}$; well-posed iff $r$ interpolates $\sqrt{1-s^2}$ at enough points in $(-1,1)$; Padé approximants are well posed iff $(m,n)$ on two diagonals ($m=n$, $m=n+2$) of the Padé table; analogous results for Chebyshev and least-squares approximations.
- **Relevance**: justifies using rational *interpolation on an interval* (Lytaev 2021/2022/2026) as well as Padé; it is a continuous (not discrete) well-posedness result — stability of the fully discrete propagator Padé in $\exp$ is a separate (L-stability) question.

#### collins1993splitstep — split-step Padé
- **Method**: $u(r+\Delta r)=\exp\!\big(ik_0\Delta r(-1+\sqrt{1+X})\big)u\approx\prod_{j=1}^n\frac{1+\alpha_{j,n}X}{1+\beta_{j,n}X}u$, $X=k_0^{-2}(\rho\partial_z\rho^{-1}\partial_z+k^2-k_0^2)$; large range steps; valid for very wide angles; Collins also notes accuracy of rational approximation in $X$ vs discretized operator $X_h$. Stability constraints (later Collins 1993 "stability constraints") push eigenvalues to damp evanescent modes.
- **Relevance**: base algorithm of pywaveprop.

#### milinazzo1997 — rotated branch cut
- **Method**: stable Padé of $\sqrt{1+z}$ with branch cut rotated into the negative half-plane: $\sqrt{1+X}=e^{i\alpha/2}\sqrt{(1+X)e^{-i\alpha}}$ with complex coefficients analytic from the real ones; damps evanescent and complex (elastic) modes.
- **Relevance**: alternative stabilization when transformed operators have spectra off the real axis (e.g. complex $|F'|^2$ in PML or lossy ground).

#### yevick_thomson2000 — complex Padé approximants
- **Method**: two split-step Padé variants: Padé of a rotated square root, and complex-coefficient rational approximations; better behaviour in selected spectral regions, attenuating evanescent modes.

#### hadley1992pade — wide-angle BPM with Padé (optics)
- **Method**: multistep (product) Padé BPM — the optics analogue of Collins' factorization. Not downloaded.

#### vassallo1996 — limitations of wide-angle BPM in nonuniform systems
- **Method/claim** (abstract): reformulation clarifying BPM assumptions; arbitrary accuracy possible for weakly guiding systems even at ~40° with small steps and high Padé order; for strongly guiding (strongly nonuniform) systems the local-mode expansion involves evanescent forward and backward local modes that the one-way propagator cannot represent.
- **Relevance**: a conformal map turns a geometric irregularity into a refractive-index variation $m^2|F'|^2$; near sharp terrain features $|F'|$ varies strongly ⇒ "strongly guiding" regime where any one-way method (even exact $\sqrt{\cdot}$) incurs errors from neglected backscatter. "Angle-unrestricted" is therefore bounded by the one-way (no-backscatter) approximation in the mapped medium, not by Padé order.

#### zhang_sarris2015 — PE numerical dispersion analysis (tunnels)
- **Method** (abstract): von Neumann/dispersion analysis of CN, ADI, LOD-ADI and Mitchell–Fairweather ADI FD schemes for narrow-angle PE; separate PE-vs-Maxwell approximation error; grid selection criteria.
- **Relevance**: example of discrete dispersion-based mesh selection in radio PE (narrow-angle only).

#### lytaev2020iccsa, lytaev2023jmse, lytaev2023iccsa, lytaev2026nmp — mesh optimization via discrete dispersion
- **Method**: minimize cost $\propto 1/(\Delta x\Delta z)$ subject to $|k_x^d(\theta)-k\cos\theta|\le\epsilon$ (per unit range) on $[0,\theta_{\max}]$; 2023 JMSE adds reference sound speed and inhomogeneous media (underwater); 2023 ICCSA 3D with splitting and optimal propagation constant; 2026 NMP (downloaded, read): jointly selects $\Delta x,\Delta z$ and propagation constant $\beta$ accounting for the **range of the inhomogeneous wavenumber** $\xi\in[\xi_a,\xi_b]$, $\xi_d=-\frac{4}{\beta^2\Delta z^2}\big(\sin^2\frac{k_z\Delta z}{2}+4\alpha\sin^4\frac{k_z\Delta z}{2}\big)+\frac{k^2(x,z)}{\beta^2}-1$; for Padé the accuracy interval $[\xi_-,\xi_+]$ is problem-independent and tabulable; rational interpolation on $[\xi_a,\xi_b]$ gives 2–3× sparser grids (6–9× speed-up). Ignoring wavenumber variation picks grids that are too coarse.
- **Relevance to transforms**: a conformal map changes $k^2\to k^2|F'|^2$, widening $[\xi_a,\xi_b]$ by $\max|F'|^2-\min|F'|^2$ (large near steep terrain/corners) — this is exactly the input the 2026 algorithm needs; the dispersion relation itself is unchanged (same $L$ structure), so the Lytaev mesh-optimization framework carries over directly to the conformally mapped equation. For non-conformal (blended) maps, mixed derivatives change the discrete symbol and the dispersion relation must be rederived.

#### lytaev2020nmp — Padé simultaneously in x and z (downloaded)
- **Method**: expresses $\partial_z^2=g(\delta^2)=\big(f^{-1}(\delta^2)\big)^2$, $f(k_z)=2(\cosh(ik_z\Delta z)-1)$, and Padé-approximates $\exp(ik\Delta x(\sqrt{1+k^{-2}g(\delta^2)}-1))\approx\prod\frac{1+a'_l\delta^2}{1+b'_l\delta^2}$ directly in $\delta^2$ (as Collins/Ehrhardt–Zisowsky); arbitrary refractive index is then reintroduced; "no restriction on the maximum propagation angle"; staircase terrain with impedance BC at $z=h(x)$.

#### lytaev2021iccs, lytaev2022jocs — rational approximation on an interval
- **Method**: replace local Padé at $\xi=0$ by Chebyshev-type rational approximation/interpolation of $\exp(ik\Delta x(\sqrt{1+\xi}-1))$ on $\xi\in[\xi_a,\xi_b]$, $\xi=-k_z^2/k^2+(n^2-1)$; relationship between refractive-index variation and maximum angle established; only coefficients change, same scheme. 
- **Relevance**: well suited to the mapped equation where $n^2|F'|^2-1$ spans a known interval.

#### lytaev2022iccs, lytaev2023jocs — differential evolution for coefficients and grid
- **Method**: choose rational coefficients and grid steps jointly by minimizing discrete dispersion error (differential evolution), with a stability constraint ($|t_l|\le1$-type) in the optimization; compared to Padé and rational interpolation.

#### lytaev2021iccsa — improved-accuracy split-step Padé (troposphere, irregular terrain)
- Per search abstract: higher-order FD PE over irregular terrain with Padé applied simultaneously in longitudinal and transversal coordinates; sparser grids. Not downloaded.

#### ramamurti2025 — AAA rational approximations for PE (downloaded arXiv)
- **Method**: AAA (adaptive Antoulas–Anderson) rational approximation of PE operators; compared with rotated-branch-cut and stability-constraint methods; stable for fluid–elastic waveguides where those fail; enables split-step Padé for fluid–elastic PE.
- **Relevance**: robust generic way to get stable rational coefficients for nonstandard spectra (e.g. complex $|F'|^2$ with PML, or wide $\xi$-intervals after a conformal map).
