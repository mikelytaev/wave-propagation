"""Generate the results sections of ``papers/terrain_transform/main.tex`` from the
JSON files written by the validation and demonstration scripts::

    python scripts/terrain_transform_make_tables.py
"""
import json
import os

DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform")
FIG = os.path.join(DIR, "figures")


def load(name):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return rf"${m}\cdot10^{{{int(e)}}}$"


def validation():
    out = []
    out.append(r"""\subsection{Flat ground}
Over flat ground the map is the identity and the terrain-following propagator
reproduces \texttt{RationalHelmholtzPropagator}, including the NLBC, to
$4\cdot10^{-14}$ (unit test \texttt{TestFlatGroundEquivalence}).
""")
    slope = load("slope_exact.json")
    refine = load("slope_staircase_refinement.json")
    if slope:
        rows = [r for r in slope if abs(r["precision"] - 1e-2) < 1e-12]
        out.append(r"""\subsection{Exact solution: beam reflected by a perfectly conducting slope}\label{sec:slope}
The incident field is a complex-source-point beam at \SI{300}{MHz}, an exact
solution of the Helmholtz equation, with waist at $(-50,30)$\,m, Rayleigh parameter
\SI{100}{m} and direction $-10^\circ$. Its image in the line $z=x\tan\theta_s$ gives
the exact field above the Dirichlet slope. The reflected beam leaves at
$2\theta_s+10^\circ$, i.e.\ $30$, $50$ and $70^\circ$ for $\theta_s=10$, $20$ and $30^\circ$.
\begin{itemize}
\item Both solvers start from the exact field: on the curved trajectory $\xi=0$
(boundary not mirrored at $x=0$) and on the line $x=0$ respectively.
\item Both use Pad\'e $(7,8)$, the aperture $\theta_{\max}=10^\circ+2\theta_s+5^\circ$
and grids chosen by the respective optimizers.
\item The error is $\max|u-u_{\rm exact}|/\max|u_{\rm exact}|$ over the computational
domain above the ground.
\end{itemize}
""")
        out.append(r"""\begin{table}[h]\centering
\caption{Exact slope test (Pad\'e $(7,8)$, precision parameter $10^{-2}$).}\label{tab:slope}
\begin{tabular}{rrrrrrrr}\toprule
& & \multicolumn{3}{c}{terrain-following} & \multicolumn{3}{c}{staircase}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
$\theta_s$ & aperture & $\Delta x$, m & $\Delta z$, m & error & $\Delta x$, m & $\Delta z$, m & error\\\midrule
""")
        for r in rows:
            t, s = r["tf"], r["staircase"]
            out.append(f"${r['slope_deg']:.0f}^\\circ$ & ${r['theta_max_deg']:.0f}^\\circ$ & {t['dx']:.2f} & {t['dz']:.4f} & "
                       f"{sci(t['err'])} & {s['dx']:.2f} & {s['dz']:.4f} & {sci(s['err'])}\\\\\n")
        out.append(r"""\bottomrule\end{tabular}\end{table}
""")
        out.append(r"""\begin{figure}[t]\centering
\includegraphics[width=\textwidth]{slope_exact_30deg_maps.png}\\
\includegraphics[width=0.49\textwidth]{slope_exact_20deg_error.png}\hfill
\includegraphics[width=0.49\textwidth]{slope_exact_30deg_error.png}
\caption{Beam reflected by a perfectly conducting slope. Top: $|u|$ for
$\theta_s=30^\circ$ (reflected beam at $70^\circ$); the terrain-following domain is
bounded by the curved trajectories $\xi=0$ and $\xi=x_{\max}$. Bottom: error versus
range for $\theta_s=20^\circ$ and $30^\circ$.}\label{fig:slope}
\end{figure}
""")
        txt = (r"""The terrain-following error (\cref{tab:slope}, \cref{fig:slope}) is
$10^{-4}$--$3\cdot10^{-3}$ up to a reflected beam at $70^\circ$, on grids
comparable to those of the staircase. The staircase loses most of the reflected
beam, and its error is $0.5$--$0.8$.""")
        if refine:
            r = refine["runs"]
            txt += (rf""" Refining the staircase grid for $\theta_s=20^\circ$ to
$\Delta z={r[2]['dz']}$\,m ($\lambda/200$) leaves the error at {r[2]['err']:.2f}, and with
$\Delta x={r[3]['dx']}$\,m it is still {r[3]['err']:.2f}: the staircase error is
geometric and decreases only slowly.""")
        out.append(txt + "\n")
        out.append(r"""
The flux-conserving step is essential:
\begin{itemize}
\item With the Liouville rescaling alone ($q=0$) the error in the $20^\circ$ case is
$6.4\cdot10^{-2}$, an amplitude drift along the reflected beam. Without the
rescaling it is larger still.
\item With unrotated $[2/2]$ factors a spurious evanescent component grows to
$0.14$ (\cref{sec:transformed}).
\item The rotated factors give $1.2\cdot10^{-3}$.
\end{itemize}
""")
    wedge = load("wedge_uwa_25hz.json")
    if wedge:
        tf, st, rc = wedge["tf_0.001"], wedge["stair_0.001"], wedge["ram_coarse"]
        conv = wedge.get("convergence")
        out.append(r"""\subsection{Penetrable wedge}
The upslope wedge of the ASA benchmark \cite{jensen1990benchmark} has the
following parameters:
\begin{itemize}
\item \SI{25}{Hz}, water $c=\SI{1500}{m/s}$;
\item bottom \SI{1700}{m/s}, $\rho=1.5$, \SI{0.5}{dB/\lambda};
\item source at \SI{100}{m};
\item depth \SI{200}{m} with a flat lead-in to \SI{500}{m}, then a $2.86^\circ$ upslope to \SI{3.5}{km}.
\end{itemize}
All solvers use the same smoothed bathymetry ($\sigma=\SI{20}{m}$) and start from
the self-starter of RAM~1.5 at $r=\SI{1}{m}$; a patched copy of RAM writes it out.
RAM is single precision. On fine grids its tridiagonal systems lose all
accuracy: at $\Delta r=\SI{1}{m}$, $\Delta z=\SI{0.05}{m}$ the field is wrong by
\SI{77}{dB}. We therefore rebuilt it in double precision. It then converges: the
grids $(2.5,0.1)$\,m and $(1,0.025)$\,m differ by \SI{0.013}{dB} (median),
and the latter is the reference.
""")
        out.append(r"""\begin{table}[h]\centering
\caption{Penetrable wedge: median / 90th percentile of $|\Delta\mathrm{TL}|$ (dB) with
respect to the RAM reference, $r>\SI{200}{m}$.}\label{tab:wedge}
\begin{tabular}{lcc}\toprule
solver (grid) & $z=\SI{30}{m}$ & $z=\SI{150}{m}$\\\midrule
""")
        def row(name, d):
            return (f"{name} & {d['z30']['median_abs_db']:.2f} / {d['z30']['p90_abs_db']:.2f} & "
                    f"{d['z150']['median_abs_db']:.2f} / {d['z150']['p90_abs_db']:.2f}\\\\\n")
        out.append(row(rf"terrain-following ($\Delta x={tf['dx']:.0f}$, $\Delta z={tf['dz']:.1f}$\,m)", tf))
        out.append(row(rf"same scheme, stair-step bottom ($\Delta x={st['dx']:.0f}$, $\Delta z={st['dz']:.1f}$\,m)", st))
        out.append(row(r"RAM ($\Delta r=10$, $\Delta z=0.5$\,m)", rc))
        out.append(r"""\bottomrule\end{tabular}\end{table}
""")
        out.append(r"""\begin{figure}[t]\centering
\includegraphics[width=0.66\textwidth]{wedge_uwa_25hz_cuts.png}\hfill
\includegraphics[width=0.33\textwidth]{wedge_uwa_25hz_convergence.png}
\caption{Penetrable wedge. Left: $20\log|p|$ at \SI{30}{m} and \SI{150}{m} depth.
Right: median error versus $\Delta z$ at the same range step.}\label{fig:wedge}
\end{figure}
""")
        if conv:
            out.append(rf"""The terrain-following solution agrees with the reference to
{tf['z30']['median_abs_db']:.2f}--{tf['z150']['median_abs_db']:.2f}\,dB (\cref{{tab:wedge}},
\cref{{fig:wedge}}). The stair-step bottom in the same scheme gives
{st['z30']['median_abs_db']:.1f}--{st['z150']['median_abs_db']:.1f}\,dB.

Refining $\Delta z$ from 2 to \SI{{0.25}}{{m}} does not reduce either error
(terrain-following {min(conv['tf']):.2f}--{max(conv['tf']):.2f}\,dB, stair-step
{min(conv['stair']):.2f}--{max(conv['stair']):.2f}\,dB), and neither does refining
$\Delta x$ to \SI{{1.25}}{{m}} or raising the flux order to $q=3$. The two floors have
different origins:
\begin{{itemize}}
\item \emph{{Stair-step.}} This is the energy-conservation defect of one-way models
across vertical density steps \cite{{porter1991energy}}. Each step makes an error
$O(\Delta h)$ and there are $O(1/\Delta h)$ steps, so the bias does not vanish.
RAM avoids it through its density-scaled variable.
\item \emph{{Terrain-following.}} The residual of about \SI{{0.1}}{{dB}} is the
difference between two one-way models of the same sloping interface (RAM's
stair-steps, ours continuous). It is grid-independent. It is larger at the
receiver that lies in the sediment beyond \SI{{1.5}}{{km}}.
\end{{itemize}}
""")
    return "".join(out)


def realdata():
    out = []
    tro = load("troposphere_highlands.json")
    if tro:
        out.append(rf"""\subsection{{Troposphere: Scottish Highlands, GFS refractivity}}
The path runs from \ang{{57.2163}}N \ang{{4.9214}}W (top of a \SI{{664}}{{m}} ridge) to
\ang{{57.40}}N \ang{{4.60}}W (\SI{{28}}{{km}}). The data are:
\begin{{itemize}}
\item elevation tiles of \texttt{{pywaveprop.environment}} at \ang{{0.0005}} resolution
(terrain from 166 to \SI{{664}}{{m}}, smoothed with $\sigma=\SI{{100}}{{m}}$, maximum slope
${tro['slope_deg']:.1f}^\circ$);
\item the modified refractivity of the GFS analysis {tro['cycle']} interpolated
along the path and averaged over range (\cref{{fig:tropo}}, right).
\end{{itemize}}
A \SI{{300}}{{MHz}} H-polarised antenna on a \SI{{30}}{{m}} mast (beam $4^\circ$, tilted
$1^\circ$ down) radiates over perfectly conducting ground. The aperture is
$5^\circ+2\cdot{tro['slope_deg']:.1f}^\circ={tro['aperture_deg']:.1f}^\circ$
(\cref{{cor:angles}}) for both solvers, and the staircase solver sees the same
smoothed terrain.
""")
        a = tro.get('tf_0.01_vs_tf_0.001_agl10')
        b = tro.get('st_0.01_vs_st_0.001_agl10')
        c = tro.get('tf_0.001_vs_st_0.001_agl10')
        c50 = tro.get('tf_0.001_vs_st_0.001_agl50')
        if a and b and c:
            out.append(rf"""Path loss \SI{{10}}{{m}} above the ground (median / 90th percentile of the difference):
\begin{{itemize}}
\item the two precision settings of the terrain-following solver differ by
{a['median_db']:.2f} / {a['p90_db']:.2f}\,dB;
\item those of the staircase solver differ by {b['median_db']:.2f} / {b['p90_db']:.2f}\,dB;
\item the two solvers at the finer setting differ by {c['median_db']:.2f} /
{c['p90_db']:.2f}\,dB (at \SI{{50}}{{m}}: {c50['median_db']:.2f} / {c50['p90_db']:.2f}\,dB).
\end{{itemize}}
Each solver is thus converged with respect to its own grid; the two agree to
about \\SI{{1}}{{dB}} (median) \\SI{{10}}{{m}} above the ground and \\SI{{0.4}}{{dB}} at
\\SI{{50}}{{m}}. The largest differences, up to a few
decibels, occur on the slopes facing the transmitter (21--23\,km and the first
kilometres). There the staircase solution carries small-scale ripple that the
smooth-boundary solution does not, consistent with scattering by the steps
(\cref{{sec:slope}}).
Run times: terrain-following {tro['time_tf_0.01']:.0f} and {tro['time_tf_0.001']:.0f}\,s,
staircase {tro['time_st_0.01']:.0f} and {tro['time_st_0.001']:.0f}\,s.
""")
        out.append(r"""\begin{figure}[t]\centering
\includegraphics[width=0.74\textwidth]{troposphere_highlands.png}\hfill
\includegraphics[width=0.25\textwidth]{troposphere_highlands_M.png}
\caption{Scottish Highlands, \SI{300}{MHz}. Path loss of the terrain-following
solver (top), path loss \SI{10}{m} and \SI{50}{m} above the ground for both solvers
and two precision settings, and the GFS M-profile.}\label{fig:tropo}
\end{figure}
""")
    import glob
    runs_uw = sorted((load(os.path.basename(f)) for f in glob.glob(os.path.join(FIG, "underwater_bermuda_*deg.json"))),
                     key=lambda d: -d["aperture_deg"])
    uw = runs_uw[0] if runs_uw else None
    if uw:
        out.append(rf"""\subsection{{Underwater acoustics: Bermuda pedestal, Argo sound speed}}
The path runs from \ang{{32.475}}N \ang{{64.598}}W (\SI{{100}}{{m}} deep, on the
edge of the Bermuda platform) to \ang{{32.70}}N \ang{{64.40}}W (\SI{{3400}}{{m}}).
The data are:
\begin{{itemize}}
\item bathymetry from the elevation tiles at \ang{{0.002}} (smoothed with
$\sigma=\SI{{500}}{{m}}$, maximum slope ${uw['slope_deg']:.1f}^\circ$);
\item the mean Argo sound speed profile of the area (2020--2023) to \SI{{2000}}{{m}},
extended with the deep isothermal gradient \SI{{0.0165}}{{s^{{-1}}}};
\item a fluid bottom (\SI{{1700}}{{m/s}}, 1.5, \SI{{0.5}}{{dB/\lambda}}) whose interface
follows the bathymetry.
\end{{itemize}}
A \SI{{100}}{{Hz}} source \SI{{50}}{{m}} deep radiates down the slope. All solvers start
from RAM's self-starter. The terrain-following domain extends to
\SI{{{uw['domain_depth_m']:.0f}}}{{m}}, and the NLBC lies in the homogeneous bottom. The
reference is double-precision RAM at $\Delta r=\SI{{2.5}}{{m}}$,
$\Delta z=\SI{{0.125}}{{m}}$; RAM at $(5, 0.25)$\,m differs from it by
{uw['ram_coarse']['z50']['median_abs_db']:.2f}--{uw['ram_coarse']['z500']['median_abs_db']:.2f}\,dB (median).
""")
        out.append(r"""\begin{table}[h]\centering
\caption{Bermuda pedestal: median / 90th percentile of $|\Delta\mathrm{TL}|$ (dB) with
respect to RAM ($\Delta r=2.5$, $\Delta z=0.125$\,m), $r>\SI{500}{m}$.}\label{tab:bermuda}
\begin{tabular}{llcc}\toprule
aperture & solver (grid) & $z=\SI{50}{m}$ & $z=\SI{500}{m}$\\\midrule
""")

        def row(ap, name, d):
            return (f"{ap} & {name} & {d['z50']['median_abs_db']:.2f} / {d['z50']['p90_abs_db']:.2f} & "
                    f"{d['z500']['median_abs_db']:.2f} / {d['z500']['p90_abs_db']:.2f}\\\\\n")
        for res, ap in ((uw, f"${uw['aperture_deg']:.0f}^\\circ$"),):
            if res is None:
                continue
            out.append(row(ap, rf"terrain-following ($\Delta x={res['tf']['dx']:.1f}$, $\Delta z={res['tf']['dz']:.2f}$\,m)", res['tf']))
            out.append(row(ap, rf"stair-step ($\Delta x={res['stair']['dx']:.1f}$, $\Delta z={res['stair']['dz']:.2f}$\,m)", res['stair']))
        out.append(row("--", r"RAM ($\Delta r=5$, $\Delta z=0.25$\,m)", uw['ram_coarse']))
        out.append(rf"""\bottomrule\end{{tabular}}\end{{table}}
\begin{{figure}}[t]\centering
\includegraphics[width=0.8\textwidth]{{underwater_bermuda_{int(uw['aperture_deg'])}deg.png}}
\caption{{Bermuda pedestal, \SI{{100}}{{Hz}}: field of the terrain-following solver
(${int(uw['aperture_deg'])}^\circ$ aperture) and $20\log|p|$ at \SI{{50}}{{m}} and \SI{{500}}{{m}} depth.}}\label{{fig:bermuda}}
\end{{figure}}
""")
        tf, st = uw['tf'], uw['stair']
        txt = rf"""With a ${uw['aperture_deg']:.0f}^\circ$ aperture (\cref{{tab:bermuda}}, \cref{{fig:bermuda}}) the
terrain-following solution differs from the RAM reference by
{tf['z50']['median_abs_db']:.2f} and {tf['z500']['median_abs_db']:.2f}\,dB (median at 50 and
\SI{{500}}{{m}}); the stair-step bottom gives {st['z50']['median_abs_db']:.2f} and
{st['z500']['median_abs_db']:.2f}\,dB."""
        sens = load("underwater_bermuda_sensitivity.json")
        if sens:
            r = sens["runs"]
            txt += rf"""

The difference is not a discretisation error:
\begin{{itemize}}
\item halving $\Delta x$ changes it by \SI{{0.03}}{{dB}} ({r[0]['z50']:.2f}/{r[0]['z500']:.2f} $\to$ {r[1]['z50']:.2f}/{r[1]['z500']:.2f}\,dB);
\item raising the flux order to $q=3$ changes it by less than \SI{{0.02}}{{dB}} ({r[3]['z50']:.2f}/{r[3]['z500']:.2f}\,dB);
\item lowering $\etab$ from \SI{{3600}}{{m}} to \SI{{1200}}{{m}} reduces it to {r[2]['z50']:.2f}/{r[2]['z500']:.2f}\,dB.
\end{{itemize}}
With $\etab=\max B$ the $\sigma$-map compresses the \SI{{100}}{{m}} shelf column 36 times, so $r$ jumps by a factor of about 30 across the seabed. Part of the
difference is therefore the second-order treatment of this extreme interface
row, and a less compressive choice of $\etab$ is preferable for large depth
contrasts. The remainder is of the same order as the difference between the two
stair-step solvers (ours and RAM). Over a \ang{{16}} penetrable slope, RAM is
itself a stair-step one-way model, and we have no exact reference to say which
one-way treatment is closer to the full solution. The exact test of
\cref{{sec:slope}} and the wedge of \cref{{tab:wedge}} are the discriminating
cases."""
        out.append(txt + rf"""
Times: terrain-following {tf['time_s']:.0f}\,s, stair-step {st['time_s']:.0f}\,s, RAM
{uw['ram_coarse_time_s']:.0f}\,s and {uw['ram_ref_time_s']:.0f}\,s.
""")
    return "".join(out)


if __name__ == "__main__":
    with open(os.path.join(DIR, "results_validation.tex"), "w") as f:
        f.write(validation())
    with open(os.path.join(DIR, "results_realdata.tex"), "w") as f:
        f.write(realdata())
    print("written")
