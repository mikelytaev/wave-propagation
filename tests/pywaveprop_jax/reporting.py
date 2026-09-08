"""
Per-test result reports.

Every test wrapped in :class:`TestReport` writes, into the report directory,

* ``<test_name>.html`` -- a self-contained page with the explanation of the
  test, the setup tables, the figures (embedded as PNG), the checked metrics
  with their thresholds and the pass/fail outcome;
* ``<test_name>.npz`` -- the raw arrays (fields, grids, cuts) used for the
  figures, so the results can be re-analysed without re-running the test;
* ``<test_name>.json`` -- a short metadata record used to build ``index.html``,
  which lists all reports found in the directory.

The directory is taken from the ``TEST_REPORTS`` environment variable
(default ``test_reports/`` at the repository root); ``TEST_REPORTS=0`` disables
reporting entirely, in which case all methods are no-ops.

Typical use::

    with TestReport(self, "Explanation of what is checked and why.") as rep:
        rep.params("Source", freq_hz=3e8, height_m=30)
        rep.field2d("Field", field, x, z)
        rep.cuts("Horizontal cuts", {"FD": fd, "SSF": ssf}, heights=[30, 100])
        rep.check("median dB difference at z=30 m", med, "<", 0.5)
        rep.assert_all()

``assert_all`` raises for the failed checks; any other exception escaping the
``with`` block is also recorded in the report before being re-raised.
"""
import base64
import datetime as _dt
import html
import io
import json
import os
import pathlib
import traceback

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ENV = os.environ.get("TEST_REPORTS", "")
ENABLED = _ENV not in ("0", "false", "no", "off")
REPORT_DIR = pathlib.Path(_ENV) if _ENV and ENABLED else _ROOT / "test_reports"

if ENABLED:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, complex):
        return f"{v.real:.6g}{v.imag:+.6g}j"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, np.ndarray):
        return np.array2string(v, precision=4, max_line_width=80, threshold=12)
    return str(v)


def _db(a):
    return 20 * np.log10(np.abs(np.asarray(a)) + 1e-30)


class TestReport:
    """Collects explanation, parameters, figures, raw data and checks of one test."""

    __test__ = False  # not a test class, despite the name

    def __init__(self, test_case, description: str, title: str = None, suffix: str = None):
        """``suffix`` distinguishes several reports written by one test method
        (e.g. one per ``subTest``)."""
        self.test_case = test_case
        test_id = test_case.id()
        # drop the leading package path, keep "<module>.<Class>.<method>"
        self.name = ".".join(test_id.split(".")[-3:])
        if suffix:
            self.name += "_" + _slug(suffix)
        self.title = title or self.name.split(".")[-1].replace("_", " ")
        self.description = description
        self._blocks = []        # (kind, payload) in insertion order
        self._checks = []        # dict(label, value, op, threshold, ok)
        self._arrays = {}
        self._status = "passed"
        self._error = None
        self._started = _dt.datetime.now()

    # ------------------------------------------------------------ content API
    def text(self, paragraph: str):
        """Free-form explanation paragraph."""
        if ENABLED:
            self._blocks.append(("text", paragraph))
        return self

    def params(self, section: str, **values):
        """Table of setup parameters."""
        if ENABLED:
            self._blocks.append(("table", (section, [(k, _fmt(v)) for k, v in values.items()])))
        return self

    def table(self, section: str, header, rows):
        if ENABLED:
            self._blocks.append(("grid", (section, list(header), [[_fmt(c) for c in r] for r in rows])))
        return self

    def data(self, **arrays):
        """Raw arrays to store in the ``.npz`` file."""
        if ENABLED:
            for k, v in arrays.items():
                self._arrays[k] = np.asarray(v)
        return self

    def figure(self, fig, caption: str):
        """Embed a matplotlib figure (closed afterwards)."""
        if not ENABLED:
            return self
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        self._blocks.append(("figure", (base64.b64encode(buf.getvalue()).decode("ascii"), caption)))
        return self

    def field2d(self, caption: str, fields: dict, x_grid, z_grid, dyn_range_db=60.0, terrain=None):
        """Side-by-side 2D amplitude maps (dB re. common maximum) of one or more fields.

        ``fields`` maps label -> complex array of shape ``(len(x_grid), len(z_grid))``.
        """
        if not ENABLED:
            return self
        x = np.asarray(x_grid)
        z = np.asarray(z_grid)
        n = len(fields)
        fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.2), squeeze=False)
        vmax = max(float(np.max(_db(f)[len(x) // 10:, :])) for f in fields.values())
        for ax, (label, f) in zip(axes[0], fields.items()):
            amp = _db(f).T
            pm = ax.pcolormesh(x, z, amp, shading="auto", cmap="jet", vmin=vmax - dyn_range_db, vmax=vmax)
            if terrain is not None:
                ax.plot(x, np.asarray(terrain), "k-", lw=1.5)
            ax.set_title(label)
            ax.set_xlabel("Range (m)")
            ax.set_ylabel("Height (m)")
            fig.colorbar(pm, ax=ax, label="|u| (dB)")
            self._arrays[f"field_{_slug(label)}"] = np.asarray(f)
        self._arrays.setdefault("x_grid", x)
        self._arrays.setdefault("z_grid", z)
        return self.figure(fig, caption)

    def cuts(self, caption: str, fields: dict, x_grid, z_grid, heights, dyn_range_db=80.0, diff=True):
        """Horizontal cuts |u|(x) in dB at the given heights, with the dB difference
        to the first field when ``diff`` is set.

        Values more than ``dyn_range_db`` below the maximum of the cut (numerical
        floor, e.g. ahead of the beam) are clipped, and the difference is not
        drawn where either field is below that floor.
        """
        if not ENABLED:
            return self
        x = np.asarray(x_grid)
        z = np.asarray(z_grid)
        labels = list(fields)
        ncols = 2 if (diff and len(labels) > 1) else 1
        fig, axes = plt.subplots(len(heights), ncols, figsize=(6.5 * ncols, 3.0 * len(heights)), squeeze=False)
        for row, h in zip(axes, heights):
            iz = int(np.argmin(np.abs(z - h)))
            cuts = {label: _db(fields[label][:, iz]) for label in labels}
            top = max(float(np.max(c)) for c in cuts.values())
            floor = top - dyn_range_db
            ref = None
            for label in labels:
                cut = cuts[label]
                row[0].plot(x, np.maximum(cut, floor), label=label)
                self._arrays[f"cut_{_slug(label)}_z{h:g}"] = np.asarray(fields[label][:, iz])
                if ref is None:
                    ref = cut
                elif ncols == 2:
                    d = np.where((cut > floor) & (ref > floor), cut - ref, np.nan)
                    row[1].plot(x, d, label=f"{label} − {labels[0]}")
            row[0].set_title(f"|u| at z = {h:g} m")
            row[0].set_xlabel("Range (m)")
            row[0].set_ylabel("dB")
            row[0].set_ylim(floor, top + 5)
            row[0].legend()
            if ncols == 2:
                row[1].set_title(f"difference at z = {h:g} m")
                row[1].set_xlabel("Range (m)")
                row[1].set_ylabel("dB")
                row[1].axhline(0, color="k", lw=0.5)
                row[1].legend()
        return self.figure(fig, caption)

    def lines(self, caption: str, x, series: dict, xlabel="", ylabel="", title=""):
        """Generic line plot of ``series`` (label -> y values) versus ``x``."""
        if not ENABLED:
            return self
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for label, y in series.items():
            ax.plot(np.asarray(x), np.asarray(y), label=label)
            self._arrays[f"series_{_slug(label)}"] = np.asarray(y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        return self.figure(fig, caption)

    # ------------------------------------------------------------- checks
    def check(self, label: str, value, op: str, threshold, unit: str = ""):
        """Record a metric and whether it satisfies ``value <op> threshold``.

        Nothing is raised here; call :meth:`assert_all` at the end of the test.
        """
        ok = bool(_OPS[op](value, threshold))
        self._checks.append(dict(label=label, value=value, op=op, threshold=threshold, unit=unit, ok=ok))
        return ok

    def assert_all(self):
        failed = [c for c in self._checks if not c["ok"]]
        if failed:
            msg = "; ".join(f"{c['label']}: {_fmt(c['value'])} !{c['op']} {_fmt(c['threshold'])} {c['unit']}"
                            for c in failed)
            self.test_case.fail(msg)

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._status = "failed" if issubclass(exc_type, AssertionError) else "error"
            self._error = "".join(traceback.format_exception(exc_type, exc, tb))
        elif any(not c["ok"] for c in self._checks):
            self._status = "failed"
        self._write()
        return False

    def _write(self):
        if not ENABLED:
            return
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.name.replace(".", "_")
        if self._arrays:
            np.savez_compressed(REPORT_DIR / f"{stem}.npz", **self._arrays)
        (REPORT_DIR / f"{stem}.html").write_text(self._render(stem), encoding="utf-8")
        meta = dict(name=self.name, title=self.title, status=self._status,
                    time=self._started.isoformat(timespec="seconds"),
                    duration_s=(_dt.datetime.now() - self._started).total_seconds(),
                    checks=len(self._checks), failed=sum(not c["ok"] for c in self._checks),
                    description=_first_sentence(self.description))
        (REPORT_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        write_index()

    # ------------------------------------------------------------- rendering
    def _render(self, stem):
        e = html.escape
        parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{e(self.title)}</title>",
                 _CSS, "</head><body>",
                 f"<p class='nav'><a href='index.html'>&larr; all reports</a></p>",
                 f"<h1>{e(self.title)} <span class='status {self._status}'>{self._status.upper()}</span></h1>",
                 f"<p class='meta'>{e(self.name)} &middot; {self._started:%Y-%m-%d %H:%M:%S}"
                 + (f" &middot; raw data: <a href='{stem}.npz'>{stem}.npz</a>" if self._arrays else "") + "</p>",
                 "<h2>What is tested</h2>"]
        parts += [f"<p>{e(p.strip())}</p>" for p in self.description.strip().split("\n\n") if p.strip()]
        for kind, payload in self._blocks:
            if kind == "text":
                parts.append(f"<p>{e(payload)}</p>")
            elif kind == "table":
                section, rows = payload
                parts.append(f"<h3>{e(section)}</h3><table class='kv'>")
                parts += [f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in rows]
                parts.append("</table>")
            elif kind == "grid":
                section, header, rows = payload
                parts.append(f"<h3>{e(section)}</h3><table class='grid'><tr>"
                             + "".join(f"<th>{e(str(h))}</th>" for h in header) + "</tr>")
                parts += ["<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows]
                parts.append("</table>")
            elif kind == "figure":
                png, caption = payload
                parts.append(f"<figure><img src='data:image/png;base64,{png}'/>"
                             f"<figcaption>{e(caption)}</figcaption></figure>")
        if self._checks:
            parts.append("<h2>Checks</h2><table class='grid'><tr><th>metric</th><th>value</th>"
                         "<th>criterion</th><th>result</th></tr>")
            for c in self._checks:
                cls = "ok" if c["ok"] else "bad"
                parts.append(f"<tr class='{cls}'><td>{e(c['label'])}</td><td>{e(_fmt(c['value']))} {e(c['unit'])}</td>"
                             f"<td>{e(c['op'])} {e(_fmt(c['threshold']))} {e(c['unit'])}</td>"
                             f"<td>{'pass' if c['ok'] else 'FAIL'}</td></tr>")
            parts.append("</table>")
        if self._error:
            parts.append(f"<h2>Error</h2><pre>{e(self._error)}</pre>")
        parts.append("</body></html>")
        return "\n".join(parts)


def write_index():
    """(Re)build ``index.html`` from the ``.json`` records in the report directory."""
    if not ENABLED or not REPORT_DIR.exists():
        return
    records = []
    for p in sorted(REPORT_DIR.glob("*.json")):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    e = html.escape
    rows = []
    for r in sorted(records, key=lambda r: r["name"]):
        stem = r["name"].replace(".", "_")
        rows.append(f"<tr class='{'ok' if r['status'] == 'passed' else 'bad'}'>"
                    f"<td><a href='{stem}.html'>{e(r['title'])}</a><br><small>{e(r['name'])}</small></td>"
                    f"<td>{e(r['description'])}</td><td>{e(r['status'])}</td>"
                    f"<td>{r['checks'] - r['failed']}/{r['checks']}</td>"
                    f"<td>{e(r['time'])}</td><td>{r['duration_s']:.1f} s</td></tr>")
    page = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Test reports</title>" + _CSS
            + "</head><body><h1>Test reports</h1>"
            + f"<p class='meta'>{len(records)} reports, "
              f"{sum(r['status'] == 'passed' for r in records)} passed &middot; "
              f"generated {_dt.datetime.now():%Y-%m-%d %H:%M:%S}</p>"
            + "<table class='grid'><tr><th>test</th><th>what is tested</th><th>status</th>"
              "<th>checks</th><th>run</th><th>duration</th></tr>" + "\n".join(rows) + "</table></body></html>")
    (REPORT_DIR / "index.html").write_text(page, encoding="utf-8")


def _first_sentence(text):
    text = " ".join(text.split())
    end = text.find(". ")
    return text if end < 0 else text[:end + 1]


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_")


_CSS = """<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222;line-height:1.45}
h1{font-size:1.6em} h2{margin-top:1.6em;border-bottom:1px solid #ddd} h3{margin-bottom:.3em}
.meta,.nav{color:#666;font-size:.9em}
.status{font-size:.55em;vertical-align:middle;padding:.15em .6em;border-radius:.6em;color:#fff}
.status.passed{background:#2e8b57} .status.failed,.status.error{background:#c0392b}
table{border-collapse:collapse;margin:.4em 0 1em} td,th{border:1px solid #ddd;padding:.25em .6em;text-align:left;vertical-align:top}
table.kv th{background:#f6f6f6;font-weight:600;width:14em} table.grid th{background:#f6f6f6}
tr.ok td:last-child{color:#2e8b57;font-weight:600} tr.bad td{background:#fdecea} tr.bad td:last-child{color:#c0392b;font-weight:700}
figure{margin:1em 0} figure img{max-width:100%;border:1px solid #eee} figcaption{color:#555;font-size:.9em;margin-top:.3em}
pre{background:#f6f6f6;padding:.6em;overflow-x:auto;font-size:.85em}
</style>"""
