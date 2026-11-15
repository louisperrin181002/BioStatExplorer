# src/plotting.py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

from src.statistics_engine import StatisticsEngine, stars, format_p

# Colour sets offered in the UI (seaborn names work too)
PALETTES = {
    "Modern": ["#2563EB", "#F97316", "#10B981", "#E11D48", "#8B5CF6",
               "#0EA5E9", "#EAB308", "#64748B", "#EC4899", "#14B8A6"],
    "Colour-blind safe": ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                          "#CC79A7", "#56B4E9", "#F0E442", "#000000"],
    "Nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
               "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "Soft": ["#7AA6DC", "#F2A97E", "#8CCB9B", "#E8879C", "#B39DDB",
             "#80CBC4", "#F6D186", "#A3A9B5", "#F4A6C6", "#9ED2E6"],
    "Bold": ["#1D4ED8", "#C2410C", "#047857", "#B91C1C", "#6D28D9",
             "#0369A1", "#A16207", "#334155", "#BE185D", "#0F766E"],
    # for print, and for model-vs-measured plots where the bars should
    # read as one series seen three ways rather than three categories
    "Greyscale": ["#1A1A1A", "#8C8C8C", "#DCDCDC", "#5A5A5A", "#B4B4B4",
                  "#2E2E2E", "#A0A0A0", "#6E6E6E"],
}
PALETTE_NAMES = list(PALETTES) + ["muted", "deep", "pastel", "rocket", "crest"]


def palette_colors(name, n):
    """*n* colours from a named palette (cycled if needed)."""
    n = max(n, 1)
    if name in PALETTES:
        base = PALETTES[name]
        return [base[i % len(base)] for i in range(n)]
    try:
        return [mcolors.to_hex(c) for c in sns.color_palette(name, n_colors=n)]
    except (ValueError, TypeError):
        return palette_colors("Modern", n)



# ---------------------------------------------------------------------------
# The click inspector
#
# Any chart can record where each row of data was drawn; clicking near a
# point then names it and prints its values. Kept as parallel arrays on the
# axes rather than as pickable artists on purpose: seaborn builds its own
# collections and splits them per hue level, so the artist a click lands on
# says little about which row it is, while a position always does.
# ---------------------------------------------------------------------------

#: How close a click has to land, in screen pixels, to count as a hit.
#: Generous enough to forgive a trackpad, tight enough that two neighbours
#: in a dense scatter are still told apart.
PICK_RADIUS_PX = 14.0


def register_points(ax, xs, ys, frame, label_col=None, fields=None,
                    x_slack=None, note=None):
    """Remember that row *i* of *frame* was drawn at (xs[i], ys[i]).

    *x_slack*, in data units, marks a chart whose x position is jittered
    or dodged rather than meaningful — a swarm, a raincloud. The lookup
    then matches on the y value within that much slack sideways, which is
    the only coordinate that identifies the reading.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = min(len(xs), len(ys), len(frame))
    ax._bse_points = {
        "x": xs[:n], "y": ys[:n],
        "frame": frame.iloc[:n] if len(frame) > n else frame,
        "label": label_col, "fields": fields, "x_slack": x_slack,
        "note": note,
    }


def clear_points(ax):
    ax._bse_points = None


def _fmt_value(v):
    """A value as a person would write it, not as repr() renders it."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):d}"
    if isinstance(v, (float, np.floating)):
        a = abs(float(v))
        if a and (a < 1e-3 or a >= 1e5):
            return f"{v:.3g}"
        return f"{float(v):.4g}"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    return str(v)


def point_at(ax, event):
    """What was drawn where the mouse is, or None.

    Returns ``(title, [(field, value), ...], (x, y), note)`` with the
    position in data coordinates, so a caller can put the label beside the
    point rather than beside the cursor.
    """
    reg = getattr(ax, "_bse_points", None)
    if not reg or event is None or event.x is None:
        return None
    xs, ys = reg["x"], reg["y"]
    if not len(xs):
        return None
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return None
    pix = ax.transData.transform(np.column_stack([np.where(ok, xs, 0.0),
                                                  np.where(ok, ys, 0.0)]))
    dx = pix[:, 0] - event.x
    dy = pix[:, 1] - event.y
    slack = reg.get("x_slack")
    if slack:
        p = ax.transData.transform([[0.0, 0.0], [float(slack), 0.0]])
        slack_px = abs(p[1, 0] - p[0, 0])
        d = np.where(ok & (np.abs(dx) <= slack_px), np.abs(dy), np.inf)
    else:
        d = np.where(ok, np.hypot(dx, dy), np.inf)
    i = int(np.argmin(d))
    if not np.isfinite(d[i]) or d[i] > PICK_RADIUS_PX:
        return None
    frame = reg["frame"]
    row = frame.iloc[i]
    label = reg.get("label")
    if label and label in frame.columns:
        title = _fmt_value(row[label])
    else:
        title = f"row {frame.index[i]}"
    fields = reg.get("fields")
    if fields is None:
        fields = [c for c in frame.columns if not str(c).startswith("_")]
    out = [(str(c).lstrip("_"), _fmt_value(row[c]))
           for c in fields if c in frame.columns and c != label]
    return title, out, (float(xs[i]), float(ys[i])), reg.get("note")


class PlottingEngine:
    #: Chart types that compare predictions with measurements. They take
    #: the Metric as the measurement and *predicted_cols* as the models,
    #: so they bypass the usual group-and-test machinery entirely.
    MODEL_CHARTS = ("Model vs Measured", "Model vs Measured (bars)")

    #: Charts that take over the axes completely — they place their own
    #: ticks, labels and legend, and share none of the categorical
    #: group / significance-bracket pipeline the others run through.
    SELF_DRAWN = ("Heatmap", "Volcano Plot", "Dendrogram", "Network Graph",
                  "Stacked Histogram", "Stacked Bar", "Density (overlaid)",
                  "Ridge plot", "Venn diagram", "UpSet plot", "Bubble chart",
                  "Dot plot")

    #: The set charts. Both read the X groups as the sets and the Color-by
    #: column as what is in them, so both need a Color-by column set.
    SET_CHARTS = ("Venn diagram", "UpSet plot")

    #: Mean of the replicates at each X with error bars, drawn the way
    #: growth curves appear in papers: open markers, thin bars, one series
    #: per condition on shared axes, optionally with a fitted curve.
    GROWTH_CHART = "Growth curve (mean ± error)"

    #: What the error bars may show. SD describes the spread of the
    #: replicates; SEM and CI describe how well the mean is pinned down.
    #: Papers usually print SD for growth curves and say so in the legend.
    ERROR_BARS = ("SD", "SEM", "95% CI")

    #: How the error is drawn. A time course read every few minutes has so
    #: many points that capped bars merge into a smear, so a shaded band
    #: around the mean is the readable form — and the one growth curves are
    #: usually published with. MEAN_ONLY draws no error at all, for a clean
    #: figure where the spread is reported elsewhere.
    MEAN_ONLY = "Mean only (no error)"
    ERROR_STYLES = ("Band (continuous)", "Bars", MEAN_ONLY)

    #: What the error is drawn around. Centred on the fitted curve the
    #: figure reads as one smooth curve with the spread wrapped round it;
    #: centred on the mean it stays with the data, wander and all. Only the
    #: centre line moves — the error is the replicates' spread either way,
    #: and the band is clamped to enclose whichever is chosen.
    CENTRE_FIT = "Fitted curve"
    CENTRE_MEAN = "Mean curve"
    ERROR_CENTRES = (CENTRE_FIT, CENTRE_MEAN)

    #: Comparing whole curves means reducing each replicate series to one
    #: number first — you cannot test "the curves differ" without saying in
    #: what respect. Each of these answers a different question: the area
    #: covers the whole time course, the maximum is the yield, the rate is
    #: the speed, and the last value is the endpoint.
    CURVE_MEASURES = ("Area under the curve", "Maximum value",
                      "Fitted rate / EC50", "Value at the last X")
    NO_CURVE_STAT = "None"

    #: The one parameter worth putting in the legend, per model, tried in
    #: order. Everything else stays in the parameter table.
    _HEADLINE = (("r (growth rate, /h)", "r = {:.2f}/h"),
                 ("µmax (/h)", "µmax = {:.2f}/h"),
                 ("EC50", "EC50 = {:.3g}"))

    @staticmethod
    def _headline(params):
        r2 = params.get("R²")
        tail = f", R² = {r2:.3f}" if r2 is not None and np.isfinite(r2) else ""
        for key, fmt in PlottingEngine._HEADLINE:
            v = params.get(key)
            if v is not None and np.isfinite(v):
                return fmt.format(v) + tail
        return tail.lstrip(", ") or "fitted"

    @staticmethod
    def curve_summaries(df, x_col, y_col, hue_col, rep_col, measure,
                        model=None):
        """One number per replicate series, ready for a group comparison.

        Returns a frame with the condition in *hue_col* (or "All") and the
        measure in a column named after it. A series needs at least three
        points; shorter ones are dropped rather than guessed at.
        """
        cols = [c for c in (x_col, y_col, hue_col, rep_col) if c]
        d = df[list(dict.fromkeys(cols))].copy()
        for c in (x_col, y_col):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        if d.empty or not rep_col:
            return pd.DataFrame()
        keys = [c for c in (hue_col, rep_col) if c]
        rows = []
        for key, g in d.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            g = g.sort_values(x_col)
            xs = g[x_col].to_numpy(float)
            ys = g[y_col].to_numpy(float)
            if len(xs) < 3:
                continue
            if measure == "Area under the curve":
                val = float(np.trapezoid(ys, xs)) if hasattr(np, "trapezoid") \
                    else float(np.trapz(ys, xs))
            elif measure == "Maximum value":
                val = float(np.nanmax(ys))
            elif measure == "Value at the last X":
                val = float(ys[-1])
            else:
                try:
                    from src import curves as _curves
                except ImportError:                      # pragma: no cover
                    continue
                fitter = _curves.FITTERS.get(model or _curves.GROWTH,
                                             _curves.fit_growth)
                params, fn = fitter(xs, ys)
                val = next((params[k] for k, _ in PlottingEngine._HEADLINE
                            if np.isfinite(params.get(k, np.nan))), np.nan)
                if fn is None or not np.isfinite(val):
                    continue
            rows.append({(hue_col or "Condition"): (str(key[0]) if hue_col
                                                    else "All"),
                         measure: val})
        return pd.DataFrame(rows)

    @staticmethod
    def _curve_stat_box(ax, df, x_col, y_col, hue_col, rep_col, measure,
                        stat_mode, reference, text_color, max_lines=6,
                        model=None):
        """Compare the conditions on *measure* and print the result on the
        plot. Brackets make no sense over a time axis, so the comparison is
        stated instead of drawn."""
        from src.statistics_engine import StatisticsEngine as SE, format_p
        summ = PlottingEngine.curve_summaries(df, x_col, y_col, hue_col,
                                              rep_col, measure, model)
        gcol = hue_col or "Condition"
        if summ.empty or summ[gcol].nunique() < 2:
            msg = ("Pick a replicate column to compare curves"
                   if summ.empty else
                   "Only one condition — nothing to compare")
            ax.text(0.015, 0.985, msg, transform=ax.transAxes, ha="left",
                    va="top", fontsize=7.5, color=text_color, alpha=0.85)
            return
        order = list(dict.fromkeys(summ[gcol]))
        res = SE.compare(summ, gcol, measure, stat_mode, reference, None,
                         group_order=order)
        n = summ.groupby(gcol)[measure].count()
        head = f"{measure} — {res.test or stat_mode}  (n = " + \
            (f"{n.min()}" if n.min() == n.max() else f"{n.min()}–{n.max()}") + ")"
        lines = [head]
        for c in res.comparisons[:max_lines]:
            star = "✱" if c.significant else "ns"
            lines.append(f"   {c.a} vs {c.b}:  {format_p(c.p)}  {star}")
        extra = len(res.comparisons) - max_lines
        if extra > 0:
            lines.append(f"   +{extra} more — see the Statistics tab")
        ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=7.4, color=text_color,
                linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#CBD5E1", linewidth=0.7, alpha=0.9))

    @staticmethod
    def _spread(values, kind):
        """Half-length of the error bar for one set of replicates."""
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        n = v.size
        if n < 2:
            return 0.0
        if kind == "SD":
            return float(v.std(ddof=1))
        sem = float(stats.sem(v))
        if kind == "95% CI":
            return float(sem * stats.t.ppf(0.975, n - 1))
        return sem

    @staticmethod
    def _band(means, errs, smooth=True):
        """The lower and upper edges of the error band.

        Smoothing a noisy series can pull an edge past the mean it is
        supposed to enclose, which reads as an error — so the edges are
        pushed back out over the mean and re-smoothed a few times, each
        pass with a shorter window, and clamped once at the end. The result
        is a smooth envelope that always contains the mean line, with a
        gentle bulge where a timepoint strays rather than a hard kink.
        """
        means = np.asarray(means, float)
        errs = np.asarray(errs, float)
        if not smooth:
            return means - errs, means + errs
        lo = PlottingEngine._smooth(means - errs)
        hi = PlottingEngine._smooth(means + errs)
        for _ in range(3):
            lo = PlottingEngine._smooth(np.minimum(lo, means), frac=0.05)
            hi = PlottingEngine._smooth(np.maximum(hi, means), frac=0.05)
        return np.minimum(lo, means), np.maximum(hi, means)

    @staticmethod
    def _interp(xs, ys, grid):
        """*ys* resampled onto *grid* as a smooth, shape-preserving curve.

        Monotone cubic (PCHIP) rather than a plain cubic spline: it passes
        through every point without the overshoot that would invent an
        error larger than any replicate ever showed, or a negative one.
        """
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 2:
            return np.full(np.size(grid), ys[ok][0] if ok.any() else np.nan)
        try:
            from scipy.interpolate import PchipInterpolator
            return PchipInterpolator(xs[ok], ys[ok])(grid)
        except Exception:                            # pragma: no cover
            return np.interp(grid, xs[ok], ys[ok])

    @staticmethod
    def _dense_band(xs, means, errs, grid, fn=None):
        """Band edges along *grid* instead of along the timepoints.

        A moving average cannot smooth four points, so on a plate read with
        only a handful of timepoints the ribbon stayed a chain of straight
        segments while the fitted curve beside it was drawn on 300 — the
        band looked nothing like the mean it belonged to. Resampling the
        spread onto the curve's own grid fixes that at the source: the
        error is still exactly the replicate spread at each timepoint, and
        between them it varies as smoothly as the curve does.

        Centred on the fit (*fn*), ``centre ± error`` encloses it already.
        Centred on the mean, the drawn mean line is still straight, so the
        edges are clamped to it — the band can bulge away from that line
        but never cross it.
        """
        errs_d = np.clip(PlottingEngine._interp(xs, errs, grid), 0.0, None)
        centre_d = (np.asarray(fn(grid), float) if fn is not None
                    else PlottingEngine._interp(xs, means, grid))
        lo, hi = centre_d - errs_d, centre_d + errs_d
        if fn is None:
            line = np.interp(grid, xs, means)
            lo, hi = np.minimum(lo, line), np.maximum(hi, line)
        return lo, hi

    @staticmethod
    def _smooth(y, frac=0.12):
        """A short centred moving average, for drawing only.

        Used on the edges of the error band so a plate read every few
        minutes gives a smooth ribbon instead of a frayed one. The window
        shrinks at the two ends rather than padding, so the curve starts
        and finishes on the real values instead of being dragged flat.
        Deliberately a local average and not the fitted model: the band
        has to stay on the mean curve even where the model does not.
        """
        y = np.asarray(y, float)
        n = y.size
        win = int(max(3, round(n * frac)))
        if n < 5 or win < 3:
            return y
        win += 1 - win % 2                       # odd, so it stays centred
        half = win // 2
        out = np.empty(n, float)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            seg = y[lo:hi]
            seg = seg[np.isfinite(seg)]
            out[i] = seg.mean() if seg.size else y[i]
        return out

    @staticmethod
    def _draw_growth(df, x_col, y_col, ax, hue_col, palette, text_color,
                     grid_color, bw_mode=False, hue_levels=None,
                     error_bars="SD", fit=True, log_scale=False,
                     error_style="Band (continuous)", stat_box=False,
                     model=None, log_x=False, smooth_error=True,
                     error_centre=None):
        """Mean ± error at each X, one series per condition.

        *fit* overlays one logistic per condition, fitted to the mean curve
        — the same model and the same axes for every condition, so the
        curves can be read against each other rather than each on its own
        terms.

        *smooth_error* draws the band as a pair of smooth curves around the
        mean curve, instead of a ribbon that frays at every timepoint. It
        changes nothing about the error itself, which is always the spread
        of the replicates at that X — only how its envelope is drawn.
        """
        cols = [x_col, y_col] + ([hue_col] if hue_col else [])
        d = df[cols].copy()
        for c in (x_col, y_col):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        if d.empty:
            ax.text(0.5, 0.5, f"'{x_col}' and '{y_col}' must both be "
                    "numeric — a growth curve needs a time axis",
                    ha="center", va="center", transform=ax.transAxes,
                    color=text_color, fontsize=10, wrap=True)
            return

        levels = (hue_levels or list(d[hue_col].dropna().unique())) \
            if hue_col else [y_col]
        colors = (palette if len(palette) >= len(levels)
                  else sns.color_palette("muted", n_colors=max(len(levels), 1)))
        markers = ["o", "s", "^", "D", "v", "P"]

        # The fit follows whichever model is chosen, so one chart covers
        # growth curves and dose-response alike.
        try:
            from src import curves as _curves
            fitter = _curves.FITTERS.get(model or _curves.GROWTH)
        except ImportError:                              # pragma: no cover
            _curves, fitter = None, None

        handles, n_seen = [], set()
        for i, lev in enumerate(levels):
            sub = d[d[hue_col] == lev] if hue_col else d
            g = sub.groupby(x_col)[y_col]
            xs = np.asarray(sorted(g.groups), float)
            means = g.mean().reindex(xs).to_numpy(float)
            errs = np.array([PlottingEngine._spread(g.get_group(x), error_bars)
                             for x in xs])
            n_seen.update(g.count().unique().tolist())
            colour = colors[i % len(colors)]
            marker = markers[i % len(markers)] if bw_mode else "o"

            fitted = None
            if fit and fitter is not None and len(xs) >= 4:
                params, fn = fitter(xs, means)
                # Not every model reports R² — the max-slope fit has none —
                # so a usable fitting function is what counts.
                if fn is not None:
                    fitted = (params, fn)

            # What the error is drawn around. On the fitted curve it reads
            # as one smooth curve with the replicate spread wrapped round
            # it; on the mean it stays with the data, wander and all. The
            # error is the same either way — only its centre line moves —
            # and whichever is chosen, the band is clamped to enclose it.
            centre = means
            on_fit = False
            if fitted and error_centre in (None, PlottingEngine.CENTRE_FIT):
                modelled = np.asarray(fitted[1](xs), dtype=float)
                if modelled.shape == means.shape and np.all(
                        np.isfinite(modelled)):
                    centre, on_fit = modelled, True

            # The grid the fitted curve is drawn on, and now the band too,
            # so the two are the same shape rather than one curve beside a
            # chain of segments.
            grid = (np.logspace(np.log10(xs[xs > 0].min()),
                                np.log10(xs.max()), 300)
                    if log_x and np.any(xs > 0)
                    else np.linspace(xs.min(), xs.max(), 300))

            def edges():
                """Lower and upper edge of the error, and their X."""
                if not smooth_error:
                    lo, hi = PlottingEngine._band(centre, errs, False)
                    return xs, lo, hi
                lo, hi = PlottingEngine._dense_band(
                    xs, means, errs, grid,
                    fn=fitted[1] if on_fit else None)
                return grid, lo, hi

            if error_style == "Bars":
                # Open markers with capped bars. A thin line joins them
                # only when there is no fit to carry the eye instead.
                handle = ax.errorbar(
                    xs, centre, yerr=errs, fmt=marker, markersize=4.5,
                    markerfacecolor="none", markeredgecolor=colour,
                    markeredgewidth=1.1, ecolor=colour, elinewidth=0.9,
                    capsize=2.0, capthick=0.9,
                    linestyle="none" if fitted else "-",
                    color=colour, linewidth=1.0, zorder=3)
                if smooth_error:
                    # The caps alone show the spread one timepoint at a
                    # time; these two curves trace where it is going, which
                    # is what you want when the bars are dense enough to
                    # read as a smear.
                    ex, lo, hi = edges()
                    for edge in (lo, hi):
                        ax.plot(ex, edge, color=colour, linewidth=0.9,
                                alpha=0.8, zorder=2)
            else:
                # Continuous band: the mean as a line, the error as a
                # shaded ribbon around it. Clipped at zero on a log axis,
                # where mean − SD can fall below the scale. With MEAN_ONLY
                # the ribbon is left out and the mean line carries the
                # figure on its own, so it is drawn at full weight unless a
                # fitted curve is there to lead the eye instead.
                if error_style != PlottingEngine.MEAN_ONLY:
                    # The band's edges are mean ± error at every timepoint,
                    # so on a noisy plate read they jitter as much as the
                    # data does and the ribbon looks frayed. Smoothing runs
                    # a short centred average along both the mean and the
                    # spread, which rounds the edges into curves while
                    # keeping the band on the mean curve — it is the drawn
                    # envelope that is smoothed, never the data or the
                    # statistics.
                    ex, lo, hi = edges()
                    if log_scale:
                        lo = np.clip(lo, np.nanmin(means[means > 0]) / 10.0
                                     if np.any(means > 0) else 1e-9, None)
                    ax.fill_between(ex, lo, hi, color=colour, alpha=0.22,
                                    linewidth=0, zorder=2)
                    # Outline the ribbon. A fill alone has no definite edge,
                    # so where several conditions overlap you cannot tell
                    # whose band you are looking at; the drawn edge is also
                    # what makes the published form of this figure legible
                    # in print, where a 22% tint nearly disappears.
                    for edge in (lo, hi):
                        ax.plot(ex, edge, color=colour, linewidth=0.8,
                                alpha=0.9, zorder=2)
                mean_only = error_style == PlottingEngine.MEAN_ONLY
                handle, = ax.plot(
                    xs, means, color=colour,
                    linewidth=1.8 if (mean_only and not fitted) else 0.9,
                    zorder=3,
                    linestyle="--" if fitted else "-",
                    alpha=0.75 if fitted else 1.0,
                    marker=marker if bw_mode else "None", markersize=3.5,
                    markerfacecolor="none", markevery=max(1, len(xs) // 12))
            if fitted:
                params, fn = fitted
                ax.plot(grid, fn(grid), color=colour, linewidth=1.8,
                        zorder=4)
                label = f"{lev}  ({PlottingEngine._headline(params)})"
            else:
                label = str(lev)
            handles.append((handle, label))

        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        if log_x:
            ax.set_xscale("log")
        else:
            ax.set_xlim(left=float(d[x_col].min()))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        reps = f"n = {min(n_seen)}" if len(n_seen) == 1 else \
            f"n = {min(n_seen)}–{max(n_seen)}"
        # Name the curve the error is wrapped around, so the figure cannot
        # be read as spread about the mean when it is spread about the fit.
        on_fit = (fit and fitter is not None
                  and error_centre in (None, PlottingEngine.CENTRE_FIT))
        centred = "fit" if on_fit else "mean"
        if error_style == PlottingEngine.MEAN_ONLY:
            note = f"mean of replicates, {reps}"
        elif error_style == "Bars":
            note = (f"{centred} ± {error_bars} (smoothed), {reps}"
                    if smooth_error else f"{centred} ± {error_bars}, {reps}")
        else:
            # Say when the envelope has been smoothed, so nobody reads the
            # ribbon's edge as the error at that exact timepoint.
            note = (f"{centred} ± {error_bars} (smoothed band), {reps}"
                    if smooth_error
                    else f"{centred} ± {error_bars} (shaded), {reps}")
        # Under the axes, where a figure caption belongs: a growth curve
        # fills every corner at some point, and inside the axes this note
        # ended up printing through the legend. Both the window (via
        # tight_layout) and every export (via bbox_inches="tight") grow to
        # include it, so it is never clipped.
        ax.text(1.0, -0.22, note, transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color=text_color, alpha=0.8)
        # "best" rather than a fixed corner: a growth curve fills whichever
        # corner you pick, and the fit labels make the legend wide.
        legend = ax.legend([h for h, _ in handles], [t for _, t in handles],
                           title=hue_col if hue_col else None, fontsize=8.5,
                           title_fontsize=9, frameon=False,
                           loc="lower right" if stat_box else "best")
        if legend:
            for text_obj, (_, lbl) in zip(legend.get_texts(), handles):
                text_obj.set_picker(True)
                text_obj.set_url("group_color:" + lbl.split("  (")[0])

    @staticmethod
    def _draw_model_check(ax, df, label_col, measured_col, predicted_cols,
                          plot_type, pc, palette, log_scale, custom_title,
                          text_color, fold_band=None):
        """Measured against one or more model columns."""
        from src import model_check
        cols = [c for c in predicted_cols if c and c != measured_col]
        if not cols:
            ax.text(0.5, 0.5, "Choose a Predicted column\n"
                    "(the Metric is the measured value)", ha="center",
                    va="center", transform=ax.transAxes, color=text_color)
            ax.axis("off")
            return
        table, dropped = model_check.prepare(df, label_col, measured_col,
                                             cols)
        if table.empty:
            ax.text(0.5, 0.5, "Nothing to compare — a log comparison needs\n"
                    "positive values in both columns.", ha="center",
                    va="center", transform=ax.transAxes, color=text_color)
            ax.axis("off")
            return
        colors = palette_colors(palette, len(cols) + 1)[1:]
        if plot_type == "Model vs Measured (bars)":
            model_check.draw_bars(ax, table, measured_col, cols, colors,
                                  pc=pc, label_col=label_col,
                                  log=log_scale or True,
                                  measured_color=pc.get("line", "#334155"))
        else:
            model_check.draw_scatter(ax, table, measured_col, cols, colors,
                                     pc=pc, label_col=label_col,
                                     log=log_scale or True,
                                     band=fold_band)
        if plot_type != "Model vs Measured (bars)":
            st = model_check.summarise(table, measured_col, cols)
            xs, ys, recs = [], [], []
            meas = table[measured_col].to_numpy(float)
            for c in cols:
                if c not in st:
                    continue
                scaled = table[c].to_numpy(float) * st[c]["scale"]
                xs.extend(meas.tolist())
                ys.extend(scaled.tolist())
                part = pd.DataFrame({
                    "model": c,
                    measured_col: meas,
                    "predicted (scaled)": scaled,
                    "fold off": np.where(meas > 0, scaled / meas, np.nan)})
                if label_col and label_col in table.columns:
                    part.insert(0, label_col, table[label_col].to_numpy())
                recs.append(part)
            if recs:
                register_points(ax, xs, ys, pd.concat(recs, ignore_index=True),
                                label_col=label_col if (
                                    label_col and label_col in table.columns)
                                else None)
        title = custom_title or f"{measured_col} — measured against model"
        if dropped:
            title += f"  ({dropped} row(s) dropped)"
        ax.set_title(title, color=text_color, fontsize=11)

    @staticmethod
    def generate_plot(df, plot_type, x_col, y_col, ax, dark_mode=False, palette="Modern",
                       stat_mode="Auto (Prism Mode)", hue_col=None, log_scale=False,
                       reference=None, pairing_col=None, show_pvalue=False,
                       y_min=None, y_max=None, group_colors=None, bw_mode=False,
                       hist_bins=20, custom_title=None, custom_xlabel=None,
                       custom_ylabel=None, group_order=None, show_ns=False,
                       hue_order=None, predicted_cols=None,
                       error_bars="SD", fit_curve=True,
                       error_style="Band (continuous)",
                       curve_measure="None", replicate_col=None,
                       curve_model=None, log_x=False, smooth_error=True,
                       error_centre=None, fold_band=None, size_col=None,
                       dot_sort=True):
        """Generates plots while respecting active application theme states and overlays stars."""
        ax.clear()
        # Whatever the last chart recorded for the click inspector is gone
        # with the artists it described, and a stale position table would
        # name the wrong row.
        clear_points(ax)

        from src.theme import plot_colors
        pc = plot_colors(dark_mode)
        plt.style.use("dark_background" if dark_mode else "default")
        plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica",
                                             "Arial", "DejaVu Sans"],
                             "axes.titleweight": "semibold",
                             "axes.labelweight": "medium"})
        ax.set_facecolor(pc["bg"])
        ax.figure.set_facecolor(pc["bg"])
        text_color = pc["text"]
        grid_color = pc["grid"]
        line_color = pc["line"]

        if df is None or df.empty or not x_col or not y_col:
            ax.text(0.5, 0.5, "Select X and Y variables to plot", ha='center', va='center', transform=ax.transAxes, color=text_color)
            return

        if plot_type in PlottingEngine.MODEL_CHARTS:
            PlottingEngine._draw_model_check(
                ax, df, x_col, y_col, predicted_cols or [], plot_type, pc,
                palette, log_scale, custom_title, text_color,
                fold_band=fold_band)
            PlottingEngine._style_spines(ax, text_color)
            return
        if x_col == y_col:
            ax.text(0.5, 0.5, f"'{x_col}' is both the X axis and the Metric —\n"
                    "choose a different Metric", ha='center', va='center',
                    transform=ax.transAxes, color=text_color)
            return

        # light horizontal guide lines behind the data
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        ax.xaxis.grid(False)

        # Keep an explicit categorical order for alignment
        unique_groups = list(df[x_col].dropna().unique())
        if len(unique_groups) == 0:
            return

        if group_order:
            # Manual reordering (from right-clicking a group's label/legend
            # entry on the plot). Groups not mentioned in group_order keep
            # their natural (first-seen) relative order, appended at the end
            # - so a newly-appearing group (new file loaded, filter changed)
            # doesn't just vanish from the axis.
            ordered = [g for g in group_order if g in unique_groups]
            remaining = [g for g in unique_groups if g not in ordered]
            unique_groups = ordered + remaining

        hue_col = hue_col if (hue_col and hue_col in df.columns and hue_col != x_col) else None
        legend_needed = hue_col is not None
        is_line = (plot_type == "Line")

        # --- Resolve the actual color for each "color category" (hue levels if a
        # Group/Color By column is set, otherwise the x-axis groups themselves).
        # Any category the user has manually recolored (via clicking the plot)
        # keeps that color; everything else falls back to the chosen palette. ---
        group_colors = group_colors or {}
        if hue_col:
            color_categories = list(df[hue_col].dropna().unique())
            if hue_order:     # order chosen in the Groups panel
                first = [h for h in color_categories
                         if str(h) in [str(o) for o in hue_order]]
                first.sort(key=lambda h: [str(o) for o in hue_order].index(str(h)))
                color_categories = first + [h for h in color_categories
                                            if h not in first]
        elif plot_type in ("Line", PlottingEngine.GROWTH_CHART):
            # A single line (no Group/Color-By column) has no natural set of
            # "categories" to click-recolor - key it off y_col itself so the
            # curve is still individually addressable/recolorable.
            color_categories = [y_col]
        else:
            color_categories = unique_groups

        if bw_mode:
            # Black & white mode: replace hue entirely with grayscale shades
            # (still one shade per category so groups stay distinguishable),
            # ignoring the seaborn palette and any manual color picks.
            n = max(len(color_categories), 1)
            shades = np.linspace(0.75, 0.05, n) if n > 1 else np.array([0.2])
            default_colors = [(s, s, s) for s in shades]
        else:
            default_colors = palette_colors(palette, len(color_categories))
        resolved_palette = [group_colors.get(str(cat), default_colors[i]) if not bw_mode
                             else default_colors[i]
                             for i, cat in enumerate(color_categories)]

        if plot_type == PlottingEngine.GROWTH_CHART:
            PlottingEngine._draw_growth(
                df, x_col, y_col, ax, hue_col, resolved_palette, text_color,
                grid_color, bw_mode=bw_mode,
                hue_levels=color_categories if hue_col else None,
                error_bars=error_bars, fit=fit_curve, log_scale=log_scale,
                model=curve_model, log_x=log_x, smooth_error=smooth_error,
                error_centre=error_centre,
                error_style=error_style, stat_box=(
                    stat_mode != "None" and curve_measure not in (
                        None, PlottingEngine.NO_CURVE_STAT)))
            if stat_mode != "None" and curve_measure not in (
                    None, PlottingEngine.NO_CURVE_STAT):
                PlottingEngine._curve_stat_box(
                    ax, df, x_col, y_col, hue_col, replicate_col,
                    curve_measure, stat_mode, reference, text_color,
                    model=curve_model)
            PlottingEngine._register_curve_points(
                ax, df, x_col, y_col, hue_col, error_bars)
            ax.set_title(custom_title or f"{y_col} over {x_col}",
                         color=text_color, fontsize=11)
            if custom_xlabel:
                ax.set_xlabel(custom_xlabel)
            if custom_ylabel:
                ax.set_ylabel(custom_ylabel)
            if y_min is not None or y_max is not None:
                ax.set_ylim(bottom=y_min, top=y_max)
            PlottingEngine._style_spines(ax, text_color)
            return

        # --- Special plot types: these don't share the categorical group /
        # significance-overlay / xtick-rotation pipeline below, so they draw
        # themselves fully and return early. ---
        if plot_type in PlottingEngine.SELF_DRAWN:
            if plot_type == "Heatmap":
                PlottingEngine._draw_heatmap(df, x_col, y_col, hue_col, ax, palette, text_color)
            elif plot_type == "Volcano Plot":
                PlottingEngine._draw_volcano(df, x_col, y_col, hue_col, ax, text_color, grid_color)
            elif plot_type == "Dendrogram":
                PlottingEngine._draw_dendrogram(df, x_col, y_col, ax, palette, text_color)
            elif plot_type == "Network Graph":
                PlottingEngine._draw_network(df, x_col, y_col, ax, palette, text_color)
            elif plot_type == "Stacked Histogram":
                PlottingEngine._draw_stacked_histogram(df, x_col, y_col, hue_col, ax, resolved_palette,
                                                         text_color, bw_mode, hist_bins)
            elif plot_type == "Stacked Bar":
                PlottingEngine._draw_stacked_bar(df, x_col, y_col, hue_col, ax, resolved_palette,
                                                   text_color, unique_groups, bw_mode, group_colors)
            elif plot_type == "Density (overlaid)":
                PlottingEngine._draw_density(
                    df, x_col, y_col, hue_col, ax, resolved_palette,
                    text_color, grid_color, color_categories,
                    log_scale=log_scale)
            elif plot_type == "Ridge plot":
                PlottingEngine._draw_ridge(
                    df, x_col, y_col, hue_col, ax, resolved_palette,
                    text_color, grid_color, color_categories,
                    log_scale=log_scale)
            elif plot_type == "Venn diagram":
                PlottingEngine._draw_venn(df, x_col, y_col, hue_col, ax,
                                          resolved_palette, text_color)
            elif plot_type == "UpSet plot":
                PlottingEngine._draw_upset(df, x_col, y_col, hue_col, ax,
                                           resolved_palette, text_color,
                                           grid_color)
            elif plot_type == "Bubble chart":
                PlottingEngine._draw_bubble(
                    df, x_col, y_col, hue_col, size_col, ax,
                    resolved_palette, text_color, grid_color,
                    color_categories if hue_col else None,
                    log_scale=log_scale)
            elif plot_type == "Dot plot":
                PlottingEngine._draw_dotplot(
                    df, x_col, y_col, hue_col, ax, resolved_palette,
                    text_color, grid_color, unique_groups,
                    color_categories if hue_col else None,
                    error_bars=error_bars, log_scale=log_scale,
                    sort_by_value=dot_sort)
            # A custom title is offered on every chart, so the ones that
            # set their own still have to honour it — and stay clickable,
            # which is how it gets typed in the first place.
            if custom_title:
                ax.set_title(custom_title, color=text_color)
            t = ax.title
            if t.get_text():
                t.set_picker(True)
                t.set_url(f"edit_title:{y_col}")
            return

        hue_order = color_categories

        if plot_type == "Bar":
            PlottingEngine._draw_bar(df, x_col, y_col, ax, unique_groups, hue_col, resolved_palette, dark_mode, text_color, bw_mode,
                                     hue_levels=color_categories if hue_col else None)
        elif plot_type == "Line":
            PlottingEngine._draw_line(df, x_col, y_col, ax, hue_col, resolved_palette, text_color, bw_mode,
                                      error_bars=error_bars,
                                      hue_levels=color_categories if hue_col else None)
        elif plot_type == "Boxplot":
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                        palette=resolved_palette, width=0.5, order=unique_groups, hue_order=hue_order,
                        legend=legend_needed, dodge=bool(hue_col))
        elif plot_type == "Violin":
            sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                            palette=resolved_palette, inner="quartile", order=unique_groups, hue_order=hue_order,
                            legend=legend_needed, dodge=bool(hue_col))
        elif plot_type == "Scatter":
            # Color by hue_col when set, otherwise fall back to x_col so each
            # category still gets its own color (matching Boxplot/Violin/Beeswarm)
            # instead of collapsing every point to one flat blue.
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                             alpha=0.8, s=60, palette=resolved_palette, hue_order=hue_order, legend=legend_needed)
        elif plot_type == "Beeswarm":
            sns.swarmplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                          palette=resolved_palette, size=5, order=unique_groups, hue_order=hue_order,
                          legend=legend_needed, dodge=bool(hue_col))
        elif plot_type == "Box + Swarm":
            sns.boxplot(data=df, x=x_col, y=y_col, color=("#2D2D2D" if dark_mode else "white"),
                        ax=ax, width=0.5, fliersize=0, order=unique_groups)
            # Same fix as Scatter: fall back to x_col for hue so dots are
            # colored by group even when no explicit hue_col is chosen.
            sns.swarmplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                          palette=resolved_palette, size=4, alpha=0.6, order=unique_groups, hue_order=hue_order,
                          legend=legend_needed, dodge=bool(hue_col))
        elif plot_type == "Raincloud":
            PlottingEngine._draw_raincloud(
                df, x_col, y_col, ax, unique_groups, hue_col,
                resolved_palette, text_color,
                hue_levels=color_categories if hue_col else None,
                bw_mode=bw_mode)

        # Where every row landed, for the click inspector. Done once here
        # rather than inside each branch: on a categorical chart the x
        # position is the group's slot and the y position is the reading,
        # whichever of box, violin, swarm or bar was drawn over it.
        if plot_type in ("Boxplot", "Violin", "Beeswarm", "Box + Swarm",
                         "Bar", "Scatter"):
            PlottingEngine._register_categorical(
                ax, df, x_col, y_col, hue_col, unique_groups, plot_type)
        elif plot_type == "Line":
            PlottingEngine._register_curve_points(
                ax, df, x_col, y_col, hue_col, error_bars)

        if log_scale:
            ax.set_yscale("log")
            lo_hi = pd.to_numeric(df[y_col], errors="coerce")
            lo_hi = lo_hi[lo_hi > 0]
            if len(lo_hi) and lo_hi.max() / lo_hi.min() < 100:
                # under two decades the default log ticks have no labels
                from matplotlib.ticker import FuncFormatter, LogLocator
                fmt = FuncFormatter(lambda v, _: f"{v:g}")
                ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(fmt)
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))

        # --- STATISTICAL OVERLAY CORE (categorical charts only — a continuous
        # line plot compares curves, not discrete groups, so it's skipped) ---
        stat_on = stat_mode != "None"
        can_stat = (not is_line and len(unique_groups) >= 2
                    and plot_type in PlottingEngine.STAT_CHARTS)
        if stat_on and not can_stat:
            # The commonest reason the overlay "does nothing": the chart
            # type has no brackets, or there is only one group to compare.
            if len(unique_groups) < 2:
                PlottingEngine._stat_note(
                    ax, "Only one group on the X axis — nothing to compare",
                    line_color)
            elif plot_type not in PlottingEngine.STAT_CHARTS:
                PlottingEngine._stat_note(
                    ax, f"No significance brackets on a {plot_type} graph — "
                        f"use {', '.join(PlottingEngine.STAT_CHARTS[:-1])} "
                        f"or {PlottingEngine.STAT_CHARTS[-1]}", line_color)
        if stat_on and can_stat:
            bar_tops = None
            if plot_type == "Bar":
                # brackets must clear the error bars, not just the points
                keys = [x_col] + ([hue_col] if hue_col else [])
                yv = pd.to_numeric(df[y_col], errors="coerce")
                g = yv.groupby([df[k] for k in keys])
                tops = (g.mean() + g.sem().fillna(0)).dropna()
                if hue_col:
                    tops = tops.groupby(level=0).max()
                bar_tops = {str(k): float(v) for k, v in tops.items()}
            drawn = PlottingEngine.annotate_significance(
                df, x_col, y_col, unique_groups, ax, stat_mode, line_color,
                reference=reference, pairing_col=pairing_col,
                log_scale=log_scale, show_pvalue=show_pvalue,
                hue_col=hue_col, hue_order=[str(h) for h in hue_order],
                dodge_width=PlottingEngine.DODGE_WIDTH.get(plot_type, 0.8),
                show_ns=show_ns, bar_tops=bar_tops)
            if not drawn:
                # Distinguish "the test ran and found nothing" from "there
                # was never enough data to test" — they look identical on
                # the plot otherwise, and only one of them is a result.
                counts = pd.to_numeric(df[y_col], errors="coerce").groupby(
                    df[x_col].astype(str)).count()
                if (counts >= 2).sum() < 2:
                    PlottingEngine._stat_note(
                        ax, "Not enough replicates to test — a comparison "
                            "needs at least 2 values in each of 2 groups",
                        line_color)
                else:
                    PlottingEngine._stat_note(
                        ax, "No significant differences"
                            + ("" if show_ns else " — tick 'Show "
                               "non-significant (ns)' to see every bracket"),
                        line_color)

        numeric_x = plot_type == "Scatter" and pd.api.types.is_numeric_dtype(df[x_col])
        if not is_line and not numeric_x:
            # Rotate labels 45 degrees, shrink font to size 10, and right-align them
            ax.set_xticks(range(len(unique_groups)))
            ax.set_xticklabels(
                [str(g) for g in unique_groups],
                rotation=45,
                horizontalalignment='right',
                fontsize=10
            )
            if not hue_col:
                # Make each x-tick label clickable so the UI can offer
                # "click a group's label to recolor it" (only meaningful when
                # colors are assigned per x-group, i.e. no separate hue column).
                for label in ax.get_xticklabels():
                    label.set_picker(True)
                    label.set_url("group_color:" + label.get_text())

        # Clean axes: left and bottom lines only
        PlottingEngine._style_spines(ax, text_color)

        # Title / axis labels: each is picker-enabled with a distinct URL so
        # the UI can catch a click on it (same pick_event pathway used for
        # group-label recoloring) and swap in a user-typed override. Falls
        # back to the auto-generated text until the user has customized it.
        default_title = f"{y_col} vs {x_col}" + (f" (by {hue_col})" if hue_col else "")
        title_obj = ax.set_title(custom_title if custom_title else default_title,
                                  fontsize=13, pad=14, color=text_color)
        title_obj.set_picker(True)
        title_obj.set_url(f"edit_title:{y_col}")

        xlabel_obj = ax.set_xlabel(custom_xlabel if custom_xlabel else x_col,
                                    fontsize=11, color=text_color)
        xlabel_obj.set_picker(True)
        xlabel_obj.set_url("edit_xlabel")

        ylabel_obj = ax.set_ylabel(custom_ylabel if custom_ylabel else y_col,
                                    fontsize=11, color=text_color)
        ylabel_obj.set_picker(True)
        ylabel_obj.set_url(f"edit_ylabel:{y_col}")

        ax.tick_params(colors=text_color)
        if legend_needed and not is_line:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(title=hue_col, fontsize=8.5, title_fontsize=9, frameon=False)
                # Make each legend label clickable too, so hue-colored plots
                # (color assigned by Group/Color-By) can also be recolored by
                # clicking their legend entry instead of an x-tick label.
                for text_obj, lbl in zip(legend.get_texts(), labels):
                    text_obj.set_picker(True)
                    text_obj.set_url("group_color:" + lbl)

        # Manual y-axis range: applied last so it always wins over the
        # significance overlay's auto-computed headroom, whichever the
        # user actually set (only overriding the side(s) they specified).
        if not log_scale and (y_min is not None or y_max is not None):
            cur_bottom, cur_top = ax.get_ylim()
            ax.set_ylim(bottom=y_min if y_min is not None else cur_bottom,
                        top=y_max if y_max is not None else cur_top)

    @staticmethod
    def _register_curve_points(ax, df, x_col, y_col, hue_col, error_bars="SD"):
        """The drawn markers of a curve chart: one mean per X per series.

        A curve chart averages its replicates, so the point under the
        cursor is not a row of the table — it is a summary. The label says
        so by naming the number of replicates behind it alongside the
        mean, rather than pretending to show one reading.
        """
        cols = [c for c in dict.fromkeys([x_col, y_col] +
                                         ([hue_col] if hue_col else []))]
        d = df[cols].copy()
        for c in (x_col, y_col):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        if d.empty:
            return
        keys = [x_col] + ([hue_col] if hue_col else [])
        rows = []
        for key, g in d.groupby(keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            v = g[y_col].to_numpy(float)
            err = PlottingEngine._spread(v, error_bars)
            rec = {x_col: float(key[0]), "n": len(v),
                   f"mean {y_col}": float(np.mean(v)),
                   error_bars: float(err) if np.isfinite(err) else 0.0}
            if hue_col:
                rec[hue_col] = key[1]
            rows.append(rec)
        if not rows:
            return
        summary = pd.DataFrame(rows)
        register_points(ax, summary[x_col].to_numpy(float),
                        summary[f"mean {y_col}"].to_numpy(float),
                        summary, label_col=hue_col if hue_col else None,
                        note="mean of the replicates, not one reading")

    @staticmethod
    def _inspect_fields(frame, y_col, label_col=None):
        """Which of a row's columns the click label prints, and in what
        order: the measurement being plotted first, then the rest as the
        sheet has them. The label itself is left out — it is the heading.
        """
        rest = [c for c in frame.columns
                if c not in (y_col, label_col) and not str(c).startswith("_")]
        return ([y_col] if y_col in frame.columns else []) + rest

    @staticmethod
    def _register_categorical(ax, df, x_col, y_col, hue_col, groups,
                              plot_type):
        """Record each row's drawn position for the click inspector.

        For a scatter on a numeric X the position is simply the pair of
        values. For everything else the X axis is categorical, so a row
        sits in its group's slot — possibly jittered by a swarm or dodged
        by a colour — and only the y value identifies which reading it is;
        *x_slack* tells the lookup to match on y within the slot.
        """
        d = df.copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        numeric_x = (plot_type == "Scatter"
                     and pd.api.types.is_numeric_dtype(df[x_col]))
        if numeric_x:
            d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
            d = d.dropna(subset=[x_col, y_col])
            if d.empty:
                return
            d = d.reset_index(drop=True)
            register_points(ax, d[x_col].to_numpy(float),
                            d[y_col].to_numpy(float), d,
                            label_col=hue_col if hue_col else None,
                            fields=PlottingEngine._inspect_fields(
                                d, y_col, hue_col))
            return
        slots = {str(g): i for i, g in enumerate(groups)}
        d = d.dropna(subset=[y_col])
        d = d[d[x_col].astype(str).isin(slots)]
        if d.empty:
            return
        xs = d[x_col].astype(str).map(slots).to_numpy(float)
        d = d.reset_index(drop=True)
        register_points(ax, xs, d[y_col].to_numpy(float), d,
                        label_col=hue_col or x_col,
                        fields=PlottingEngine._inspect_fields(
                            d, y_col, hue_col or x_col),
                        x_slack=0.5)

    @staticmethod
    def _style_spines(ax, color):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(color)
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(length=3.5, width=0.8, colors=color)

    @staticmethod
    def _draw_bar(df, x_col, y_col, ax, unique_groups, hue_col, palette, dark_mode, text_color, bw_mode=False,
                  hue_levels=None):
        """Grouped bar chart of mean +/- SEM, matching the classic Prism-style bar plot."""
        df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna(subset=[x_col, y_col])
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[y_col])

        # In B&W mode, layer a distinct hatch pattern per category on top of
        # the grayscale shades so bars stay distinguishable without color.
        hatch_cycle = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]

        if hue_col:
            hue_levels = hue_levels or list(df_clean[hue_col].dropna().unique())
            n_hues = len(hue_levels)
            bar_width = 0.8 / max(n_hues, 1)
            # `palette` here is already the resolved per-category color list built
            # in generate_plot (defaults + any manual click-to-recolor overrides),
            # in the same order as hue_levels — use it directly rather than
            # re-deriving colors, so overrides land on the right bar.
            colors = palette if len(palette) >= n_hues else sns.color_palette("muted", n_colors=max(n_hues, 1))

            for h_idx, h_level in enumerate(hue_levels):
                sub = df_clean[df_clean[hue_col] == h_level]
                means, sems, xs = [], [], []
                for g_idx, g in enumerate(unique_groups):
                    vals = sub[sub[x_col] == g][y_col].values
                    means.append(np.mean(vals) if len(vals) else 0.0)
                    sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)
                    offset = (h_idx - (n_hues - 1) / 2.0) * bar_width
                    xs.append(g_idx + offset)
                bars = ax.bar(xs, means, width=bar_width * 0.92, yerr=sems, capsize=3,
                       color=colors[h_idx], label=str(h_level), linewidth=0,
                       error_kw={"ecolor": text_color, "elinewidth": 1, "capthick": 1})
                if bw_mode:
                    for b in bars:
                        b.set_hatch(hatch_cycle[h_idx % len(hatch_cycle)])
        else:
            # No grouping column: give each x-axis category its own color,
            # the same way Boxplot/Violin do (rather than one flat color).
            # `palette` is the resolved per-category list from generate_plot.
            colors = palette if len(palette) >= len(unique_groups) else sns.color_palette("muted", n_colors=max(len(unique_groups), 1))
            means, sems = [], []
            for g in unique_groups:
                vals = df_clean[df_clean[x_col] == g][y_col].values
                means.append(np.mean(vals) if len(vals) else 0.0)
                sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)
            bars = ax.bar(range(len(unique_groups)), means, width=0.62, yerr=sems, capsize=4,
                   color=colors, linewidth=0,
                   error_kw={"ecolor": text_color, "elinewidth": 1, "capthick": 1})
            if bw_mode:
                for g_idx, b in enumerate(bars):
                    b.set_hatch(hatch_cycle[g_idx % len(hatch_cycle)])

        ax.set_xticks(range(len(unique_groups)))
        if (df_clean[y_col] < 0).any():
            ax.axhline(0, color=text_color, linewidth=0.8, zorder=1)

    @staticmethod
    def _draw_line(df, x_col, y_col, ax, hue_col, palette, text_color, bw_mode=False, hue_levels=None,
                   error_bars="SEM"):
        """
        Connected mean +/- error line plot for continuous X data (e.g. a time
        course or substrate-concentration curve in an enzyme kinetics assay).
        One line per hue level if a Group/Color-By column is set.
        """
        df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna(subset=[x_col, y_col])
        df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[x_col, y_col])

        if df_clean.empty:
            ax.text(0.5, 0.5, f"'{x_col}' and/or '{y_col}' are not numeric — Line plots need numeric X/Y values",
                     ha='center', va='center', transform=ax.transAxes, color=text_color, fontsize=10, wrap=True)
            return

        hue_levels = (hue_levels or list(df_clean[hue_col].dropna().unique())) if hue_col else [y_col]
        # `palette` arrives pre-resolved from generate_plot (defaults, manual
        # click-to-recolor overrides, or B&W shades already applied) - use it
        # directly instead of re-deriving colors from the palette name.
        colors = palette if len(palette) >= len(hue_levels) else sns.color_palette("muted", n_colors=max(len(hue_levels), 1))
        # In B&W mode, also vary line style/marker per curve so overlapping
        # conditions stay distinguishable without relying on color/print.
        bw_styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"), ("-", "v"), ("--", "P")]

        lines_for_legend = []
        for idx, h_level in enumerate(hue_levels):
            sub = df_clean[df_clean[hue_col] == h_level] if hue_col else df_clean
            grouped = sub.groupby(x_col)[y_col]
            means = grouped.mean()
            sems = grouped.apply(
                lambda v: PlottingEngine._spread(v, error_bars))
            order = means.index.argsort()
            xs = means.index.values[order]
            ys = means.values[order]
            errs = sems.values[order]
            linestyle, marker = bw_styles[idx % len(bw_styles)] if bw_mode else ("-", "o")
            line = ax.errorbar(xs, ys, yerr=errs, marker=marker, linestyle=linestyle, capsize=3,
                        color=colors[idx], linewidth=1.8, markersize=5,
                        label=str(h_level))
            lines_for_legend.append((line, str(h_level)))

        # Always show a legend for Line plots (even a single curve) so every
        # curve has a clickable target for manual recoloring.
        legend = ax.legend(title=hue_col if hue_col else None, fontsize=8.5, title_fontsize=9, frameon=False)
        if legend:
            for text_obj, (line, lbl) in zip(legend.get_texts(), lines_for_legend):
                text_obj.set_picker(True)
                text_obj.set_url("group_color:" + lbl)

    @staticmethod
    def _draw_stacked_histogram(df, x_col, y_col, hue_col, ax, resolved_palette, text_color,
                                 bw_mode=False, bins=20):
        """
        Stacked histogram: bins y_col's numeric values, with one stacked
        layer per category - the Group/Color-By column if set, otherwise the
        x_col groups themselves - e.g. the distribution of a readout, split
        by condition, stacked into a single set of bars per bin.
        """
        stack_col = hue_col if hue_col else x_col
        df_clean = df[[y_col, stack_col]].copy()
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[y_col, stack_col])

        categories = list(df_clean[stack_col].dropna().unique())
        if not categories or df_clean.empty:
            ax.text(0.5, 0.5, f"Not enough numeric data in '{y_col}' to build a histogram",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        data_arrays = [df_clean[df_clean[stack_col] == cat][y_col].values for cat in categories]
        colors = resolved_palette if len(resolved_palette) >= len(categories) else \
            sns.color_palette("muted", n_colors=max(len(categories), 1))

        _, _, patch_layers = ax.hist(data_arrays, bins=bins, stacked=True, color=colors,
                                      edgecolor=text_color, linewidth=0.6,
                                      label=[str(c) for c in categories])

        if bw_mode:
            hatch_cycle = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]
            for idx, layer in enumerate(patch_layers):
                for patch in layer:
                    patch.set_hatch(hatch_cycle[idx % len(hatch_cycle)])

        ax.set_title(f"Stacked distribution of {y_col} by {stack_col}", fontsize=14,
                     fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel(y_col, fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel("Count", fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)

        legend = ax.legend(title=stack_col, fontsize=8, title_fontsize=9)
        if legend:
            for text_obj, cat in zip(legend.get_texts(), categories):
                text_obj.set_picker(True)
                text_obj.set_url("group_color:" + str(cat))
        sns.despine(ax=ax, left=True, bottom=True)

    @staticmethod
    def _draw_stacked_bar(df, x_col, y_col, hue_col, ax, resolved_palette, text_color,
                           unique_groups, bw_mode=False, group_colors=None):
        """
        Stacked bar chart: one bar per x_col category, with the bar's total
        height being the sum of y_col across that category's rows - shown as
        stacked segments. If a Group/Color-By column is set, each segment is
        the sum of y_col for that hue level within the x category (the
        classic "stacked bar of sub-totals" layout). Otherwise, each row
        belonging to that x category becomes its own stacked segment (e.g.
        stacking individual replicate readings into one total-per-sample bar).
        """
        hatch_cycle = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]
        df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna(subset=[x_col, y_col])
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[y_col])

        if df_clean.empty:
            ax.text(0.5, 0.5, f"Not enough numeric data in '{y_col}' to build a stacked bar chart",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        x_positions = list(range(len(unique_groups)))

        if hue_col:
            # One segment per hue level, each segment = sum of y_col for that
            # (x_col, hue level) combination.
            hue_levels = list(df_clean[hue_col].dropna().unique())
            colors = resolved_palette if len(resolved_palette) >= len(hue_levels) else \
                sns.color_palette("muted", n_colors=max(len(hue_levels), 1))

            bottoms = [0.0] * len(unique_groups)
            for h_idx, h_level in enumerate(hue_levels):
                sub = df_clean[df_clean[hue_col] == h_level]
                heights = [sub[sub[x_col] == g][y_col].sum() for g in unique_groups]
                bars = ax.bar(x_positions, heights, bottom=bottoms, width=0.6,
                               color=colors[h_idx], label=str(h_level),
                               edgecolor=text_color, linewidth=0.6)
                if bw_mode:
                    for b in bars:
                        b.set_hatch(hatch_cycle[h_idx % len(hatch_cycle)])
                bottoms = [b + h for b, h in zip(bottoms, heights)]

            legend = ax.legend(title=hue_col, fontsize=8.5, title_fontsize=9, frameon=False)
            if legend:
                for text_obj, lbl in zip(legend.get_texts(), hue_levels):
                    text_obj.set_picker(True)
                    text_obj.set_url("group_color:" + str(lbl))
        else:
            # No hue column: stack each row within a category as its own
            # segment, so e.g. 2 replicate values for "a" become 2 segments
            # in one bar, totalling their sum. Segments are named "Value 1",
            # "Value 2"... and are individually recolorable via their legend
            # entry (group_colors is keyed by that same name).
            group_colors = group_colors or {}
            max_rows = int(df_clean.groupby(x_col).size().max())
            default_colors = sns.color_palette("muted", n_colors=max(max_rows, 1))
            segment_names = [f"Value {i + 1}" for i in range(max_rows)]
            colors = [group_colors.get(name, default_colors[i]) for i, name in enumerate(segment_names)]

            bottoms = [0.0] * len(unique_groups)
            for row_idx, seg_name in enumerate(segment_names):
                heights = []
                for g in unique_groups:
                    vals = df_clean[df_clean[x_col] == g][y_col].values
                    heights.append(vals[row_idx] if row_idx < len(vals) else 0.0)
                bars = ax.bar(x_positions, heights, bottom=bottoms, width=0.6,
                               color=colors[row_idx], label=seg_name,
                               edgecolor=text_color, linewidth=0.6)
                if bw_mode:
                    for b in bars:
                        b.set_hatch(hatch_cycle[row_idx % len(hatch_cycle)])
                bottoms = [b + h for b, h in zip(bottoms, heights)]

            if max_rows > 1:
                legend = ax.legend(fontsize=8)
                if legend:
                    for text_obj, seg_name in zip(legend.get_texts(), segment_names):
                        text_obj.set_picker(True)
                        text_obj.set_url("group_color:" + seg_name)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(g) for g in unique_groups], rotation=45,
                            horizontalalignment="right", fontsize=10)

        ax.set_title(f"Stacked total {y_col} by {x_col}" + (f" (by {hue_col})" if hue_col else ""),
                     fontsize=14, fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel(x_col, fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel(y_col, fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        sns.despine(ax=ax, left=True, bottom=True)

    @staticmethod
    def _resolve_continuous_cmap(palette, diverging=False):
        """The palette selector holds seaborn *qualitative* palette names
        (muted, deep, colorblind) alongside two that are also valid Matplotlib
        colormaps (rocket, crest). Heatmaps/networks need a continuous
        colormap, so only rocket/crest pass through as-is; anything else
        falls back to a sensible continuous default."""
        if palette in ("rocket", "crest"):
            return palette
        return "vlag" if diverging else "rocket"

    @staticmethod
    def _draw_heatmap(df, x_col, y_col, hue_col, ax, palette, text_color):
        """
        2D heatmap. If a Group/Color-By column is set, pivots x_col vs hue_col
        with the mean of y_col as the cell value. Otherwise, falls back to a
        multi-metric profile heatmap: every other numeric column found in the
        dataset (e.g. GFP/mKate/OD600) becomes its own z-scored column, one
        row per x_col group - handy for comparing strains/conditions across
        several fluorescence channels at once.
        """
        ax.grid(False)
        df_clean = df.copy()
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")

        diverging = False
        if hue_col:
            pivot = df_clean.pivot_table(index=x_col, columns=hue_col, values=y_col, aggfunc="mean")
            title = f"Mean {y_col} — {x_col} x {hue_col}"
            cbar_label = f"Mean {y_col}"
            fmt = ".2g"
        else:
            numeric_cols = [c for c in df_clean.columns
                            if c != x_col and pd.api.types.is_numeric_dtype(df_clean[c])]
            if len(numeric_cols) >= 2:
                grouped = df_clean.groupby(x_col)[numeric_cols].mean()
                stds = grouped.std(ddof=0).replace(0, 1)
                pivot = (grouped - grouped.mean()) / stds
                title = f"Metric Profile Heatmap by {x_col} (z-scored)"
                cbar_label = "z-score"
                fmt = ".2f"
                diverging = True
            else:
                pivot = df_clean.groupby(x_col)[[y_col]].mean()
                title = f"Mean {y_col} by {x_col}"
                cbar_label = f"Mean {y_col}"
                fmt = ".2g"

        if pivot.empty or pivot.isna().all().all():
            ax.text(0.5, 0.5, "Not enough numeric data to build a heatmap",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        cmap = PlottingEngine._resolve_continuous_cmap(palette, diverging=diverging)
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=fmt, linewidths=0.5,
                    linecolor=text_color, cbar_kws={"label": cbar_label},
                    annot_kws={"fontsize": 8})

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel(pivot.columns.name or "", fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel(pivot.index.name or x_col, fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color, rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    @staticmethod
    def _draw_volcano(df, x_col, y_col, hue_col, ax, text_color, grid_color):
        """
        Standard volcano plot: X = log2 fold-change (numeric column chosen
        via the X Axis selector), Y = raw p-value (numeric column chosen via
        the Metric selector), rendered as -log10(p). Points are classified as
        Up/Down/Not-significant using the conventional |log2FC| >= 1 and
        p < 0.05 thresholds. If a Group/Color-By column is set, it doubles as
        a label column (e.g. gene/feature name) and the most significant
        points are annotated with it.
        """
        ax.grid(True, linestyle="--", alpha=0.5, color=grid_color)

        cols = [x_col, y_col] + ([hue_col] if hue_col else [])
        d = df[cols].copy()
        d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        d = d[d[y_col] > 0]

        if d.empty:
            ax.text(0.5, 0.5,
                    f"'{x_col}' and/or '{y_col}' aren't usable as numeric values.\n"
                    "Volcano needs a numeric log2 fold-change column (X Axis)\n"
                    "and a numeric p-value column (Metric/Y Axis), both > 0.",
                    ha="center", va="center", transform=ax.transAxes, color=text_color,
                    fontsize=10, wrap=True)
            return

        fc_thresh, p_thresh = 1.0, 0.05
        d["_neglog10p"] = -np.log10(d[y_col])

        conditions = [
            (d[y_col] < p_thresh) & (d[x_col] >= fc_thresh),
            (d[y_col] < p_thresh) & (d[x_col] <= -fc_thresh),
        ]
        d["_class"] = np.select(conditions, ["Up", "Down"], default="NS")
        color_map = {"Up": "#D62728", "Down": "#1F77B4", "NS": "#9AA0A6"}

        for cls in ["NS", "Down", "Up"]:  # draw NS first so hits sit on top
            sub = d[d["_class"] == cls]
            if sub.empty:
                continue
            ax.scatter(sub[x_col], sub["_neglog10p"], s=45, alpha=0.75,
                      color=color_map[cls], edgecolor="none", label=f"{cls} (n={len(sub)})")

        # Every point, for the click inspector: on a volcano the label is
        # the whole point of the chart, and there is room to annotate only
        # the top handful of hits.
        d["-log10(p)"] = d["_neglog10p"]
        register_points(ax, d[x_col].to_numpy(float),
                        d["_neglog10p"].to_numpy(float),
                        d.reset_index(drop=True),
                        label_col=hue_col if hue_col else None,
                        fields=[x_col, y_col, "-log10(p)", "_class"],
                        note="Up / Down at |log2FC| ≥ 1 and p < 0.05")

        ax.axhline(-np.log10(p_thresh), color=text_color, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(fc_thresh, color=text_color, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(-fc_thresh, color=text_color, linestyle="--", linewidth=1, alpha=0.6)

        if hue_col:
            top_hits = d[d["_class"] != "NS"].nlargest(8, "_neglog10p")
            for _, row in top_hits.iterrows():
                ax.annotate(str(row[hue_col]), (row[x_col], row["_neglog10p"]),
                           fontsize=7, color=text_color, xytext=(3, 3), textcoords="offset points")

        ax.set_title(f"Volcano Plot — {y_col} vs {x_col}", fontsize=14, fontweight="bold",
                    pad=15, color=text_color)
        ax.set_xlabel(f"{x_col} (log2 fold change)", fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel(f"-log10({y_col})", fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        legend = ax.legend(fontsize=8)
        if legend:
            for text in legend.get_texts():
                text.set_color(text_color)
        sns.despine(ax=ax)

    @staticmethod
    def _draw_dendrogram(df, x_col, y_col, ax, palette, text_color):
        """
        Hierarchical-clustering dendrogram of x_col groups, based on their
        mean profile across every numeric metric column available (z-scored
        so different units/scales contribute comparably). This is a
        similarity tree over the *numeric plate-reader data*, not a
        sequence-based phylogenetic tree - there's no alignment or
        nucleotide/protein distance model behind it, which the caption below
        the plot makes explicit.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist

        ax.grid(False)

        numeric_cols = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
        if y_col in df.columns and y_col not in numeric_cols:
            numeric_cols = [y_col] + numeric_cols
        numeric_cols = list(dict.fromkeys(numeric_cols))

        if not numeric_cols:
            ax.text(0.5, 0.5, "No numeric metric columns found to cluster on",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        grouped = df.groupby(x_col)[numeric_cols].mean().dropna()
        if grouped.shape[0] < 3:
            ax.text(0.5, 0.5,
                    f"Need at least 3 groups in '{x_col}' to build a dendrogram\n(found {grouped.shape[0]})",
                    ha="center", va="center", transform=ax.transAxes, color=text_color, fontsize=10, wrap=True)
            return

        stds = grouped.std(ddof=0).replace(0, 1)
        z = (grouped - grouped.mean()) / stds
        dist = pdist(z.values, metric="euclidean")
        link = linkage(dist, method="average")

        with plt.rc_context({"lines.linewidth": 1.4}):
            dendrogram(link, labels=grouped.index.astype(str).tolist(), ax=ax,
                      color_threshold=0, above_threshold_color=text_color, orientation="right")

        ax.set_title(f"Hierarchical Clustering Dendrogram ({x_col} groups)", fontsize=13,
                    fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel("Distance (z-scored Euclidean, average linkage)", fontsize=10,
                      fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.99, 0.02,
                "Clustered by numeric profile similarity — not a\nsequence-based phylogenetic tree",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=text_color,
                alpha=0.75, style="italic")

    @staticmethod
    def _draw_network(df, x_col, y_col, ax, palette, text_color):
        """
        Correlation network between the dataset's numeric metric columns
        (e.g. OD600/GFP/mKate). Nodes are placed on a circle; edges are drawn
        between pairs whose |Pearson r| clears a threshold, colored blue for
        positive and red for negative correlation, with width/opacity scaled
        to |r|. Built with plain numpy/matplotlib (no networkx dependency).
        """
        ax.grid(False)

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if y_col in df.columns and y_col not in numeric_cols:
            numeric_cols.append(y_col)
        numeric_cols = list(dict.fromkeys(numeric_cols))

        if len(numeric_cols) < 3:
            ax.text(0.5, 0.5,
                    "Need at least 3 numeric metric columns to build a correlation network\n"
                    f"(found {len(numeric_cols)}: {', '.join(numeric_cols) if numeric_cols else 'none'})",
                    ha="center", va="center", transform=ax.transAxes, color=text_color, fontsize=10, wrap=True)
            return

        corr = df[numeric_cols].corr(method="pearson")
        n = len(numeric_cols)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {col: (np.cos(a), np.sin(a)) for col, a in zip(numeric_cols, angles)}

        edge_thresh = 0.3
        for i in range(n):
            for j in range(i + 1, n):
                r = corr.iloc[i, j]
                if pd.isna(r) or abs(r) < edge_thresh:
                    continue
                x1, y1 = pos[numeric_cols[i]]
                x2, y2 = pos[numeric_cols[j]]
                edge_color = "#D62728" if r < 0 else "#1F77B4"
                ax.plot([x1, x2], [y1, y2], color=edge_color, alpha=min(abs(r), 0.9),
                       linewidth=1 + abs(r) * 4, zorder=1)

        # palette_colors(), not sns.color_palette(): the names in the
        # Style tab are this app's own, and seaborn rejects every one of
        # them — which made this chart unusable on the default palette.
        node_colors = palette_colors(palette, n)
        for idx, col in enumerate(numeric_cols):
            x0, y0 = pos[col]
            ax.scatter([x0], [y0], s=900, color=node_colors[idx], edgecolor=text_color,
                      linewidth=1.2, zorder=2)
            ax.text(x0 * 1.25, y0 * 1.25, col, ha="center", va="center", fontsize=9,
                   fontweight="bold", color=text_color, zorder=3)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Metric Correlation Network (|r| \u2265 {edge_thresh})", fontsize=14,
                    fontweight="bold", pad=15, color=text_color)

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="#1F77B4", lw=3, label="Positive correlation"),
            Line2D([0], [0], color="#D62728", lw=3, label="Negative correlation"),
        ]
        legend = ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)
        if legend:
            for text in legend.get_texts():
                text.set_color(text_color)

    # ------------------------------------------------------------------
    # Distributions: density, ridge, raincloud
    # ------------------------------------------------------------------

    @staticmethod
    def _kde(values, grid_n=256, pad=0.15):
        """(grid, density) for *values*, or None when there is no spread.

        A group of one replicate, or of several identical readings, has no
        distribution to estimate — gaussian_kde raises on it. Returning
        None lets the caller draw the value as the spike it actually is
        instead of failing the whole figure.
        """
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return None
        lo, hi = float(v.min()), float(v.max())
        if hi - lo <= 0:
            return None
        try:
            kde = stats.gaussian_kde(v)
        except (ValueError, np.linalg.LinAlgError):
            return None
        span = hi - lo
        grid = np.linspace(lo - pad * span, hi + pad * span, grid_n)
        return grid, kde(grid)

    @staticmethod
    def _split_groups(df, y_col, split_col, levels):
        """[(label, values array)] in the order *levels* asks for."""
        vals = pd.to_numeric(df[y_col], errors="coerce")
        d = pd.DataFrame({"_v": vals.to_numpy(),
                          "_g": df[split_col].astype(str).to_numpy()}).dropna()
        out = []
        for g in levels:
            sub = d.loc[d["_g"] == str(g), "_v"].to_numpy()
            if len(sub):
                out.append((str(g), sub))
        return out

    @staticmethod
    def _draw_density(df, x_col, y_col, hue_col, ax, palette, text_color,
                      grid_color, levels, log_scale=False, fill=True):
        """One density curve per group, all on the same axes.

        The comparison this chart is for is of shapes — whether two
        conditions are shifted, or whether one is simply wider — so the
        curves share an axis rather than being stacked. Every curve is a
        kernel estimate, which smooths; the rug along the bottom shows
        where the actual readings are, so a smooth hump drawn from four
        points cannot be mistaken for a well-sampled one.
        """
        split = hue_col or x_col
        ax.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        ax.set_axisbelow(True)
        series = PlottingEngine._split_groups(df, y_col, split, levels)
        if not series:
            ax.text(0.5, 0.5, f"No numeric values in '{y_col}' to draw a "
                    "density from", ha="center", va="center",
                    transform=ax.transAxes, color=text_color, fontsize=10)
            return
        for i, (label, v) in enumerate(series):
            c = palette[i % len(palette)]
            est = PlottingEngine._kde(v)
            if est is None:
                # A single distinct value: drawn as the vertical line it is.
                ax.axvline(float(v[0]), color=c, lw=1.8,
                           label=f"{label} (n={len(v)}, one value)")
                continue
            grid, dens = est
            ax.plot(grid, dens, color=c, lw=1.9, zorder=3,
                    label=f"{label} (n={len(v)})")
            if fill:
                ax.fill_between(grid, dens, color=c, alpha=0.16, zorder=2,
                                linewidth=0)
            # the readings themselves, as a rug under the curve
            ax.plot(v, np.full(len(v), 0.0), marker="|", linestyle="none",
                    markersize=6, color=c, alpha=0.7, zorder=4,
                    markeredgewidth=1.1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Density", fontsize=11, color=text_color)
        ax.set_xlabel(y_col, fontsize=11, color=text_color)
        ax.set_title(f"Distribution of {y_col} by {split}", fontsize=13,
                     pad=14, color=text_color)
        if log_scale:
            ax.set_xscale("log")
        leg = ax.legend(title=split, fontsize=8.5, title_fontsize=9,
                        frameon=False)
        if leg:
            for t in leg.get_texts():
                t.set_color(text_color)
                t.set_picker(True)
                t.set_url("group_color:" + t.get_text().split(" (n=")[0])
            leg.get_title().set_color(text_color)
        PlottingEngine._style_spines(ax, text_color)

    @staticmethod
    def _draw_ridge(df, x_col, y_col, hue_col, ax, palette, text_color,
                    grid_color, levels, log_scale=False, overlap=1.5):
        """Densities on their own baselines, stacked down the axes.

        The same estimates as the density chart, separated so that a dozen
        conditions stay readable where a dozen overlaid curves would not.
        Each curve is scaled to the same height, so the picture is of
        *where* each group sits and how wide it is — heights carry no
        information and are deliberately not comparable.
        """
        split = hue_col or x_col
        series = PlottingEngine._split_groups(df, y_col, split, levels)
        if not series:
            ax.text(0.5, 0.5, f"No numeric values in '{y_col}' to draw a "
                    "ridge plot from", ha="center", va="center",
                    transform=ax.transAxes, color=text_color, fontsize=10)
            return
        ax.grid(False)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        ax.set_axisbelow(True)
        n = len(series)
        ticks, labels = [], []
        # Top row first, and drawn last-to-first so a lower curve overlaps
        # the one above it rather than being hidden behind it.
        for i, (label, v) in reversed(list(enumerate(series))):
            base = float(n - 1 - i)
            c = palette[i % len(palette)]
            ticks.append(base)
            labels.append(f"{label}  (n={len(v)})")
            est = PlottingEngine._kde(v)
            z = 3 + i
            if est is None:
                ax.plot([float(v[0])] * 2, [base, base + overlap * 0.6],
                        color=c, lw=1.8, zorder=z)
            else:
                grid, dens = est
                peak = float(dens.max()) or 1.0
                curve = base + overlap * dens / peak
                ax.fill_between(grid, base, curve, color=c, alpha=0.75,
                                zorder=z, linewidth=0)
                ax.plot(grid, curve, color=text_color, lw=0.9, alpha=0.8,
                        zorder=z + 0.1)
            ax.axhline(base, color=text_color, lw=0.7, alpha=0.35,
                       zorder=z - 0.5, xmin=0, xmax=1)
            med = float(np.median(v))
            ax.plot([med], [base], marker="v", markersize=5, color=text_color,
                    zorder=z + 0.2, clip_on=False)
            ax.plot(v, np.full(len(v), base - 0.07), marker="|",
                    linestyle="none", markersize=5, color=text_color,
                    alpha=0.6, markeredgewidth=0.9, zorder=z + 0.2)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9.5, color=text_color)
        for lab in ax.get_yticklabels():
            lab.set_picker(True)
            lab.set_url("group_color:" + lab.get_text().split("  (n=")[0])
        ax.set_ylim(-0.4, n - 1 + overlap + 0.25)
        ax.set_xlabel(y_col, fontsize=11, color=text_color)
        ax.set_ylabel("")
        ax.set_title(f"{y_col} by {split}", fontsize=13, pad=14,
                     color=text_color)
        if log_scale:
            ax.set_xscale("log")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(text_color)
        ax.tick_params(axis="y", length=0, colors=text_color)
        ax.tick_params(axis="x", colors=text_color)
        ax.text(0.995, 0.005, "Curves scaled to equal height — "
                "the marker is the median",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color=text_color, alpha=0.7, style="italic")

    @staticmethod
    def _draw_raincloud(df, x_col, y_col, ax, groups, hue_col, palette,
                        text_color, hue_levels=None, bw_mode=False):
        """Half violin, box and raw points, side by side per group.

        A box plot hides whether a group is bimodal; a swarm hides the
        quartiles; and either alone hides how few points there are. The
        raincloud shows all three at once, which is why it is worth the
        width it takes: the cloud is the estimated shape, the box is the
        summary that gets quoted, and the rain is every reading that went
        into both.
        """
        # The whole row, not only the columns being drawn: clicking a
        # point is for finding out which sample it is, and the well, the
        # plate and the date are exactly what identifies it.
        d = df.copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[y_col])
        if d.empty:
            ax.text(0.5, 0.5, f"No numeric values in '{y_col}'", ha="center",
                    va="center", transform=ax.transAxes, color=text_color)
            return
        levels = [str(h) for h in (hue_levels or [])] if hue_col else [None]
        width = 0.8 / max(len(levels), 1)
        rng = np.random.default_rng(0)        # same jitter on every redraw
        px, py, rows = [], [], []
        for gi, g in enumerate(groups):
            in_group = d[x_col].astype(str) == str(g)
            for li, lv in enumerate(levels):
                sub = d[in_group] if lv is None else \
                    d[in_group & (d[hue_col].astype(str) == lv)]
                v = sub[y_col].to_numpy(float)
                if not len(v):
                    continue
                c = palette[(li if hue_col else gi) % len(palette)]
                centre = gi + (li - (len(levels) - 1) / 2) * width
                half = width * 0.42
                est = PlottingEngine._kde(v)
                if est is not None:
                    grid, dens = est
                    inside = (grid >= v.min()) & (grid <= v.max())
                    grid, dens = grid[inside], dens[inside]
                    if len(grid) > 1:
                        peak = float(dens.max()) or 1.0
                        edge = centre + 0.06 * width + half * dens / peak
                        ax.fill_betweenx(grid, centre + 0.06 * width, edge,
                                         color=c, alpha=0.55, linewidth=0,
                                         zorder=2)
                        ax.plot(edge, grid, color=c, lw=1.0, zorder=2.1)
                # the box, narrow, on the centre line
                bp = ax.boxplot([v], positions=[centre - 0.06 * width],
                                widths=half * 0.45,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color=text_color, lw=1.3),
                                whiskerprops=dict(color=text_color, lw=1.0),
                                capprops=dict(color=text_color, lw=1.0),
                                zorder=3)
                for box in bp["boxes"]:
                    box.set_facecolor("none")
                    box.set_edgecolor(text_color)
                    box.set_linewidth(1.0)
                # the rain
                jx = centre - 0.06 * width - half * (0.55 + rng.random(len(v)) * 0.45)
                ax.plot(jx, v, marker="o", linestyle="none", markersize=4.2,
                        color=c, alpha=0.75, markeredgecolor="none", zorder=4,
                        label=(lv if (hue_col and gi == 0) else None))
                px.extend(jx.tolist())
                py.extend(v.tolist())
                rows.append(sub)
        if rows:
            whole = pd.concat(rows, ignore_index=True)
            register_points(ax, px, py, whole, label_col=hue_col or x_col,
                            fields=PlottingEngine._inspect_fields(
                                whole, y_col, hue_col or x_col),
                            x_slack=0.45)
        ax.set_xticks(range(len(groups)))
        ax.set_xlim(-0.6, len(groups) - 0.4)
        return True

    # ------------------------------------------------------------------
    # Set charts: Venn and UpSet
    # ------------------------------------------------------------------

    @staticmethod
    def _member_sets(df, x_col, hue_col):
        """{set name: {members}} — X groups are the sets, Color by the members.

        Long-format tables are what this app reads, so a set is "the rows
        belonging to this condition" and a member is the thing being
        counted in it: the gene, the construct, the strain. Without a
        Color-by column there is nothing to intersect — rows are not
        comparable across conditions — which is what the callers report.
        """
        if not hue_col or hue_col not in df.columns:
            return {}
        d = df[[x_col, hue_col]].dropna()
        if d.empty:
            return {}
        out = {}
        for g, sub in d.groupby(d[x_col].astype(str), sort=False):
            members = set(sub[hue_col].astype(str))
            if members:
                out[str(g)] = members
        return out

    @staticmethod
    def _set_chart_note(ax, x_col, hue_col, n_sets, text_color, upto):
        """The message when Venn/UpSet has not been given what it needs."""
        if not hue_col:
            msg = ("Set 'Color by' to the column naming what is in each "
                   f"set — a gene, construct or strain.\n\nThe X groups "
                   f"('{x_col}') become the sets; the Color-by values "
                   "become their members.")
        elif n_sets == 0:
            msg = (f"No rows with both '{x_col}' and '{hue_col}' filled in.")
        elif n_sets < 2:
            msg = (f"Only one set ('{x_col}' has a single value here) — "
                   "there is nothing to intersect.")
        else:
            msg = (f"{n_sets} sets is more than a Venn diagram can draw "
                   f"honestly (up to {upto}).\nUse the UpSet plot, which "
                   "handles any number of sets.")
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color=text_color, fontsize=10,
                wrap=True, linespacing=1.5)
        ax.axis("off")

    @staticmethod
    def _draw_venn(df, x_col, y_col, hue_col, ax, palette, text_color):
        """Two or three sets as overlapping circles, with exact counts.

        The circles are fixed in size and position: drawing them to scale
        is impossible for most three-set data, and a diagram that is
        almost-to-scale invites reading areas that are not true. The
        numbers are exact and are what the figure is for.
        """
        sets = PlottingEngine._member_sets(df, x_col, hue_col)
        names = list(sets)
        if len(names) not in (2, 3):
            PlottingEngine._set_chart_note(ax, x_col, hue_col, len(names),
                                           text_color, upto=3)
            return
        ax.grid(False)
        ax.axis("off")
        ax.set_aspect("equal")
        from matplotlib.patches import Circle

        def count(include, exclude):
            s = set.intersection(*[sets[n] for n in include])
            for n in exclude:
                s = s - sets[n]
            return len(s)

        if len(names) == 2:
            a, b = names
            centres = [(-0.30, 0.0), (0.30, 0.0)]
            r = 0.52
            regions = [((-0.62, 0.0), count([a], [b])),
                       ((0.62, 0.0), count([b], [a])),
                       ((0.0, 0.0), count([a, b], []))]
            label_pos = [(-0.62, 0.62), (0.62, 0.62)]
            union = len(sets[a] | sets[b])
            inter = count([a, b], [])
            caption = (f"{inter} shared of {union} in total — "
                       f"Jaccard {inter / union:.2f}" if union else "")
        else:
            a, b, c = names
            r = 0.50
            ang = np.deg2rad([150.0, 30.0, 270.0])
            centres = [(0.29 * np.cos(t), 0.29 * np.sin(t)) for t in ang]
            singles = [(0.537 * np.cos(t), 0.537 * np.sin(t)) for t in ang]
            pair_ang = np.deg2rad([90.0, 210.0, 330.0])      # ab, ac, bc
            pairs = [(0.31 * np.cos(t), 0.31 * np.sin(t)) for t in pair_ang]
            regions = [
                (singles[0], count([a], [b, c])),
                (singles[1], count([b], [a, c])),
                (singles[2], count([c], [a, b])),
                (pairs[0], count([a, b], [c])),
                (pairs[1], count([a, c], [b])),
                (pairs[2], count([b, c], [a])),
                ((0.0, 0.0), count([a, b, c], [])),
            ]
            label_pos = [(x * 1.65, y * 1.65 + (0.12 if y > 0 else -0.12))
                         for x, y in centres]
            union = len(set.union(*sets.values()))
            caption = (f"{count([a, b, c], [])} in all three, "
                       f"{union} distinct {hue_col} values in total")

        for i, (cx, cy) in enumerate(centres):
            col = palette[i % len(palette)]
            ax.add_patch(Circle((cx, cy), r, facecolor=col, alpha=0.38,
                                edgecolor=col, linewidth=1.8, zorder=2))
        for (px, py), n in regions:
            if n == 0:
                continue
            ax.text(px, py, str(n), ha="center", va="center", fontsize=13,
                    fontweight="bold", color=text_color, zorder=4)
        for i, (px, py) in enumerate(label_pos):
            ax.text(px, py, names[i], ha="center", va="center", fontsize=10.5,
                    fontweight="semibold", color=palette[i % len(palette)],
                    zorder=4)
        lim = 1.25
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 0.95, lim * 0.95)
        ax.set_title(f"{hue_col} shared between {x_col} groups", fontsize=13,
                     pad=12, color=text_color)
        if caption:
            ax.text(0.5, -0.02, caption, transform=ax.transAxes, ha="center",
                    va="top", fontsize=8.5, color=text_color, alpha=0.8)

    @staticmethod
    def _draw_upset(df, x_col, y_col, hue_col, ax, palette, text_color,
                    grid_color, max_cols=18):
        """Intersection sizes as bars over a membership matrix.

        What a Venn diagram cannot do past three sets. Each column is one
        exact combination — the members in these sets and no others — so
        the columns partition the data and their sizes add up to the
        total, which is not true of the overlapping regions people usually
        read off a Venn.
        """
        sets = PlottingEngine._member_sets(df, x_col, hue_col)
        names = list(sets)
        if len(names) < 2:
            PlottingEngine._set_chart_note(ax, x_col, hue_col, len(names),
                                           text_color, upto=3)
            return
        ax.grid(False)
        ax.axis("off")
        # Every member's exact signature: which sets it is in.
        combos = {}
        for member in set.union(*sets.values()):
            sig = tuple(n for n in names if member in sets[n])
            combos.setdefault(sig, []).append(member)
        order = sorted(combos, key=lambda s: (-len(combos[s]), len(s)))
        hidden = max(0, len(order) - max_cols)
        order = order[:max_cols]
        # Sets ordered by their own size, largest at the bottom row, which
        # is where an UpSet plot conventionally puts it.
        rows = sorted(names, key=lambda n: len(sets[n]), reverse=True)
        n_rows, n_cols = len(rows), len(order)

        # The set names are the matrix's y tick labels, so they are drawn
        # in the gap to its left — which is why the size bars stop short of
        # it rather than running up to the dots.
        bars = ax.inset_axes([0.34, 0.42, 0.65, 0.56])
        mat = ax.inset_axes([0.34, 0.06, 0.65, 0.34], sharex=bars)
        # Not sharey with the matrix: clearing this panel's ticks would
        # clear the shared locator, and the set names are those ticks.
        side = ax.inset_axes([0.00, 0.06, 0.18, 0.34])
        for sub in (bars, mat, side):
            sub.set_facecolor("none")
            sub.tick_params(colors=text_color, labelsize=8.5)

        sizes = [len(combos[s]) for s in order]
        bars.bar(range(n_cols), sizes, width=0.62, color=palette[0 % len(palette)],
                 edgecolor="none", zorder=3)
        for i, v in enumerate(sizes):
            bars.text(i, v, str(v), ha="center", va="bottom", fontsize=8,
                      color=text_color)
        bars.set_ylim(0, max(sizes) * 1.18)
        bars.yaxis.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        bars.set_axisbelow(True)
        bars.set_ylabel("Intersection", fontsize=9.5, color=text_color)
        bars.tick_params(labelbottom=False, bottom=False)
        for s in ("top", "right", "bottom"):
            bars.spines[s].set_visible(False)
        bars.spines["left"].set_color(text_color)

        dim = mcolors.to_rgba(text_color, 0.16)
        for j, sig in enumerate(order):
            members = [i for i, nm in enumerate(rows) if nm in sig]
            mat.plot([j] * n_rows, range(n_rows), marker="o", linestyle="none",
                     markersize=7, color=dim, markeredgecolor="none", zorder=2)
            if members:
                mat.plot([j, j], [min(members), max(members)], color=text_color,
                         lw=1.6, zorder=3)
                mat.plot([j] * len(members), members, marker="o",
                         linestyle="none", markersize=7, color=text_color,
                         markeredgecolor="none", zorder=4)
        mat.set_xlim(-0.6, n_cols - 0.4)
        mat.set_ylim(-0.6, n_rows - 0.4)
        mat.set_yticks(range(n_rows))
        mat.set_yticklabels(rows, fontsize=9, color=text_color)
        mat.set_xticks([])
        for s in mat.spines.values():
            s.set_visible(False)
        mat.tick_params(length=0)

        side.barh(range(n_rows), [len(sets[n]) for n in rows], height=0.55,
                  color=palette[1 % len(palette)], edgecolor="none", zorder=3)
        side.invert_xaxis()
        side.set_ylim(mat.get_ylim())
        side.tick_params(left=False, labelleft=False)
        side.set_xlabel("Set size", fontsize=9, color=text_color)
        side.xaxis.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        side.set_axisbelow(True)
        for s in ("top", "right", "left"):
            side.spines[s].set_visible(False)
        side.spines["bottom"].set_color(text_color)
        side.tick_params(labelsize=8)

        ax.set_title(f"{hue_col} shared between {x_col} groups", fontsize=13,
                     pad=12, color=text_color)
        note = ("Each column is one exact combination, so the columns add up "
                "to the total")
        if hidden:
            note += f" — {hidden} smaller combination(s) not shown"
        ax.text(0.5, -0.07, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color=text_color, alpha=0.8)

    # ------------------------------------------------------------------
    # Bubble and dot plots
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_bubble(df, x_col, y_col, hue_col, size_col, ax, palette,
                     text_color, grid_color, levels, log_scale=False):
        """Scatter with a third measurement in the marker area.

        Area, not radius: the eye reads a bubble by how much ink it is, so
        scaling the radius exaggerates large values about fourfold. The
        legend shows three real values from the data at their drawn size,
        because no one can read an area off an axis.
        """
        ax.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        ax.set_axisbelow(True)
        if not size_col or size_col not in df.columns:
            ax.text(0.5, 0.5, "Choose a 'Bubble size' column — the third "
                    "measurement,\ndrawn as the area of each point.",
                    ha="center", va="center", transform=ax.transAxes,
                    color=text_color, fontsize=10, linespacing=1.5)
            return
        cols = list(dict.fromkeys([c for c in (x_col, y_col, size_col, hue_col)
                                   if c]))
        d = df[cols].copy()
        for c in (x_col, y_col, size_col):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[y_col, size_col])
        numeric_x = d[x_col].notna().any()
        if not numeric_x:
            d = df[cols].copy()
            d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
            d[size_col] = pd.to_numeric(d[size_col], errors="coerce")
            d = d.dropna(subset=[y_col, size_col])
        if d.empty:
            ax.text(0.5, 0.5, f"'{y_col}' and '{size_col}' need numeric "
                    "values in the same rows", ha="center", va="center",
                    transform=ax.transAxes, color=text_color, fontsize=10)
            return
        s_raw = d[size_col].to_numpy(float)
        lo, hi = float(np.nanmin(s_raw)), float(np.nanmax(s_raw))
        S_MIN, S_MAX = 40.0, 900.0
        if hi > lo:
            area = S_MIN + (s_raw - lo) / (hi - lo) * (S_MAX - S_MIN)
        else:
            area = np.full(len(s_raw), (S_MIN + S_MAX) / 2)
        if numeric_x:
            xpos = d[x_col].to_numpy(float)
            cats = None
        else:
            cats = [str(g) for g in dict.fromkeys(df[x_col].astype(str))
                    if str(g) in set(d[x_col].astype(str))]
            index = {g: i for i, g in enumerate(cats)}
            xpos = d[x_col].astype(str).map(index).to_numpy(float)
        if hue_col and hue_col in d.columns:
            keys = [str(v) for v in (levels or [])]
            colmap = {k: palette[i % len(palette)] for i, k in enumerate(keys)}
            colors = [colmap.get(str(v), palette[0]) for v in d[hue_col]]
        else:
            colors = [palette[0 % len(palette)]] * len(d)
        ax.scatter(xpos, d[y_col].to_numpy(float), s=area, c=colors,
                   alpha=0.62, edgecolor=text_color, linewidth=0.6, zorder=3)
        register_points(ax, xpos, d[y_col].to_numpy(float),
                        d.reset_index(drop=True),
                        label_col=hue_col if hue_col else None)
        if cats is not None:
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=10)
            ax.set_xlim(-0.6, len(cats) - 0.4)
        if log_scale:
            ax.set_yscale("log")
        # A size key, from three values actually present in the data.
        from matplotlib.lines import Line2D
        qs = np.nanpercentile(s_raw, [5, 50, 95])
        keyed, seen = [], set()
        for v in qs:
            r = float(np.round(v, 3 - int(np.floor(np.log10(abs(v)))) - 1)
                      if np.isfinite(v) and v != 0 else v)
            if r in seen:
                continue
            seen.add(r)
            a = (S_MIN + (v - lo) / (hi - lo) * (S_MAX - S_MIN)) if hi > lo \
                else (S_MIN + S_MAX) / 2
            keyed.append(Line2D([0], [0], marker="o", linestyle="none",
                                markerfacecolor="none",
                                markeredgecolor=text_color,
                                markersize=float(np.sqrt(a)), label=f"{r:g}"))
        if keyed:
            key = ax.legend(handles=keyed, title=size_col, loc="upper left",
                            frameon=False, labelspacing=1.6, fontsize=8.5,
                            title_fontsize=9, handletextpad=1.4,
                            borderpad=0.9)
            key.get_title().set_color(text_color)
            for t in key.get_texts():
                t.set_color(text_color)
            ax.add_artist(key)
        if hue_col:
            handles = [Line2D([0], [0], marker="o", linestyle="none",
                              color=palette[i % len(palette)],
                              markersize=7, label=str(k))
                       for i, k in enumerate(levels or [])]
            if handles:
                leg = ax.legend(handles=handles, title=hue_col, fontsize=8.5,
                                title_fontsize=9, frameon=False,
                                loc="lower right")
                leg.get_title().set_color(text_color)
                for t, k in zip(leg.get_texts(), levels or []):
                    t.set_color(text_color)
                    t.set_picker(True)
                    t.set_url("group_color:" + str(k))
        ax.set_xlabel(x_col, fontsize=11, color=text_color)
        ax.set_ylabel(y_col, fontsize=11, color=text_color)
        ax.set_title(f"{y_col} vs {x_col}, sized by {size_col}", fontsize=13,
                     pad=14, color=text_color)
        PlottingEngine._style_spines(ax, text_color)

    @staticmethod
    def _draw_dotplot(df, x_col, y_col, hue_col, ax, palette, text_color,
                      grid_color, groups, levels, error_bars="95% CI",
                      log_scale=False, sort_by_value=True):
        """Group means on a horizontal axis, one row per group.

        A Cleveland dot plot. It answers the same question as a bar chart
        with a fraction of the ink, reads cleanly with thirty groups where
        bars do not, and — because it is not anchored at zero — does not
        imply that zero is meaningful for a ratio or a normalised value.
        The interval is drawn so the dots are not read as exact.
        """
        ax.grid(False)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.8, color=grid_color)
        ax.set_axisbelow(True)
        d = df[[c for c in dict.fromkeys([x_col, y_col] +
                                         ([hue_col] if hue_col else []))]].copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[y_col])
        if d.empty:
            ax.text(0.5, 0.5, f"No numeric values in '{y_col}'", ha="center",
                    va="center", transform=ax.transAxes, color=text_color)
            return
        series = [None] if not hue_col else [str(h) for h in (levels or [])]
        stats_rows = []
        for g in groups:
            in_g = d[x_col].astype(str) == str(g)
            for lv in series:
                sub = d[in_g] if lv is None else \
                    d[in_g & (d[hue_col].astype(str) == lv)]
                v = sub[y_col].to_numpy(float)
                if not len(v):
                    continue
                err = PlottingEngine._spread(v, error_bars)
                stats_rows.append({"_group": str(g), "_series": lv,
                                   "n": len(v), "mean": float(np.mean(v)),
                                   "err": float(err) if np.isfinite(err) else 0.0})
        if not stats_rows:
            ax.text(0.5, 0.5, "Nothing to summarise", ha="center", va="center",
                    transform=ax.transAxes, color=text_color)
            return
        summary = pd.DataFrame(stats_rows)
        order = list(dict.fromkeys(summary["_group"]))
        if sort_by_value:
            # Largest at the top, which is how a ranked list is read.
            rank = summary.groupby("_group")["mean"].max().sort_values()
            order = list(rank.index)
        pos = {g: i for i, g in enumerate(order)}
        px, py = [], []
        for li, lv in enumerate(series):
            part = summary[summary["_series"] == lv] if lv is not None \
                else summary
            if part.empty:
                continue
            off = 0.0 if lv is None else \
                (li - (len(series) - 1) / 2) * (0.7 / max(len(series), 1))
            ys = [pos[g] + off for g in part["_group"]]
            c = palette[li % len(palette)] if hue_col else None
            for j, (g, m, e) in enumerate(zip(part["_group"], part["mean"],
                                              part["err"])):
                colour = c if c is not None else \
                    palette[list(order).index(g) % len(palette)]
                y = pos[g] + off
                if e:
                    ax.plot([m - e, m + e], [y, y], color=colour, lw=1.5,
                            alpha=0.85, zorder=2,
                            solid_capstyle="butt")
                ax.plot([m], [y], marker="o", markersize=8, color=colour,
                        markeredgecolor=text_color, markeredgewidth=0.7,
                        zorder=3, linestyle="none",
                        label=(str(lv) if (hue_col and j == 0) else None))
                px.append(m)
                py.append(y)
        register_points(ax, px, py, summary.reset_index(drop=True),
                        label_col="_group",
                        fields=["_series", "n", "mean", "err"])
        # The guide lines run from the left spine to each dot, so they are
        # drawn once the axis has settled on its range. Earlier they would
        # have been anchored to a default limit and dragged the axis out to
        # it — on a fold change or a normalised value that means implying
        # zero is on the scale when it is not.
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)
        left = ax.get_xlim()[0]
        for m, y in zip(px, py):
            ax.plot([left, m], [y, y], color=text_color, lw=0.7, alpha=0.18,
                    zorder=0)
        ax.set_xlim(left=left)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([str(g) for g in order], fontsize=10,
                           color=text_color)
        for lab in ax.get_yticklabels():
            lab.set_picker(True)
            lab.set_url("group_color:" + lab.get_text())
        ax.set_ylim(-0.6, len(order) - 0.4)
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(f"{y_col}  (mean ± {error_bars})", fontsize=11,
                      color=text_color)
        ax.set_ylabel("")
        ax.set_title(f"{y_col} by {x_col}", fontsize=13, pad=14,
                     color=text_color)
        if hue_col:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                leg = ax.legend(title=hue_col, fontsize=8.5,
                                title_fontsize=9, frameon=False)
                leg.get_title().set_color(text_color)
                for t, lbl in zip(leg.get_texts(), labels):
                    t.set_color(text_color)
                    t.set_picker(True)
                    t.set_url("group_color:" + lbl)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(text_color)
        ax.tick_params(axis="y", length=0, colors=text_color)
        ax.tick_params(axis="x", colors=text_color)

    # Dodge width used by each chart type when a Color-By column is set
    # (where each colour's bar/box sits inside its x group)
    DODGE_WIDTH = {"Bar": 0.8, "Boxplot": 0.5, "Violin": 0.8,
                   "Beeswarm": 0.8, "Box + Swarm": 0.8, "Raincloud": 0.8}
    MAX_BRACKETS = 15
    #: The chart types significance brackets can be drawn on. The rest are
    #: either not group comparisons (Line, Scatter, Heatmap, Volcano…) or
    #: draw themselves whole (Stacked Bar, the model checks).
    STAT_CHARTS = ["Boxplot", "Violin", "Beeswarm", "Box + Swarm", "Bar",
                   "Raincloud"]

    @staticmethod
    def _stat_note(ax, text, line_color):
        """Say why no brackets were drawn, instead of leaving the plot
        looking as though the test never ran."""
        ax.text(0.99, 0.99, text, transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color=line_color, alpha=0.85)

    @staticmethod
    def _artist_tops(ax, n_groups) -> dict:
        """{x position: highest y drawn there} over everything on the axes
        (violins reach past the data points; bars carry error bars)."""
        from matplotlib.collections import PathCollection
        xs, ys = [], []
        for coll in ax.collections:
            if coll.get_transform() != ax.transData and \
                    not isinstance(coll, PathCollection):
                continue
            if isinstance(coll, PathCollection):
                off = np.asarray(coll.get_offsets())
                if off.size:
                    xs.append(off[:, 0]); ys.append(off[:, 1])
            else:
                for path in coll.get_paths():
                    v = path.vertices
                    if len(v):
                        i = np.argmax(v[:, 1])
                        xs.append([v[:, 0].mean()]); ys.append([v[i, 1]])
        for line in ax.lines:
            xy = np.asarray(line.get_xydata(), float)
            if xy.size:
                xs.append(xy[:, 0]); ys.append(xy[:, 1])
        for patch in ax.patches:
            try:
                bb = patch.get_bbox()
                xs.append([(bb.x0 + bb.x1) / 2]); ys.append([bb.y1])
            except Exception:
                continue
        if not xs:
            return {}
        x = np.concatenate([np.ravel(a) for a in xs]).astype(float)
        y = np.concatenate([np.ravel(a) for a in ys]).astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        out = {}
        for i in range(n_groups):
            m = np.abs(x - i) < 0.5
            if m.any():
                out[i] = float(y[m].max())
        return out

    @staticmethod
    def annotate_significance(df, x_col, y_col, groups, ax, stat_mode, line_color,
                               reference=None, pairing_col=None, log_scale=False,
                               show_pvalue=False, hue_col=None, hue_order=None,
                               dodge_width=0.8, show_ns=False, bar_tops=None):
        """
        Draws significance brackets from StatisticsEngine.compare() — the
        same results the Statistical Analysis tab reports. Without a
        Color-By column the x groups are compared; with one, the colours are
        compared inside each x group (Holm-corrected across groups).
        Brackets are stacked from the shortest up, each just above the data
        it spans. Returns the comparisons drawn.
        """
        SE = StatisticsEngine
        d = df.copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        if d.empty:
            return []
        d["_x"] = d[x_col].astype(str)
        groups = [str(g) for g in groups]
        xpos = {g: i for i, g in enumerate(groups)}

        if hue_col:
            d["_h"] = d[hue_col].astype(str)
            levels = [str(h) for h in (hue_order or d["_h"].unique())]
            n = max(len(levels), 1)
            step = dodge_width / n

            def pos(c):
                base = xpos[c.within]
                return (base + (levels.index(c.a) - (n - 1) / 2) * step,
                        base + (levels.index(c.b) - (n - 1) / 2) * step)
            results = SE.compare_within(d, "_x", "_h", y_col, stat_mode,
                                        reference, pairing_col, levels)
            comps = [c for r in results.values() for c in r.comparisons
                     if c.within in xpos]
        else:
            r = SE.compare(d, "_x", y_col, stat_mode, reference, pairing_col,
                           group_order=groups)
            comps = [c for c in r.comparisons if c.a in xpos and c.b in xpos]

            def pos(c):
                return xpos[c.a], xpos[c.b]

        shown = [c for c in comps if show_ns or c.significant]
        if not shown:
            return []

        # work in display units (log10 on a log axis)
        to_u = (lambda v: np.log10(v)) if log_scale else (lambda v: v)
        from_u = (lambda u: 10 ** u) if log_scale else (lambda u: u)
        vals = d[y_col].to_numpy(float)
        if log_scale:
            vals = vals[vals > 0]
            if vals.size == 0:
                return []
        lo, hi = to_u(vals.min()), to_u(vals.max())
        if bar_tops:
            hi = max(hi, max(to_u(v) for v in bar_tops.values() if v > 0)
                     if any(v > 0 for v in bar_tops.values()) else hi)
        span = (hi - lo) or 1.0
        step_u = span * (0.13 if show_pvalue else 0.10)
        tick = step_u * 0.18

        drawn_tops = PlottingEngine._artist_tops(ax, len(groups))
        if drawn_tops:
            hi = max([hi] + [to_u(v) for v in drawn_tops.values() if v > 0])
            span = (hi - lo) or 1.0
            step_u = span * (0.13 if show_pvalue else 0.10)
            tick = step_u * 0.18

        def top_of(x1, x2):
            """Highest thing drawn (points, bar + error, box, violin)
            between two x positions."""
            a, b = sorted((x1, x2))
            gs = [g for g, i in xpos.items() if a - 0.5 < i < b + 0.5]
            sub = d[d["_x"].isin(gs)][y_col].to_numpy(float)
            if log_scale:
                sub = sub[sub > 0]
            t = to_u(sub.max()) if sub.size else hi
            for g in gs:
                v = drawn_tops.get(xpos[g])
                if v is not None and (v > 0 or not log_scale):
                    t = max(t, to_u(v))
            if bar_tops:
                t = max([t] + [to_u(bar_tops[g]) for g in gs
                               if bar_tops.get(g, 0) > 0])
            return t

        placed = []                         # (x1, x2, level)
        shown.sort(key=lambda c: abs(pos(c)[1] - pos(c)[0]))
        drawn = []
        for c in shown[:PlottingEngine.MAX_BRACKETS]:
            x1, x2 = sorted(pos(c))
            level = top_of(x1, x2) + step_u * 0.6
            for (a, b, lev) in placed:      # stack above overlapping ones
                if a - 0.05 <= x2 and x1 <= b + 0.05:
                    level = max(level, lev + step_u)
            placed.append((x1, x2, level))
            label = stars(c.p)
            if show_pvalue:
                label = f"{label} ({format_p(c.p)})" if label != "ns" \
                    else format_p(c.p)
            ys = [from_u(level), from_u(level + tick)]
            ax.plot([x1, x1, x2, x2], [ys[0], ys[1], ys[1], ys[0]], lw=1.1,
                    color=line_color, solid_capstyle="butt")
            ax.text((x1 + x2) / 2, from_u(level + tick), label, ha="center",
                    va="bottom", color=line_color,
                    fontsize=8.5 if show_pvalue else 10.5,
                    fontweight="bold" if label != "ns" else "normal")
            drawn.append(c)
        top = max(lev for _, _, lev in placed) + step_u
        extra = len(shown) - len(drawn)
        if extra > 0:
            ax.text(0.99, 0.99, f"+{extra} more significant comparison"
                    f"{'s' if extra != 1 else ''} — see Statistical Analysis",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=line_color)
        bottom = ax.get_ylim()[0]
        ax.set_ylim(bottom=bottom, top=from_u(top))
        return drawn
