# src/model_check.py
"""Compare a model's predictions with what was measured.

A predicted value is usually in arbitrary units, so it cannot be put on
the same axis as a measurement without being scaled first. Everything here
works in log space, because a model of this kind is right or wrong by a
*factor*, not by a difference: being out by 100 matters enormously at a
measured value of 50 and not at all at 50,000.

Each model is scaled by one constant, chosen to minimise the log-space
error — which is the same as matching the geometric means. Anchoring on a
single reference point instead would let that one point's noise set the
whole comparison.

The headline number is the **typical fold error**: the RMS of the
log-space residuals after scaling, read back as a factor. "2.2x" means a
prediction is typically out by a bit over twofold either way.
"""

# ---------------------------------------------------------------------------
# THE PIPELINE
# ---------------------------------------------------------------------------
#   prepare()    collapse replicates to one row per construct, drop what
#                cannot go on a log axis, count what was dropped
#   fit_scale()  one constant per model, fitted in log space
#   fold_error() how far off it still is, as a factor
#   log_pearson() how well it *orders* the constructs
#   summarise()  all of the above per model, into one dict
#   report() / draw_scatter() / draw_bars()  present it
#
# Two numbers are reported rather than one because they answer different
# questions and can disagree. A model can rank every construct correctly
# (high r) while being out by fivefold on all of them (large fold
# error) — that means the shape is right and something constant is
# missing. The opposite, small fold error with poor r, means it is close
# on average and random about which construct is which, which is worse.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Points at or below zero cannot go on a log axis, and a model of this
#: kind has nothing to say about them; they are dropped and counted.
MIN_POSITIVE = 0.0


def prepare(df, label_col, measured_col, predicted_cols, agg="median"):
    """One row per label, with the measured value and each prediction.

    Replicates are collapsed with *agg* ("median" or "mean"). Rows where
    the measurement or any prediction is missing or non-positive are
    dropped — the count comes back in the report.
    """
    # Measurement first, then the predictions; missing columns silently
    # dropped so a half-configured comparison still shows something.
    cols = [c for c in [measured_col, *predicted_cols] if c and c in df]
    if not label_col or label_col not in df or measured_col not in cols:
        return pd.DataFrame(), 0
    d = df[[label_col, *cols]].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    before = len(d)
    d = d.dropna(subset=cols)
    # .all(axis=1): a row survives only if *every* column is positive.
    # One model predicting zero would otherwise leave a row that cannot
    # be logged for that model but can for the others, and the models
    # would then be scored on different subsets.
    d = d[(d[cols] > MIN_POSITIVE).all(axis=1)]
    dropped = before - len(d)
    if d.empty:
        return pd.DataFrame(), dropped
    # Median by default rather than mean: replicate measurements of
    # expression are log-normal and skewed, so the median is the more
    # honest summary of the middle.
    out = d.groupby(label_col, sort=False)[cols].agg(agg).reset_index()
    return out, dropped


def fit_scale(pred, meas):
    """The single factor that best lines *pred* up with *meas*, in log
    space — i.e. the ratio of their geometric means."""
    p, m = np.asarray(pred, float), np.asarray(meas, float)
    ok = (p > 0) & (m > 0)
    if not ok.any():
        return 1.0
    # The mean difference of the logs, raised back out of log space.
    # That is exactly the ratio of the two geometric means, and it is
    # the factor that minimises the squared log residuals — so the same
    # constant that fold_error() is about to be measured against.
    return float(10 ** np.mean(np.log10(m[ok]) - np.log10(p[ok])))


def fold_error(pred, meas, scale=None):
    """Typical fold error after scaling: RMS log residual, as a factor."""
    p, m = np.asarray(pred, float), np.asarray(meas, float)
    ok = (p > 0) & (m > 0)
    if ok.sum() < 1:
        return float("nan")
    # *scale* may be passed in so that summarise() fits it once and
    # reuses it, rather than refitting here on the same data.
    s = fit_scale(p[ok], m[ok]) if scale is None else scale
    # Residuals in log space, RMS, then raised back. The result reads as
    # "typically out by this factor" — and because it is symmetric in
    # log space, 2.0 means twice as much *or* half as much.
    r = np.log10(p[ok] * s) - np.log10(m[ok])
    return float(10 ** math.sqrt(float(np.mean(r ** 2))))


def log_pearson(pred, meas):
    """Pearson r between log10 prediction and log10 measurement.

    Scaling does not change it — it only shifts the logs — so this says
    how well the model *orders* constructs, while fold_error says how
    close it gets to each one. They can disagree, and when they do the
    difference is the useful thing.
    """
    p, m = np.asarray(pred, float), np.asarray(meas, float)
    ok = (p > 0) & (m > 0)
    # Three points is the minimum at which a correlation is defined at
    # all; below that np.corrcoef returns 1.0 or nan without meaning it.
    if ok.sum() < 3:
        return float("nan")
    lp, lm = np.log10(p[ok]), np.log10(m[ok])
    # A model predicting the same value for everything has zero variance
    # and no correlation — 0/0 rather than a real answer.
    if lp.std() == 0 or lm.std() == 0:
        return float("nan")
    return float(np.corrcoef(lp, lm)[0, 1])


def r_confidence(r, n):
    """95% CI on Pearson r (Fisher z). Small n gives a very wide one, and
    saying so is the point."""
    # Fisher's z transform. r is bounded at ±1 so its sampling
    # distribution is skewed near the ends; atanh maps it to something
    # roughly normal with a simple standard error, the interval is built
    # there, and tanh brings it back. |r| = 1 is excluded because atanh
    # is infinite there, and n < 4 because the standard error needs
    # n - 3 > 0.
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))


def summarise(table, measured_col, predicted_cols):
    """How each model did: scale, fold error, r and its CI."""
    out = {}
    if table.empty:
        return out
    meas = table[measured_col].to_numpy(float)
    for c in predicted_cols:
        if c not in table:
            continue
        pred = table[c].to_numpy(float)
        # Scale fitted once here and handed to fold_error, so both are
        # computed against the same constant.
        s = fit_scale(pred, meas)
        r = log_pearson(pred, meas)
        lo, hi = r_confidence(r, len(meas))
        out[c] = {"scale": s, "fold_error": fold_error(pred, meas, s),
                  "r": r, "ci": (lo, hi), "n": int(len(meas))}
    return out


def report(table, measured_col, predicted_cols, dropped=0):
    """A short text report, for the Statistics tab."""
    if table.empty:
        return "No rows with a positive measurement and prediction."
    st = summarise(table, measured_col, predicted_cols)
    n = len(table)
    lines = [f"Model against measurement — {n} point(s)", ""]
    if dropped:
        lines.append(f"{dropped} row(s) dropped (missing, zero or negative; "
                     "a log comparison cannot use them).")
        lines.append("")
    lines.append(f"{'model':28s} {'fold error':>11s} {'r (log-log)':>12s} "
                 f"{'95% CI on r':>20s}")
    for c, s in st.items():
        ci = (f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"
              if np.isfinite(s["ci"][0]) else "n too small")
        r = f"{s['r']:.3f}" if np.isfinite(s["r"]) else "-"
        lines.append(f"{c[:28]:28s} {s['fold_error']:10.2f}x {r:>12s} "
                     f"{ci:>20s}")
    lines += ["",
              "Fold error: typical factor a prediction is out by, after "
              "scaling. r: how well the model orders the constructs.",
              "A model can order well and still be inaccurate — then the "
              "shape is right and something multiplies it.",
              "Each model is scaled by one constant fitted in log space "
              "(geometric means matched), not anchored on one point."]
    # Said out loud rather than left to the reader to notice from the
    # interval: a ranking from six constructs looks authoritative in a
    # table and is not.
    if n < 8:
        lines.append(f"With only {n} points the CI on r is very wide; "
                     "treat the ranking as indicative.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _fmt_factor(v: float) -> str:
    """2.0 -> "2", 1.5 -> "1.5": no trailing zero on a whole factor."""
    return f"{v:g}"


def draw_scatter(ax, table, measured_col, predicted_cols, colors,
                 pc=None, label_col=None, annotate_max=12, log=True,
                 band=None):
    """Predicted against measured, with the 1:1 line.

    *band*, when set, shades the region within that factor of perfect
    agreement. The eye judges distance from a diagonal poorly on a log
    plot, so "is this one within twofold?" is otherwise a question you
    answer with arithmetic rather than by looking.
    """
    pc = pc or {}
    meas = table[measured_col].to_numpy(float)
    st = summarise(table, measured_col, predicted_cols)
    # Every value from every series, so the two axes can be given the
    # same range below — which is what makes the 1:1 line a diagonal.
    allv = list(meas)
    # Different marker shapes as well as colours: these figures end up
    # printed in black and white.
    markers = ["o", "^", "s", "D", "v"]
    for i, c in enumerate(predicted_cols):
        if c not in st:
            continue
        # Plotted scaled, because the raw prediction is in arbitrary
        # units and would sit nowhere near the diagonal.
        scaled = table[c].to_numpy(float) * st[c]["scale"]
        allv += list(scaled)
        col = colors[i % len(colors)]
        fe, r = st[c]["fold_error"], st[c]["r"]
        ax.scatter(meas, scaled, s=48, marker=markers[i % len(markers)],
                   facecolor="none" if i == 0 and len(predicted_cols) > 1
                   else col,
                   edgecolor=col, linewidths=1.7, zorder=3,
                   label=f"{c}  ×{fe:.1f} err"
                         + (f", r={r:.2f}" if np.isfinite(r) else ""))
    if not allv:
        return
    # One range for both axes, with a factor of 2.2 of air at each end.
    # Equal limits are what make perfect agreement a 45-degree line; with
    # independent limits the diagonal would be meaningless.
    lo, hi = min(allv) / 2.2, max(allv) * 2.2
    # Drawn before the diagonal and at zorder 0, so it sits under both
    # the 1:1 line and the points rather than washing them out.
    if band and band > 1:
        ax.fill_between([lo, hi], [lo / band, hi / band],
                        [lo * band, hi * band],
                        color=pc.get("grid", "#94A3B8"), alpha=0.18,
                        lw=0, zorder=0,
                        label=f"within {_fmt_factor(band)}×")
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0,
            color=pc.get("grid", "#94A3B8"), zorder=1)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Measured — {measured_col}")
    ax.set_ylabel("Predicted, scaled")
    # Names on the points only while there are few enough to read; past
    # a dozen they overlap into a smear.
    if label_col and label_col in table and len(table) <= annotate_max:
        first = predicted_cols[0]
        if first in st:
            ys = table[first].to_numpy(float) * st[first]["scale"]
            for xv, yv, lab in zip(meas, ys, table[label_col].astype(str)):
                ax.annotate(lab, (xv, yv), textcoords="offset points",
                            xytext=(6, 4), fontsize=7,
                            color=pc.get("text", "#334155"))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    note = "dashed: perfect agreement"
    if band and band > 1:
        # How many points the band actually holds, counted per model and
        # reported as the worst case, so the figure states the thing the
        # shading invites the reader to estimate.
        inside = min(
            int(np.sum(np.maximum(
                (table[c].to_numpy(float) * st[c]["scale"]) / meas,
                meas / (table[c].to_numpy(float) * st[c]["scale"]))
                <= band))
            for c in predicted_cols if c in st)
        note += (f"  ·  shaded: within {_fmt_factor(band)}× "
                 f"({inside}/{len(meas)} points)")
    ax.text(0.98, 0.03, note, transform=ax.transAxes,
            ha="right", fontsize=7.5, color=pc.get("text", "#64748B"),
            alpha=0.8)


def draw_bars(ax, table, measured_col, predicted_cols, colors,
              pc=None, label_col=None, log=True, measured_color="#334155"):
    """Measured and each scaled model, side by side, one group per label."""
    pc = pc or {}
    meas = table[measured_col].to_numpy(float)
    st = summarise(table, measured_col, predicted_cols)
    series = [(measured_col, meas, measured_color)]
    for i, c in enumerate(predicted_cols):
        if c in st:
            series.append((f"{c}  (×{st[c]['fold_error']:.1f} err)",
                           table[c].to_numpy(float) * st[c]["scale"],
                           colors[i % len(colors)]))
    # Grouped bars: n series share 0.8 of the slot between ticks, each
    # capped at 0.3 wide so two series do not become two fat slabs. The
    # offset (i - (n-1)/2) * w centres the group on its tick.
    n = len(series)
    x = np.arange(len(table))
    w = min(0.8 / max(n, 1), 0.3)
    for i, (name, vals, col) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=col, linewidth=0,
               label=name)
    labels = (table[label_col].astype(str).tolist()
              if label_col and label_col in table
              else [str(i + 1) for i in range(len(table))])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(table) > 4 else 0,
                       ha="right" if len(table) > 4 else "center",
                       fontsize=8.5)
    if log:
        ax.set_yscale("log")
        # Limits from the positive values only, since a log axis cannot
        # show the others. Extra room at the top for the legend.
        pos = [v for _, vals, _ in series for v in vals if v > 0]
        if pos:
            ax.set_ylim(min(pos) / 4, max(pos) * 6)
    ax.set_ylabel(f"{measured_col} (models scaled)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
