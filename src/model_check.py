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
    cols = [c for c in [measured_col, *predicted_cols] if c and c in df]
    if not label_col or label_col not in df or measured_col not in cols:
        return pd.DataFrame(), 0
    d = df[[label_col, *cols]].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    before = len(d)
    d = d.dropna(subset=cols)
    d = d[(d[cols] > MIN_POSITIVE).all(axis=1)]
    dropped = before - len(d)
    if d.empty:
        return pd.DataFrame(), dropped
    out = d.groupby(label_col, sort=False)[cols].agg(agg).reset_index()
    return out, dropped


def fit_scale(pred, meas):
    """The single factor that best lines *pred* up with *meas*, in log
    space — i.e. the ratio of their geometric means."""
    p, m = np.asarray(pred, float), np.asarray(meas, float)
    ok = (p > 0) & (m > 0)
    if not ok.any():
        return 1.0
    return float(10 ** np.mean(np.log10(m[ok]) - np.log10(p[ok])))


def fold_error(pred, meas, scale=None):
    """Typical fold error after scaling: RMS log residual, as a factor."""
    p, m = np.asarray(pred, float), np.asarray(meas, float)
    ok = (p > 0) & (m > 0)
    if ok.sum() < 1:
        return float("nan")
    s = fit_scale(p[ok], m[ok]) if scale is None else scale
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
    if ok.sum() < 3:
        return float("nan")
    lp, lm = np.log10(p[ok]), np.log10(m[ok])
    if lp.std() == 0 or lm.std() == 0:
        return float("nan")
    return float(np.corrcoef(lp, lm)[0, 1])


def r_confidence(r, n):
    """95% CI on Pearson r (Fisher z). Small n gives a very wide one, and
    saying so is the point."""
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
    if n < 8:
        lines.append(f"With only {n} points the CI on r is very wide; "
                     "treat the ranking as indicative.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_scatter(ax, table, measured_col, predicted_cols, colors,
                 pc=None, label_col=None, annotate_max=12, log=True):
    """Predicted against measured, with the 1:1 line."""
    pc = pc or {}
    meas = table[measured_col].to_numpy(float)
    st = summarise(table, measured_col, predicted_cols)
    allv = list(meas)
    markers = ["o", "^", "s", "D", "v"]
    for i, c in enumerate(predicted_cols):
        if c not in st:
            continue
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
    lo, hi = min(allv) / 2.2, max(allv) * 2.2
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0,
            color=pc.get("grid", "#94A3B8"), zorder=1)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Measured — {measured_col}")
    ax.set_ylabel("Predicted, scaled")
    if label_col and label_col in table and len(table) <= annotate_max:
        first = predicted_cols[0]
        if first in st:
            ys = table[first].to_numpy(float) * st[first]["scale"]
            for xv, yv, lab in zip(meas, ys, table[label_col].astype(str)):
                ax.annotate(lab, (xv, yv), textcoords="offset points",
                            xytext=(6, 4), fontsize=7,
                            color=pc.get("text", "#334155"))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.98, 0.03, "dashed: perfect agreement", transform=ax.transAxes,
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
        pos = [v for _, vals, _ in series for v in vals if v > 0]
        if pos:
            ax.set_ylim(min(pos) / 4, max(pos) * 6)
    ax.set_ylabel(f"{measured_col} (models scaled)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
