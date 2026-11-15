# src/curves.py
"""Curve fitting: growth curves (logistic, as in growthcurver; and a
model-free maximum growth rate) and dose-response (4-parameter logistic).

fit_groups() fits one curve per group (e.g. per well, or per strain) and
returns a table of parameters — one row per curve — that can be compared
between strains like any other measurement.
"""
# ---------------------------------------------------------------------------
# THE SHAPE OF THIS MODULE
# ---------------------------------------------------------------------------
# Three fitters, one dispatcher.
#
#   fit_growth()     logistic curve, as growthcurver fits it
#   fit_max_slope()  no model at all: the steepest straight line through
#                    ln(OD), which is what a microbiologist means by
#                    "maximum growth rate"
#   fit_dose()       4-parameter logistic, for EC50
#
# All three share one contract, and fit_groups() depends on it:
#
#     params, fn = fitter(x, y)
#
#   params  a dict of named parameters, every key present even when the
#           fit failed, with NaN for the values and a "Note" saying why
#   fn      a callable for drawing the curve, or None if it failed
#
# Keeping the keys present on failure is what lets fit_groups() build a
# rectangular table out of a mix of good and bad fits — a well that did
# not grow becomes a row of NaN with a reason, not a missing row.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

GROWTH = "Growth curve (logistic)"
MAX_SLOPE = "Growth rate (max slope of ln OD)"
DOSE = "Dose-response (4PL)"
MODELS = [GROWTH, MAX_SLOPE, DOSE]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def logistic(t, K, N0, r):
    """Logistic growth: starts at N0, saturates at K, rate r.

    The classic sigmoid, written in the carrying-capacity form so the
    parameters are the three things a microbiologist wants to read off:
    final density, starting density, and growth rate.
    """
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def four_pl(x, bottom, top, log_ec50, hill):
    """4-parameter logistic dose-response.

    Fitted in log10 of the EC50, not the EC50 itself. Concentrations
    span orders of magnitude, so the optimiser behaves far better on the
    log scale — and the confidence interval comes out symmetric there,
    which is the form it is actually symmetric in.
    """
    return bottom + (top - bottom) / (1 + 10 ** ((log_ec50 - np.log10(x)) * hill))


def _r2(y, fit):
    """Coefficient of determination, guarding the flat-data case.

    ss_tot is zero when every y is identical; the ratio would be 0/0, so
    NaN is returned instead of a spurious R² of 1.
    """
    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_growth(t, y):
    """Logistic fit. Returns a dict of parameters (NaN when it fails)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    out = {"K (max)": np.nan, "N0 (start)": np.nan, "r (growth rate, /h)": np.nan,
           "Doubling time (h)": np.nan, "Lag time (h)": np.nan,
           "Time of max growth (h)": np.nan, "R²": np.nan, "n points": len(t)}
    # Four points for three parameters, and some actual change in y.
    # np.ptp is peak-to-peak (max - min): zero means a flat well, which
    # the optimiser would chase forever.
    if len(t) < 4 or np.ptp(y) <= 0:
        out["Note"] = "too few points or no growth"
        return out, None
    # Starting guesses, from the data rather than fixed. curve_fit is a
    # local optimiser: from a poor start on a sigmoid it converges to
    # nonsense or not at all.
    #   K0  the highest reading seen
    #   N00 the reading at the earliest timepoint, floored so it cannot
    #       be zero — N0 appears in a denominator in logistic()
    #   r0  from the maximum slope of a logistic, which is r*K/4,
    #       rearranged for r
    K0 = max(y.max(), 1e-9)
    N00 = max(y[np.argmin(t)], K0 * 1e-3)
    r0 = 4 * (y.max() - y.min()) / max(np.ptp(t), 1e-9) / K0
    try:
        # Bounds keep all three positive: a negative K or r fits some
        # noisy wells better while being biologically meaningless.
        p, _ = curve_fit(logistic, t, y, p0=[K0, N00, max(r0, 1e-3)],
                         bounds=([1e-12, 1e-12, 1e-9], [np.inf, np.inf, np.inf]),
                         maxfev=20000)
    except (RuntimeError, ValueError):
        out["Note"] = "fit did not converge"
        return out, None
    K, N0, r = p
    # The derived quantities, all from the closed form of the logistic:
    #   t_mid  the inflection, where growth is fastest (y = K/2)
    #   slope  the growth rate there, r*K/4
    #   lag    where the tangent at the inflection crosses the starting
    #          level — the standard geometric definition of lag phase
    t_mid = np.log(max((K - N0) / N0, 1e-12)) / r
    slope = r * K / 4
    lag = t_mid - (K / 2 - N0) / slope
    out.update({"K (max)": K, "N0 (start)": N0, "r (growth rate, /h)": r,
                # Lag clamped at zero: the tangent construction can put
                # it slightly negative on a culture already growing at
                # t=0, and a negative lag has no meaning.
                "Doubling time (h)": np.log(2) / r, "Lag time (h)": max(lag, 0.0),
                "Time of max growth (h)": t_mid, "R²": _r2(y, logistic(t, *p))})
    return out, lambda tt: logistic(np.asarray(tt, float), *p)


def fit_max_slope(t, y, window=3):
    """Model-free: the steepest straight line through *window* consecutive
    points of ln(y) (y > 0). µmax per hour, its doubling time, and the lag
    (where that line crosses the starting level)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[ok], y[ok]
    # Sorted by time, because the window below slides over *consecutive*
    # points and the rows may arrive in any order. Logged here once, so
    # the regression is on ln(OD) — a straight line in ln(OD) against
    # time is exponential growth, and its slope is µ.
    order = np.argsort(t)
    t, ly = t[order], np.log(y[order])
    out = {"µmax (/h)": np.nan, "Doubling time (h)": np.nan, "Lag time (h)": np.nan,
           "n points": len(t)}
    if len(t) < window:
        out["Note"] = f"needs at least {window} points above 0"
        return out, None
    # Slide a window of *window* points and keep the steepest fit. This
    # is the model-free alternative to the logistic: it makes no
    # assumption about the shape of the curve, so it still gives a
    # sensible rate for a culture that diauxies or never saturates.
    best = None
    for i in range(len(t) - window + 1):
        res = stats.linregress(t[i:i + window], ly[i:i + window])
        if best is None or res.slope > best[0].slope:
            best = (res, i)
    res, i = best
    mu = res.slope
    lag = (ly[0] - res.intercept) / mu if mu > 0 else np.nan
    out.update({"µmax (/h)": mu, "Doubling time (h)": np.log(2) / mu if mu > 0 else np.nan,
                "Lag time (h)": lag})
    tt0, tt1 = t[i], t[i + window - 1]
    # The drawing function returns NaN outside the fitted window, so the
    # tangent is drawn only across the points it was fitted to rather
    # than extrapolated across the whole plot. Defaults bind the current
    # values into the lambda, as everywhere else in this codebase.
    return out, (lambda tt, a=res.intercept, b=mu, lo=tt0, hi=tt1:
                 np.where((np.asarray(tt) >= lo) & (np.asarray(tt) <= hi),
                          np.exp(a + b * np.asarray(tt)), np.nan))


def fit_dose(x, y):
    """4-parameter logistic on x > 0. EC50 with a 95% CI (from the fit's
    standard error of log EC50)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    out = {"EC50": np.nan, "EC50 95% CI low": np.nan, "EC50 95% CI high": np.nan,
           "Hill slope": np.nan, "Bottom": np.nan, "Top": np.nan, "R²": np.nan,
           "n points": len(x)}
    # Four parameters need at least four distinct concentrations, or the
    # curve passes exactly through the points and means nothing.
    if len(x) < 5 or len(np.unique(x)) < 4:
        out["Note"] = "needs at least 4 concentrations (> 0) and 5 points"
        return out, None
    lx = np.log10(x)
    # Starting guess. The Hill slope's *sign* is taken from the
    # correlation between dose and response, so an inhibition curve
    # starts searching downward instead of fighting its way there.
    p0 = [y.min(), y.max(), np.median(lx), 1.0 if np.corrcoef(lx, y)[0, 1] >= 0 else -1.0]
    try:
        p, cov = curve_fit(four_pl, x, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        out["Note"] = "fit did not converge"
        return out, None
    bottom, top, le, hill = p
    # The covariance matrix's diagonal holds the parameter variances;
    # index 2 is log_ec50. An infinite covariance means the fit was not
    # identifiable, so the interval is NaN rather than invented.
    se = np.sqrt(np.diag(cov))[2] if np.all(np.isfinite(cov)) else np.nan
    # t rather than normal, with n - 4 degrees of freedom for the four
    # fitted parameters; 0.975 for a two-sided 95% interval.
    tcrit = stats.t.ppf(0.975, max(len(x) - 4, 1))
    # Interval computed in log space, then raised — which is why it is
    # asymmetric around the EC50, as a concentration interval should be.
    out.update({"EC50": 10 ** le, "EC50 95% CI low": 10 ** (le - tcrit * se),
                "EC50 95% CI high": 10 ** (le + tcrit * se), "Hill slope": hill,
                "Bottom": bottom, "Top": top, "R²": _r2(y, four_pl(x, *p))})
    return out, lambda xx: four_pl(np.asarray(xx, float), *p)


FITTERS = {GROWTH: fit_growth, MAX_SLOPE: fit_max_slope, DOSE: fit_dose}


def fit_groups(df, x_col, y_col, by, model, keep=()):
    """One fit per combination of the *by* columns. Returns (parameters
    table, {group key: (x, y, fitted function)}). *keep*: other columns
    copied into the table when they are constant within a group (e.g. the
    Strain of a well)."""
    # Coerce rather than assume: these columns come from a spreadsheet
    # and may hold text. Non-numbers become NaN and are dropped below.
    d = df.copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    by = [b for b in by if b in d.columns]
    # sort=False keeps the groups in the order they appear, so the table
    # matches the plate. The else branch makes the no-grouping case look
    # like a single group with an empty key, so the loop below needs no
    # special case.
    groups = d.groupby(by, sort=False) if by else [((), d)]
    rows, curves = [], {}
    fitter = FITTERS[model]
    for key, g in groups:
        # pandas hands back a bare value when grouping by one column and
        # a tuple when grouping by several; normalise to a tuple.
        key = key if isinstance(key, tuple) else (key,)
        params, fn = fitter(g[x_col].to_numpy(), g[y_col].to_numpy())
        row = dict(zip(by, key))
        for k in keep:
            # Only carried across when constant within the group: the
            # Strain of a well is, so it travels with the fit; the OD at
            # each timepoint is not, and copying one of them would be
            # arbitrary.
            if k in g.columns and k not in row and g[k].nunique(dropna=False) == 1:
                row[k] = g[k].iloc[0]
        row.update(params)
        rows.append(row)
        curves[key] = (g[x_col].to_numpy(), g[y_col].to_numpy(), fn)
    return pd.DataFrame(rows), curves
