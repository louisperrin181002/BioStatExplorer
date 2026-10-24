# src/curves.py
"""Curve fitting: growth curves (logistic, as in growthcurver; and a
model-free maximum growth rate) and dose-response (4-parameter logistic).

fit_groups() fits one curve per group (e.g. per well, or per strain) and
returns a table of parameters — one row per curve — that can be compared
between strains like any other measurement.
"""
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
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def four_pl(x, bottom, top, log_ec50, hill):
    return bottom + (top - bottom) / (1 + 10 ** ((log_ec50 - np.log10(x)) * hill))


def _r2(y, fit):
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
    if len(t) < 4 or np.ptp(y) <= 0:
        out["Note"] = "too few points or no growth"
        return out, None
    K0 = max(y.max(), 1e-9)
    N00 = max(y[np.argmin(t)], K0 * 1e-3)
    r0 = 4 * (y.max() - y.min()) / max(np.ptp(t), 1e-9) / K0
    try:
        p, _ = curve_fit(logistic, t, y, p0=[K0, N00, max(r0, 1e-3)],
                         bounds=([1e-12, 1e-12, 1e-9], [np.inf, np.inf, np.inf]),
                         maxfev=20000)
    except (RuntimeError, ValueError):
        out["Note"] = "fit did not converge"
        return out, None
    K, N0, r = p
    t_mid = np.log(max((K - N0) / N0, 1e-12)) / r
    slope = r * K / 4
    lag = t_mid - (K / 2 - N0) / slope
    out.update({"K (max)": K, "N0 (start)": N0, "r (growth rate, /h)": r,
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
    order = np.argsort(t)
    t, ly = t[order], np.log(y[order])
    out = {"µmax (/h)": np.nan, "Doubling time (h)": np.nan, "Lag time (h)": np.nan,
           "n points": len(t)}
    if len(t) < window:
        out["Note"] = f"needs at least {window} points above 0"
        return out, None
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
    if len(x) < 5 or len(np.unique(x)) < 4:
        out["Note"] = "needs at least 4 concentrations (> 0) and 5 points"
        return out, None
    lx = np.log10(x)
    p0 = [y.min(), y.max(), np.median(lx), 1.0 if np.corrcoef(lx, y)[0, 1] >= 0 else -1.0]
    try:
        p, cov = curve_fit(four_pl, x, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        out["Note"] = "fit did not converge"
        return out, None
    bottom, top, le, hill = p
    se = np.sqrt(np.diag(cov))[2] if np.all(np.isfinite(cov)) else np.nan
    tcrit = stats.t.ppf(0.975, max(len(x) - 4, 1))
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
    d = df.copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    by = [b for b in by if b in d.columns]
    groups = d.groupby(by, sort=False) if by else [((), d)]
    rows, curves = [], {}
    fitter = FITTERS[model]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        params, fn = fitter(g[x_col].to_numpy(), g[y_col].to_numpy())
        row = dict(zip(by, key))
        for k in keep:
            if k in g.columns and k not in row and g[k].nunique(dropna=False) == 1:
                row[k] = g[k].iloc[0]
        row.update(params)
        rows.append(row)
        curves[key] = (g[x_col].to_numpy(), g[y_col].to_numpy(), fn)
    return pd.DataFrame(rows), curves
