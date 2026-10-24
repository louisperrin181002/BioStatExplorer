"""Curve fits recover known parameters.  Run: python tests/test_curves.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import curves  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


rng = np.random.default_rng(4)
t = np.arange(0, 24.5, 1.0)
y = curves.logistic(t, 1.2, 0.02, 0.6) * (1 + rng.normal(0, 0.01, len(t)))
p, fn = curves.fit_growth(t, y)
check(abs(p["K (max)"] - 1.2) < 0.03 and abs(p["r (growth rate, /h)"] - 0.6) < 0.03,
      f"logistic: K {p['K (max)']:.3f} (1.2), r {p['r (growth rate, /h)']:.3f} (0.6)")
check(abs(p["Doubling time (h)"] - np.log(2) / 0.6) < 0.06 and p["R²"] > 0.99,
      f"doubling time {p['Doubling time (h)']:.2f} h, R² {p['R²']:.4f}")
# exponential phase: ln y slope = mu
te = np.arange(0, 6, 0.5)
ye = 0.01 * np.exp(0.8 * te)
p, _ = curves.fit_max_slope(np.r_[te, 6.5, 7], np.r_[ye, ye[-1] * 1.05, ye[-1] * 1.06])
check(abs(p["µmax (/h)"] - 0.8) < 1e-6, f"max slope µ = {p['µmax (/h)']:.4f} (0.8)")
x = np.logspace(-3, 2, 11)
yd = curves.four_pl(x, 5, 105, np.log10(0.7), 1.3) + rng.normal(0, 1.5, len(x))
p, _ = curves.fit_dose(x, yd)
check(abs(np.log10(p["EC50"]) - np.log10(0.7)) < 0.08 and p["EC50 95% CI low"] < 0.7 < p["EC50 95% CI high"],
      f"4PL EC50 {p['EC50']:.3f} (0.7), CI {p['EC50 95% CI low']:.3f}–{p['EC50 95% CI high']:.3f}")
check(abs(p["Hill slope"] - 1.3) < 0.3, f"Hill slope {p['Hill slope']:.2f} (1.3)")
# per-well fits keep the strain
rows = []
for well, strain, r in [("A1", "WT", 0.5), ("A2", "WT", 0.52), ("B1", "LP107", 0.3), ("B2", "LP107", 0.31)]:
    for ti in t:
        rows.append({"Well": well, "Strain": strain, "Time (h)": ti, "OD": curves.logistic(ti, 1.0, 0.02, r)})
tab, cur = curves.fit_groups(pd.DataFrame(rows), "Time (h)", "OD", ["Well"], curves.GROWTH, keep=["Strain"])
check(list(tab["Strain"]) == ["WT", "WT", "LP107", "LP107"] and np.allclose(tab["r (growth rate, /h)"], [0.5, 0.52, 0.3, 0.31], atol=1e-3),
      "one fit per well, strain kept for comparing strains")
check(curves.fit_growth([0, 1], [0.1, 0.2])[0]["Note"].startswith("too few"), "too few points: explained, no crash")
print(f"\n{len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
