"""Checks the statistics engine against established packages.

Run:  python tests/test_statistics.py
(needs pingouin and scikit-posthocs for the reference values; tests that
need a missing package are skipped).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.statistics_engine import StatisticsEngine as SE, format_p, stars  # noqa: E402

warnings.filterwarnings("ignore")
try:
    import pingouin as pg
except ImportError:
    pg = None
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

FAILS = []
TOL = 1e-6


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(b))


def frame(groups):
    names = [chr(65 + i) for i in range(len(groups))]
    return pd.DataFrame({"g": np.repeat(names, [len(v) for v in groups]),
                         "v": np.concatenate(groups)}), names


def by_pair(res):
    return {(c.a, c.b): c for c in res.comparisons}


def datasets(seed=0, n_sets=25):
    rng = np.random.default_rng(seed)
    for _ in range(n_sets):
        k = rng.integers(3, 6)
        yield [rng.normal(rng.normal(0, 1), rng.uniform(0.5, 2),
                          rng.integers(3, 12)) for _ in range(k)]


# --------------------------------------------------------------------------
# two groups
# --------------------------------------------------------------------------
def test_two_groups():
    rng = np.random.default_rng(1)
    worst = {"t": 0, "w": 0, "mw": 0, "d": 0}
    for _ in range(40):
        a = rng.normal(0, 1, rng.integers(3, 15))
        b = rng.normal(0.8, 1.5, rng.integers(3, 15))
        df, _ = frame([a, b])
        for test, key, ref in [
                (SE.STUDENTS_T, "t", st.ttest_ind(a, b).pvalue),
                (SE.WELCH_T, "w", st.ttest_ind(a, b, equal_var=False).pvalue),
                (SE.MANN_WHITNEY, "mw",
                 st.mannwhitneyu(a, b, alternative="two-sided").pvalue)]:
            c = SE.compare(df, "g", "v", test).comparisons[0]
            worst[key] = max(worst[key], abs(c.p - ref))
        if pg is not None:
            worst["d"] = max(worst["d"], abs(
                SE.cohens_d(a, b) - pg.compute_effsize(a, b, eftype="cohen")))
    check(worst["t"] < TOL, f"Student's t = scipy (max diff {worst['t']:.1e})")
    check(worst["w"] < TOL, f"Welch's t = scipy (max diff {worst['w']:.1e})")
    check(worst["mw"] < TOL, f"Mann-Whitney = scipy (max diff {worst['mw']:.1e})")
    if pg is not None:
        check(worst["d"] < TOL, f"Cohen's d = pingouin (max diff {worst['d']:.1e})")


# --------------------------------------------------------------------------
# more than two groups
# --------------------------------------------------------------------------
def test_anova_tukey_dunnett():
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    w_f = w_tk = w_dn = w_eta = 0
    for groups in datasets(2):
        df, names = frame(groups)
        r = SE.compare(df, "g", "v", SE.ANOVA_TUKEY)
        w_f = max(w_f, abs(r.omnibus_p - st.f_oneway(*groups).pvalue))
        sm = pairwise_tukeyhsd(df.v, df.g)
        smp = {(a, b): p for (a, b), p in zip(
            [(sm.groupsunique[i], sm.groupsunique[j])
             for i in range(len(names)) for j in range(i + 1, len(names))],
            sm.pvalues)}
        for (a, b), c in by_pair(r).items():
            w_tk = max(w_tk, abs(c.p - smp[(a, b)]))
        rr = SE.compare(df, "g", "v", SE.ANOVA_TUKEY, reference="A")
        dn = st.dunnett(*groups[1:], control=groups[0])
        for i, c in enumerate(rr.comparisons):
            w_dn = max(w_dn, abs(c.p - dn.pvalue[i]))
        if pg is not None:
            eta = pg.anova(data=df, dv="v", between="g", effsize="n2")["n2"][0]
            w_eta = max(w_eta, abs(float(r.effect.split("=")[1]) - eta))
    check(w_f < TOL, f"one-way ANOVA = scipy (max diff {w_f:.1e})")
    # statsmodels rounds Tukey p-values to ~3 decimals
    check(w_tk < 2e-3, f"Tukey = statsmodels (max diff {w_tk:.1e})")
    check(w_dn < 1e-3, f"Dunnett vs reference = scipy (max diff {w_dn:.1e})")
    if pg is not None:
        check(w_eta < 1e-3, f"eta² = pingouin (max diff {w_eta:.1e})")


def test_welch_games_howell():
    if pg is None:
        print("SKIP Welch ANOVA / Games-Howell (pingouin not installed)")
        return
    w_f = w_gh = 0
    for groups in datasets(3):
        df, names = frame(groups)
        r = SE.compare(df, "g", "v", SE.WELCH_ANOVA)
        ref = pg.welch_anova(data=df, dv="v", between="g")
        w_f = max(w_f, abs(r.omnibus_p - ref["p_unc"][0]))
        gh = pg.pairwise_gameshowell(data=df, dv="v", between="g")
        for _, row in gh.iterrows():
            c = by_pair(r)[(row["A"], row["B"])]
            w_gh = max(w_gh, abs(c.p - row["pval"]))
    check(w_f < 1e-6, f"Welch ANOVA = pingouin (max diff {w_f:.1e})")
    check(w_gh < 1e-3, f"Games-Howell = pingouin (max diff {w_gh:.1e})")


def test_kruskal_dunn():
    if sp is None:
        print("SKIP Dunn's test (scikit-posthocs not installed)")
        return
    w_h = w_all = w_ref = 0
    for groups in datasets(4):
        groups = [np.round(g, 1) for g in groups]          # with ties
        df, names = frame(groups)
        r = SE.compare(df, "g", "v", SE.KRUSKAL_DUNN)
        w_h = max(w_h, abs(r.omnibus_p - st.kruskal(*groups).pvalue))
        ref = sp.posthoc_dunn(df, val_col="v", group_col="g",
                              p_adjust="bonferroni")
        for (a, b), c in by_pair(r).items():
            w_all = max(w_all, abs(c.p - ref.loc[a, b]))
        raw = sp.posthoc_dunn(df, val_col="v", group_col="g")
        rr = SE.compare(df, "g", "v", SE.KRUSKAL_DUNN, reference="A")
        m = len(names) - 1
        for c in rr.comparisons:
            w_ref = max(w_ref, abs(c.p - min(raw.loc["A", c.b] * m, 1)))
    check(w_h < TOL, f"Kruskal-Wallis = scipy (max diff {w_h:.1e})")
    check(w_all < TOL, f"Dunn all pairs = scikit-posthocs (max diff {w_all:.1e})")
    check(w_ref < TOL, "Dunn vs reference corrects for k-1 comparisons "
          f"(max diff {w_ref:.1e})")


def test_holm_pairwise():
    from statsmodels.stats.multitest import multipletests
    groups = list(datasets(5, 1))[0]
    df, names = frame(groups)
    r = SE.compare(df, "g", "v", SE.WELCH_T)
    raw = [c.p_raw for c in r.comparisons]
    ref = multipletests(raw, method="holm")[1]
    check(np.allclose([c.p for c in r.comparisons], ref),
          "two-group test on > 2 groups is Holm-corrected")


# --------------------------------------------------------------------------
# paired
# --------------------------------------------------------------------------
def test_paired():
    rng = np.random.default_rng(6)
    subj = np.repeat(np.arange(8), 3)
    cond = np.tile(["T0", "T1", "T2"], 8)
    base = rng.normal(10, 2, 8)
    vals = base[subj] + np.tile([0, 0.8, 1.5], 8) + rng.normal(0, 0.6, 24)
    df = pd.DataFrame({"rep": subj, "cond": cond, "v": vals})
    rng.shuffle(df.values)                                 # order mustn't matter
    two = df[df.cond.isin(["T0", "T1"])]
    r = SE.compare(two, "cond", "v", SE.PAIRED_T, pairing_col="rep")
    w = df.pivot(index="rep", columns="cond", values="v")
    check(close(r.comparisons[0].p, st.ttest_rel(w.T0, w.T1).pvalue),
          "paired t = scipy (matched by pairing column)")
    r = SE.compare(two, "cond", "v", SE.WILCOXON, pairing_col="rep")
    check(close(r.comparisons[0].p, st.wilcoxon(w.T0, w.T1).pvalue),
          "Wilcoxon = scipy")
    r = SE.compare(df, "cond", "v", SE.WILCOXON, pairing_col="rep")
    check(close(r.omnibus_p, st.friedmanchisquare(w.T0, w.T1, w.T2).pvalue),
          "Friedman = scipy")
    if pg is not None:
        r = SE.compare(df, "cond", "v", SE.PAIRED_T, pairing_col="rep")
        ref = pg.rm_anova(data=df, dv="v", within="cond", subject="rep")
        check(close(r.omnibus_p, ref["p_unc"][0], 1e-6),
              "repeated-measures ANOVA = pingouin")
    dup = pd.DataFrame({"cond": list("AAAABBBB"), "rep": [1, 1, 2, 2] * 2,
                        "v": range(8)})
    r = SE.compare(dup, "cond", "v", SE.PAIRED_T, pairing_col="rep")
    check(r.error is not None and "several values" in r.error,
          "duplicated pairing IDs are refused (no fake pairs)")


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------
def test_behaviour():
    df = pd.DataFrame({"g": list("AABB"), "v": [1, 1.1, 5, 5.2]})
    r = SE.compare(df, "g", "v")
    check(r.test == SE.STUDENTS_T, "Auto with n = 2 assumes normal (t-test, "
          "not a rank test that can't reach p < 0.05)")
    check(format_p(1e-9) == "p < 0.0001" and format_p(0.01234) == "p = 0.0123",
          "p formatting")
    check([stars(p) for p in (5e-5, 5e-4, 5e-3, 0.04, 0.2)] ==
          ["****", "***", "**", "*", "ns"], "significance stars")
    df = pd.DataFrame({"g": ["A"] * 4 + ["B"] * 4 + ["C"],
                       "v": [1, 2, 3, 4, 2, 3, 4, 5, 9]})
    r = SE.compare(df, "g", "v", SE.ANOVA_TUKEY)
    check(r.groups == ["A", "B", "C"] and "C" in " ".join(r.notes)
          and len(r.comparisons) == 1, "a group with 1 value is left out, "
          "and said so")
    r = SE.compare(df, "g", "v", group_order=["C", "B", "A"])
    check(r.groups == ["C", "B", "A"], "group order follows the plot")
    nums = pd.DataFrame({"g": [1, 1, 1, 2, 2, 2], "v": [1, 2, 3, 4, 5, 6]})
    r = SE.compare(nums, "g", "v", reference=1)
    check(r.error is None and r.comparisons[0].a == "1",
          "numeric group labels work (reference given as a number)")


# --------------------------------------------------------------------------
# two-way ANOVA and outliers
# --------------------------------------------------------------------------
def test_two_way():
    rng = np.random.default_rng(9)
    rows = []
    for a in ["WT", "LP107", "LP109"]:
        for b in ["LB", "M9"]:
            n = rng.integers(3, 7)                       # unbalanced
            eff = {"WT": 0, "LP107": 1, "LP109": 2}[a] + (0.8 if b == "M9" else 0) \
                + (1.2 if (a, b) == ("LP109", "M9") else 0)
            rows += [{"strain": a, "medium": b, "v": eff + rng.normal(0, 0.7)} for _ in range(n)]
    df = pd.DataFrame(rows)
    table, res = SE.two_way(df, "strain", "medium", "v")
    if pg is not None:
        ref = pg.anova(data=df, dv="v", between=["strain", "medium"], ss_type=2)
        ok = all(close(t["p"], ref["p_unc"][i], 1e-6) for i, t in enumerate(table[:3]))
        check(ok, "two-way ANOVA (type II) = pingouin")
    # simple effect by hand: MSE from the full model, Šídák over 3 tests
    from statsmodels.formula.api import ols
    m = ols("v ~ C(strain) * C(medium)", data=df).fit()
    mse, dfr = m.mse_resid, m.df_resid
    sub = df[df.strain == "LP109"]
    a, b = sub[sub.medium == "LB"].v, sub[sub.medium == "M9"].v
    t = (a.mean() - b.mean()) / np.sqrt(mse * (1 / len(a) + 1 / len(b)))
    p = 2 * st.t.sf(abs(t), dfr)
    c = res["LP109"].comparisons[0]
    check(close(c.p_raw, p) and close(c.p, 1 - (1 - p) ** 3),
          "two-way post-hoc: pooled error, Šídák-corrected")
    check(SE.compare(df, "strain", "v", SE.TWO_WAY).error is not None,
          "two-way without a second factor explains what is missing")


def test_grubbs():
    # NIST/SEMATECH handbook example: 245.57 is an outlier (G = 2.4687)
    v = [199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57]
    check(SE.grubbs(v) == [7], "Grubbs flags the NIST example outlier")
    check(SE.grubbs(v[:-1]) == [], "no outlier left after removing it")
    df = pd.DataFrame({"g": ["A"] * 8 + ["B"] * 5, "v": v + [1, 1.1, 0.9, 1.05, 0.95]})
    flag = SE.find_outliers(df, "g", "v")
    check(list(flag[flag].index) == [7], "outliers found per group")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
