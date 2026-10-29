# src/statistics_engine.py
"""
Group comparisons for the Statistical Analysis tab *and* the significance
stars on plots — both use compare(), so the plot can never show a result
the report doesn't.

Tests (all two-sided, alpha 0.05):
  2 groups      Student's t, Welch's t, Mann-Whitney U
                paired: paired t, Wilcoxon signed-rank
  > 2 groups    one-way ANOVA + Tukey (all pairs) or Dunnett (vs reference)
                Welch ANOVA + Games-Howell
                Kruskal-Wallis + Dunn's (Bonferroni over the comparisons made)
                paired: repeated-measures ANOVA / Friedman + pairwise paired
                tests, Holm-corrected
A two-group test forced onto > 2 groups runs on each pair, Holm-corrected.

Checked against scipy, statsmodels, pingouin and scikit-posthocs
(see tests/test_statistics.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except ImportError:                                   # pragma: no cover
    _HAS_STATSMODELS = False

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_p(p) -> str:
    """GraphPad style: 'p < 0.0001', else 4 decimals."""
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p < 0.0001:
        return "p < 0.0001"
    return f"p = {p:.4f}"


def stars(p) -> str:
    """'****' p < 0.0001, '***' < 0.001, '**' < 0.01, '*' < 0.05, else 'ns'."""
    if p is None or not np.isfinite(p):
        return ""
    for cut, s in ((0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < cut:
            return s
    return "ns"


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same as statsmodels 'holm')."""
    p = np.asarray(pvalues, float)
    if p.size == 0:
        return []
    if _HAS_STATSMODELS:
        return list(multipletests(p, method="holm")[1])
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return list(adj)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Comparison:
    a: str
    b: str
    p: float                    # adjusted p-value (the one to report)
    p_raw: float | None = None  # before multiple-comparison correction
    diff: float | None = None   # mean(a) - mean(b) (median for rank tests)
    ci: tuple | None = None     # 95% CI of diff, when the test gives one
    effect: str = ""            # e.g. "Cohen's d = 0.81"
    n_a: int = 0
    n_b: int = 0
    within: str | None = None   # x group, when comparing Color-By levels

    @property
    def significant(self) -> bool:
        return self.p is not None and np.isfinite(self.p) and self.p < ALPHA


@dataclass
class Result:
    test: str = ""                       # name of the test that was run
    omnibus: str = ""                    # e.g. "F(2, 9) = 13.40, p = 0.0020"
    omnibus_p: float | None = None
    effect: str = ""                     # omnibus effect size
    posthoc: str = ""                    # e.g. "Tukey HSD (all pairs)"
    comparisons: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    ns: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    error: str | None = None
    table: list = field(default_factory=list)    # two-way ANOVA table


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StatisticsEngine:
    # Test names, exposed to the UI dropdown
    AUTO = "Auto (from assumption checks)"
    STUDENTS_T = "Student's t-test"
    WELCH_T = "Welch's t-test"
    MANN_WHITNEY = "Mann-Whitney U"
    ANOVA_TUKEY = "One-way ANOVA + Tukey / Dunnett"
    WELCH_ANOVA = "Welch ANOVA + Games-Howell"
    KRUSKAL_DUNN = "Kruskal-Wallis + Dunn's"
    PAIRED_T = "Paired t-test / RM-ANOVA"
    WILCOXON = "Wilcoxon / Friedman (paired)"
    TWO_WAY = "Two-way ANOVA (X × Color by)"
    NONE = "None"

    ALL_TESTS = [AUTO, STUDENTS_T, WELCH_T, MANN_WHITNEY, ANOVA_TUKEY,
                 WELCH_ANOVA, KRUSKAL_DUNN, PAIRED_T, WILCOXON, TWO_WAY, NONE]
    PAIRED = (PAIRED_T, WILCOXON)
    TWO_GROUP = (STUDENTS_T, WELCH_T, MANN_WHITNEY)
    # names used by older versions of the app
    _ALIASES = {"Auto (Prism Mode)": AUTO,
                "One-Way ANOVA + Tukey": ANOVA_TUKEY,
                "Paired t-test": PAIRED_T,
                "Wilcoxon signed-rank": WILCOXON}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compare(df, x_col, y_col, test=None, reference=None, pairing_col=None,
                group_order=None) -> Result:
        """Compare the *y_col* values of the *x_col* groups. *reference*:
        compare every group with this one only. *pairing_col*: the column
        that matches samples across groups (paired tests)."""
        SE = StatisticsEngine
        test = SE._ALIASES.get(test, test) or SE.AUTO
        res = Result()
        if df is None or df.empty or not x_col or not y_col \
                or x_col not in df or y_col not in df:
            res.error = "No data to compare."
            return res
        cols = [x_col, y_col] + ([pairing_col] if pairing_col
                                 and pairing_col in df else [])
        d = df[cols].copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        d[x_col] = d[x_col].astype(str)
        names = list(dict.fromkeys(d[x_col]))
        if group_order:
            order = [str(g) for g in group_order if str(g) in names]
            names = order + [g for g in names if g not in order]
        res.groups = names
        res.ns = {g: int((d[x_col] == g).sum()) for g in names}
        if len(names) < 2:
            res.error = "Need at least 2 groups with values to compare."
            return res
        if reference is not None and str(reference) not in names:
            reference = None
        reference = None if reference is None else str(reference)

        if test in SE.PAIRED:
            return SE._paired(d, x_col, y_col, names, test, reference,
                              pairing_col, res)
        if test == SE.TWO_WAY:
            res.error = ("Two-way ANOVA needs a second factor: choose a "
                         "'Color by' column.")
            return res

        groups = {g: d.loc[d[x_col] == g, y_col].to_numpy(float)
                  for g in names}
        tiny = [g for g in names if len(groups[g]) < 2]
        if tiny:
            res.notes.append("Left out (fewer than 2 values): "
                             + ", ".join(tiny))
            names = [g for g in names if g not in tiny]
            groups = {g: groups[g] for g in names}
            if reference in tiny:
                reference = None
            if len(names) < 2:
                res.error = ("Need at least 2 groups with 2 or more values "
                             "each.")
                return res

        normal, equal_var = SE._assumptions(groups, res)
        k = len(names)
        if test == SE.AUTO:
            n_min = min(len(v) for v in groups.values())
            floor = SE.rank_test_p_floor(k, n_min)
            if not normal and floor > ALPHA:
                # The checks asked for a rank test that cannot return a
                # significant result at this n, whatever the data show —
                # choosing it is choosing "not significant" in advance.
                # Shapiro-Wilk has almost no power at this n either, so
                # the evidence against normality is weak. Use the Welch
                # form: valid under unequal variance and usable at small n.
                res.notes.append(
                    f"The assumption checks pointed to a rank test, but with "
                    f"{n_min} value(s) per group the smallest p it could "
                    f"return for two neighbouring groups is "
                    f"{format_p(floor)} — it cannot reach {ALPHA} however "
                    f"far apart the groups are. Auto used the Welch form "
                    f"instead (robust to unequal variance, and it works at "
                    f"this n). Choose the rank test by hand if you want it.")
                normal, equal_var = True, False
            if k == 2:
                test = (SE.MANN_WHITNEY if not normal else
                        SE.STUDENTS_T if equal_var else SE.WELCH_T)
            else:
                test = (SE.KRUSKAL_DUNN if not normal else
                        SE.ANOVA_TUKEY if equal_var else SE.WELCH_ANOVA)
            res.notes.append(
                f"Auto chose {test} from the checks above. These checks "
                "have little power with few replicates — if you know your "
                "data's distribution, choose the test yourself.")
            if test in (SE.MANN_WHITNEY, SE.KRUSKAL_DUNN) and floor > ALPHA:
                res.notes.append(
                    f"With {n_min} value(s) per group this rank test cannot "
                    f"reach p < {ALPHA} (best possible: {format_p(floor)}).")

        if test in SE.TWO_GROUP:
            pairs = SE._pairs(names, reference)
            for a, b in pairs:
                res.comparisons.append(SE._two_group(test, a, b, groups[a],
                                                     groups[b]))
            res.test = test
            if len(pairs) > 1:
                adj = holm([c.p_raw for c in res.comparisons])
                for c, p in zip(res.comparisons, adj):
                    c.p = p
                res.posthoc = (f"{test} on each pair, Holm-corrected for "
                               f"{len(pairs)} comparisons")
            return res
        if test == SE.ANOVA_TUKEY:
            return SE._anova(groups, names, reference, res)
        if test == SE.WELCH_ANOVA:
            return SE._welch_anova(groups, names, reference, res)
        if test == SE.KRUSKAL_DUNN:
            return SE._kruskal(groups, names, reference, res)
        res.error = f"Unknown test: {test}"
        return res

    @staticmethod
    def compare_within(df, x_col, hue_col, y_col, test=None, reference=None,
                       pairing_col=None, hue_order=None) -> dict:
        """Grouped plots: compare the *hue_col* levels inside each *x_col*
        group. Returns {x group: Result}; every comparison's p is then
        Holm-corrected across all x groups together. With TWO_WAY the
        comparisons come from the two-way model (Šídák)."""
        if test == StatisticsEngine.TWO_WAY:
            return StatisticsEngine.two_way(df, x_col, hue_col, y_col,
                                            hue_order)[1]
        out = {}
        for g in dict.fromkeys(df[x_col].dropna().astype(str)):
            sub = df[df[x_col].astype(str) == g]
            r = StatisticsEngine.compare(sub, hue_col, y_col, test,
                                         reference, pairing_col, hue_order)
            for c in r.comparisons:
                c.within = g
            out[g] = r
        allc = [c for r in out.values() for c in r.comparisons]
        if len(out) > 1 and allc:
            for c, p in zip(allc, holm([c.p for c in allc])):
                c.p = p
        return out

    @staticmethod
    def analyze_groups(df, x_col, y_col, test_override=None, reference=None,
                       pairing_col=None, group_order=None):
        """Report text (kept for the Statistical Analysis tab and exports)."""
        r = StatisticsEngine.compare(df, x_col, y_col, test_override,
                                     reference, pairing_col, group_order)
        return StatisticsEngine.format_report(r, x_col, y_col)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(r: Result, x_col="", y_col="", within=None) -> str:
        lines = ["=== STATISTICAL ANALYSIS REPORT ===", ""]
        if y_col:
            lines.append(f"Comparing '{y_col}' between '{x_col}' groups"
                         + (f" within '{within}'" if within else ""))
            lines.append("Groups (n): " + ", ".join(
                f"{g} ({n})" for g, n in r.ns.items()))
            lines.append("")
        if r.error:
            lines.append(r.error)
            lines += [f"Note: {n}" for n in r.notes]
            return "\n".join(lines)
        if r.assumptions:
            lines.append("1. Assumption checks")
            lines += [f"  - {a}" for a in r.assumptions]
            lines.append("")
        lines.append(("2. " if r.assumptions else "") + f"Test: {r.test}")
        if r.omnibus:
            lines.append(f"  - {r.omnibus}")
        if r.effect:
            lines.append(f"  - Effect size: {r.effect}")
        if r.comparisons:
            lines.append("")
            lines.append("Comparisons" + (f" — {r.posthoc}" if r.posthoc
                                          else "") + ":")
            for c in r.comparisons:
                parts = [f"[{c.a}] vs [{c.b}]: {format_p(c.p)} "
                         f"{stars(c.p)}"]
                if c.diff is not None:
                    d = f"difference = {c.diff:.4g}"
                    if c.ci is not None and all(np.isfinite(c.ci)):
                        d += f" (95% CI {c.ci[0]:.4g} to {c.ci[1]:.4g})"
                    parts.append(d)
                if c.effect:
                    parts.append(c.effect)
                if c.p_raw is not None and c.p_raw != c.p:
                    parts.append(f"unadjusted {format_p(c.p_raw)}")
                lines.append("  - " + "; ".join(parts))
        if r.notes:
            lines.append("")
            lines += [f"Note: {n}" for n in r.notes]
        lines.append("")
        lines.append("Significance: ns p >= 0.05, * p < 0.05, ** p < 0.01, "
                     "*** p < 0.001, **** p < 0.0001 (two-sided).")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Assumptions
    # ------------------------------------------------------------------

    @staticmethod
    def rank_test_p_floor(k: int, n_min: int) -> float:
        """The smallest p a rank test could return for two *neighbouring*
        groups, if those groups separated perfectly.

        A rank test only sees orderings, so with few replicates there are
        not enough distinct orderings to be surprising. Dunn's on three
        groups of four bottoms out at p = 0.35: no effect size, however
        large, can beat it. Auto uses this to avoid picking a test that
        has already decided the answer.

        Two groups: the exact Mann-Whitney tail. More: Dunn's normal
        approximation at perfect separation (neighbouring groups differ by
        n_min in mean rank), times the number of comparisons.
        """
        if k < 2 or n_min < 1:
            return 1.0
        if k == 2:
            return min(1.0, 2.0 / math.comb(2 * n_min, n_min))
        N = k * n_min
        se = math.sqrt(N * (N + 1) / 12.0 * (2.0 / n_min))
        p = 2 * stats.norm.sf(n_min / se)
        return float(min(1.0, p * k * (k - 1) / 2))

    @staticmethod
    def _assumptions(groups: dict, res: Result) -> tuple[bool, bool]:
        """Shapiro-Wilk per group (n >= 3) and Brown-Forsythe (Levene,
        median-centred) for equal variances. Groups too small to test are
        assumed normal (as Prism does), and said so."""
        normal = True
        for g, v in groups.items():
            if len(v) < 3:
                res.assumptions.append(
                    f"Normality [{g}]: n = {len(v)}, too few to test "
                    "(assumed normal)")
                continue
            if np.ptp(v) == 0:
                res.assumptions.append(f"Normality [{g}]: all values equal")
                continue
            p = stats.shapiro(v).pvalue
            ok = p > ALPHA
            normal &= ok
            res.assumptions.append(
                f"Normality [{g}] (Shapiro-Wilk): {format_p(p)} — "
                + ("consistent with normal" if ok else "NOT normal"))
        equal_var = True
        vals = list(groups.values())
        if any(len(v) < 3 for v in vals):
            res.assumptions.append("Equal variances: too few values to test "
                                   "(assumed equal)")
        elif any(np.ptp(v) > 0 for v in vals):
            p = stats.levene(*vals, center="median").pvalue
            equal_var = bool(p > ALPHA) if np.isfinite(p) else True
            res.assumptions.append(
                f"Equal variances (Brown-Forsythe): {format_p(p)} — "
                + ("equal" if equal_var else "UNEQUAL"))
        return normal, equal_var

    @staticmethod
    def _pairs(names, reference):
        if reference:
            return [(reference, g) for g in names if g != reference]
        return [(names[i], names[j]) for i in range(len(names))
                for j in range(i + 1, len(names))]

    # ------------------------------------------------------------------
    # Two groups
    # ------------------------------------------------------------------

    @staticmethod
    def _two_group(test, a, b, va, vb) -> Comparison:
        SE = StatisticsEngine
        c = Comparison(a, b, np.nan, n_a=len(va), n_b=len(vb))
        if test in (SE.STUDENTS_T, SE.WELCH_T):
            r = stats.ttest_ind(va, vb, equal_var=(test == SE.STUDENTS_T))
            c.p = c.p_raw = float(r.pvalue)
            c.diff = float(np.mean(va) - np.mean(vb))
            try:
                ci = r.confidence_interval()
                c.ci = (float(ci.low), float(ci.high))
            except Exception:
                pass
            d = SE.cohens_d(va, vb)
            if np.isfinite(d):
                g = d * (1 - 3 / (4 * (len(va) + len(vb)) - 9))
                c.effect = f"Cohen's d = {d:.3f}, Hedges' g = {g:.3f}"
        else:
            r = stats.mannwhitneyu(va, vb, alternative="two-sided")
            c.p = c.p_raw = float(r.pvalue)
            c.diff = float(np.median(va) - np.median(vb))
            rbc = 2 * r.statistic / (len(va) * len(vb)) - 1
            c.effect = f"U = {r.statistic:g}, rank-biserial r = {rbc:.3f}"
        return c

    @staticmethod
    def cohens_d(a, b) -> float:
        """Cohen's d with the pooled (n - 1 weighted) standard deviation."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return np.nan
        sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1)
                      * np.var(b, ddof=1)) / (na + nb - 2))
        return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan

    # ------------------------------------------------------------------
    # More than two groups
    # ------------------------------------------------------------------

    @staticmethod
    def _anova(groups, names, reference, res) -> Result:
        vals = [groups[g] for g in names]
        f = stats.f_oneway(*vals)
        n = sum(len(v) for v in vals)
        k = len(vals)
        res.test = "One-way ANOVA"
        res.omnibus = (f"F({k - 1}, {n - k}) = {f.statistic:.3f}, "
                       f"{format_p(f.pvalue)}")
        res.omnibus_p = float(f.pvalue)
        allv = np.concatenate(vals)
        ss_tot = ((allv - allv.mean()) ** 2).sum()
        ss_b = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in vals)
        if ss_tot > 0:
            res.effect = f"eta² = {ss_b / ss_tot:.3f}"
        if reference:
            others = [g for g in names if g != reference]
            # (numerical integration; fixed seed = same p every time)
            dn = stats.dunnett(*[groups[g] for g in others],
                               control=groups[reference], random_state=0)
            ci = dn.confidence_interval()
            res.posthoc = f"Dunnett's test (each group vs '{reference}')"
            for i, g in enumerate(others):
                res.comparisons.append(Comparison(
                    reference, g, float(dn.pvalue[i]),
                    diff=float(groups[reference].mean() - groups[g].mean()),
                    ci=(-float(ci.high[i]), -float(ci.low[i])),
                    n_a=len(groups[reference]), n_b=len(groups[g])))
        else:
            tk = stats.tukey_hsd(*vals)
            ci = tk.confidence_interval()
            res.posthoc = "Tukey HSD (all pairs)"
            for i in range(k):
                for j in range(i + 1, k):
                    res.comparisons.append(Comparison(
                        names[i], names[j], float(tk.pvalue[i, j]),
                        diff=float(vals[i].mean() - vals[j].mean()),
                        ci=(float(ci.low[i, j]), float(ci.high[i, j])),
                        n_a=len(vals[i]), n_b=len(vals[j])))
        return res

    @staticmethod
    def welch_anova(vals) -> tuple[float, float, float, float]:
        """Welch's one-way ANOVA: (F, df1, df2, p)."""
        n = np.array([len(v) for v in vals], float)
        m = np.array([np.mean(v) for v in vals])
        var = np.array([np.var(v, ddof=1) for v in vals])
        w = n / var
        k = len(vals)
        mw = (w * m).sum() / w.sum()
        a = (w * (m - mw) ** 2).sum() / (k - 1)
        lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
        b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
        f = a / b
        df2 = (k ** 2 - 1) / (3 * lam)
        return float(f), float(k - 1), float(df2), float(
            stats.f.sf(f, k - 1, df2))

    @staticmethod
    def games_howell(va, vb, k) -> tuple[float, float]:
        """Games-Howell p-value and mean difference for one pair."""
        na, nb = len(va), len(vb)
        sa, sb = np.var(va, ddof=1) / na, np.var(vb, ddof=1) / nb
        diff = float(np.mean(va) - np.mean(vb))
        se = np.sqrt(sa + sb)
        if se == 0:
            return (1.0 if diff == 0 else 0.0), diff
        df = (sa + sb) ** 2 / (sa ** 2 / (na - 1) + sb ** 2 / (nb - 1))
        q = abs(diff) / se * np.sqrt(2)
        return float(stats.studentized_range.sf(q, k, df)), diff

    @staticmethod
    def _welch_anova(groups, names, reference, res) -> Result:
        SE = StatisticsEngine
        vals = [groups[g] for g in names]
        if any(np.var(v, ddof=1) == 0 for v in vals):
            res.notes.append("A group has no spread; Welch ANOVA needs "
                             "variance in every group — used ordinary "
                             "ANOVA instead.")
            return SE._anova(groups, names, reference, res)
        f, df1, df2, p = SE.welch_anova(vals)
        res.test = "Welch's ANOVA (unequal variances)"
        res.omnibus = f"F({df1:g}, {df2:.2f}) = {f:.3f}, {format_p(p)}"
        res.omnibus_p = p
        k = len(vals)
        res.posthoc = ("Games-Howell" + (f" (vs '{reference}')"
                                         if reference else " (all pairs)"))
        for a, b in SE._pairs(names, reference):
            pv, diff = SE.games_howell(groups[a], groups[b], k)
            res.comparisons.append(Comparison(
                a, b, pv, diff=diff, n_a=len(groups[a]),
                n_b=len(groups[b])))
        return res

    @staticmethod
    def dunns_test(groups, group_names, p_adjust="bonferroni", pairs=None):
        """Dunn's test after Kruskal-Wallis (rank z-tests with tie
        correction). *pairs*: the comparisons to make (default: all); the
        Bonferroni correction counts only these. Returns a list of dicts
        group_a, group_b, p_value (adjusted), p_raw."""
        all_vals = np.concatenate(groups)
        N = len(all_vals)
        ranks = stats.rankdata(all_vals)
        _, counts = np.unique(all_vals, return_counts=True)
        tie = np.sum(counts ** 3 - counts) / (12.0 * (N - 1)) if N > 1 else 0
        mean_rank, idx = {}, 0
        for name, g in zip(group_names, groups):
            mean_rank[name] = (ranks[idx: idx + len(g)].mean(), len(g))
            idx += len(g)
        if pairs is None:
            pairs = [(group_names[i], group_names[j])
                     for i in range(len(group_names))
                     for j in range(i + 1, len(group_names))]
        m = len(pairs)
        out = []
        for a, b in pairs:
            (ra, na), (rb, nb) = mean_rank[a], mean_rank[b]
            se = np.sqrt((N * (N + 1) / 12.0 - tie) * (1 / na + 1 / nb))
            p = 1.0 if se == 0 else float(
                2 * stats.norm.sf(abs(ra - rb) / se))
            padj = min(p * m, 1.0) if p_adjust == "bonferroni" else p
            out.append({"group_a": a, "group_b": b, "p_value": padj,
                        "p_raw": p})
        if p_adjust == "holm":
            for r, p in zip(out, holm([r["p_raw"] for r in out])):
                r["p_value"] = p
        return out

    @staticmethod
    def _kruskal(groups, names, reference, res) -> Result:
        SE = StatisticsEngine
        vals = [groups[g] for g in names]
        allv = np.concatenate(vals)
        res.test = "Kruskal-Wallis"
        if np.ptp(allv) == 0:
            res.error = "All values are identical — nothing to rank."
            return res
        h = stats.kruskal(*vals)
        n = len(allv)
        res.omnibus = (f"H({len(vals) - 1}) = {h.statistic:.3f}, "
                       f"{format_p(h.pvalue)}")
        res.omnibus_p = float(h.pvalue)
        res.effect = f"epsilon² = {h.statistic / (n - 1):.3f}"
        pairs = SE._pairs(names, reference)
        res.posthoc = (f"Dunn's test, Bonferroni-corrected for {len(pairs)} "
                       "comparison" + ("s" if len(pairs) != 1 else ""))
        for r in SE.dunns_test(vals, names, pairs=pairs):
            a, b = r["group_a"], r["group_b"]
            res.comparisons.append(Comparison(
                a, b, r["p_value"], p_raw=r["p_raw"],
                diff=float(np.median(groups[a]) - np.median(groups[b])),
                n_a=len(groups[a]), n_b=len(groups[b])))
        return res

    # ------------------------------------------------------------------
    # Paired
    # ------------------------------------------------------------------

    @staticmethod
    def pair_values(df, x_col, y_col, group_a, group_b, pairing_col=None):
        """Matched value arrays for two groups, by *pairing_col* (e.g. a
        replicate ID shared across conditions), or by row order when there
        is none and both groups have the same size. Returns (a, b, error)."""
        wide, err = StatisticsEngine._wide(df, x_col, y_col, [str(group_a),
                                                              str(group_b)],
                                           pairing_col)
        if err:
            return None, None, err
        return (wide[str(group_a)].to_numpy(float),
                wide[str(group_b)].to_numpy(float), None)

    @staticmethod
    def _wide(d, x_col, y_col, names, pairing_col):
        """Subjects x groups table of matched values (complete subjects
        only), or (None, error)."""
        d = d.copy()
        d[x_col] = d[x_col].astype(str)
        d = d[d[x_col].isin(names)]
        if pairing_col and pairing_col in d:
            dup = d.duplicated([pairing_col, x_col], keep=False)
            if dup.any():
                row = d[dup].iloc[0]
                return None, (
                    f"'{pairing_col}' must identify one value per group, but "
                    f"'{row[pairing_col]}' has several values in "
                    f"'{row[x_col]}'. Average them first, or choose another "
                    "pairing column.")
            wide = d.pivot(index=pairing_col, columns=x_col, values=y_col)
            wide = wide.reindex(columns=names).dropna()
            if len(wide) < 2:
                return None, (f"Fewer than 2 '{pairing_col}' values have a "
                              "value in every group.")
            return wide, None
        sizes = {g: int((d[x_col] == g).sum()) for g in names}
        if len(set(sizes.values())) != 1:
            return None, ("Unequal group sizes (" + ", ".join(
                f"{g}: {n}" for g, n in sizes.items()) + ") and no pairing "
                "column — choose the column that matches samples across "
                "groups.")
        wide = pd.DataFrame({g: d.loc[d[x_col] == g, y_col].to_numpy(float)
                             for g in names})
        return wide, None

    @staticmethod
    def _paired(d, x_col, y_col, names, test, reference, pairing_col, res):
        SE = StatisticsEngine
        wide, err = SE._wide(d, x_col, y_col, names, pairing_col)
        if err:
            res.error = err
            return res
        if not pairing_col:
            res.notes.append("No pairing column: values are paired by row "
                             "order within each group.")
        n = len(wide)
        res.ns = {g: n for g in names}
        k = len(names)
        param = test == SE.PAIRED_T
        if k > 2:
            if param:
                from statsmodels.stats.anova import AnovaRM
                long = wide.reset_index(drop=True).rename_axis("_s") \
                    .reset_index().melt(id_vars="_s", var_name="_g",
                                        value_name="_v")
                t = AnovaRM(long, "_v", "_s", within=["_g"]).fit() \
                    .anova_table.iloc[0]
                res.test = "Repeated-measures one-way ANOVA"
                res.omnibus = (f"F({t['Num DF']:g}, {t['Den DF']:g}) = "
                               f"{t['F Value']:.3f}, {format_p(t['Pr > F'])}")
                res.omnibus_p = float(t["Pr > F"])
                res.notes.append("Sphericity is assumed (no Greenhouse-"
                                 "Geisser correction).")
            else:
                fr = stats.friedmanchisquare(*[wide[g] for g in names])
                res.test = "Friedman test"
                res.omnibus = (f"chi²({k - 1}) = {fr.statistic:.3f}, "
                               f"{format_p(fr.pvalue)}")
                res.omnibus_p = float(fr.pvalue)
        else:
            res.test = "Paired t-test" if param else "Wilcoxon signed-rank"
        pairs = SE._pairs(names, reference)
        for a, b in pairs:
            va, vb = wide[a].to_numpy(float), wide[b].to_numpy(float)
            c = Comparison(a, b, np.nan, n_a=n, n_b=n)
            diff = va - vb
            if param:
                r = stats.ttest_rel(va, vb)
                c.p = c.p_raw = float(r.pvalue)
                c.diff = float(diff.mean())
                try:
                    ci = r.confidence_interval()
                    c.ci = (float(ci.low), float(ci.high))
                except Exception:
                    pass
                sd = np.std(diff, ddof=1)
                if sd > 0:
                    c.effect = f"Cohen's dz = {diff.mean() / sd:.3f}"
            else:
                if np.all(diff == 0):
                    c.p = c.p_raw = 1.0
                else:
                    r = stats.wilcoxon(va, vb)
                    c.p = c.p_raw = float(r.pvalue)
                c.diff = float(np.median(diff))
            res.comparisons.append(c)
        if len(pairs) > 1:
            for c, p in zip(res.comparisons,
                            holm([c.p_raw for c in res.comparisons])):
                c.p = p
            res.posthoc = (("Paired t-tests" if param else "Wilcoxon tests")
                           + f", Holm-corrected for {len(pairs)} comparisons")
        res.notes.append(f"{n} matched sets" + (f" (by '{pairing_col}')"
                                                 if pairing_col else ""))
        return res

    # ------------------------------------------------------------------
    # Two-way ANOVA
    # ------------------------------------------------------------------

    @staticmethod
    def two_way(df, a_col, b_col, y_col, b_order=None):
        """Two-way ANOVA with interaction (type II sums of squares), then
        the *b_col* levels compared within each *a_col* level using the
        model's pooled error, Šídák-corrected over all those comparisons
        (as Prism's "compare within each row").
        Returns (table: list of dicts, {a level: Result})."""
        from statsmodels.formula.api import ols
        import statsmodels.api as sm
        d = df[[a_col, b_col, y_col]].copy()
        d.columns = ["_a", "_b", "_y"]
        d["_y"] = pd.to_numeric(d["_y"], errors="coerce")
        d = d.dropna()
        d["_a"] = d["_a"].astype(str)
        d["_b"] = d["_b"].astype(str)
        out = {}
        if d["_a"].nunique() < 2 or d["_b"].nunique() < 2:
            r = Result(error="Two-way ANOVA needs at least 2 levels of each factor.")
            return [], {"(all)": r}
        model = ols("_y ~ C(_a) * C(_b)", data=d).fit()
        tab = sm.stats.anova_lm(model, typ=2)
        ss_res = float(tab.loc["Residual", "sum_sq"])
        df_res = float(tab.loc["Residual", "df"])
        mse = ss_res / df_res
        names = {"C(_a)": a_col, "C(_b)": b_col, "C(_a):C(_b)": f"{a_col} × {b_col}"}
        table = []
        for key, label in names.items():
            ss = float(tab.loc[key, "sum_sq"])
            table.append({"Effect": label, "df": float(tab.loc[key, "df"]),
                          "F": float(tab.loc[key, "F"]), "p": float(tab.loc[key, "PR(>F)"]),
                          "partial eta²": ss / (ss + ss_res)})
        table.append({"Effect": "Residual", "df": df_res, "F": np.nan, "p": np.nan,
                      "partial eta²": np.nan})
        omnibus = "; ".join(f"{t['Effect']}: F({t['df']:g}, {df_res:g}) = {t['F']:.3f}, "
                            f"{format_p(t['p'])}" for t in table[:3])
        levels_b = list(dict.fromkeys(d["_b"]))
        if b_order:
            first = [str(b) for b in b_order if str(b) in levels_b]
            levels_b = first + [b for b in levels_b if b not in first]
        comps = []
        for a in dict.fromkeys(d["_a"]):
            sub = d[d["_a"] == a]
            r = Result(test="Two-way ANOVA", omnibus=omnibus,
                       posthoc=f"Šídák (within each {a_col})")
            r.groups = [b for b in levels_b if (sub["_b"] == b).any()]
            r.ns = {b: int((sub["_b"] == b).sum()) for b in r.groups}
            for i in range(len(r.groups)):
                for j in range(i + 1, len(r.groups)):
                    va = sub.loc[sub["_b"] == r.groups[i], "_y"].to_numpy(float)
                    vb = sub.loc[sub["_b"] == r.groups[j], "_y"].to_numpy(float)
                    diff = va.mean() - vb.mean()
                    se = np.sqrt(mse * (1 / len(va) + 1 / len(vb)))
                    t = diff / se if se > 0 else np.inf
                    p = float(2 * stats.t.sf(abs(t), df_res))
                    tcrit = stats.t.ppf(1 - ALPHA / 2, df_res)
                    c = Comparison(r.groups[i], r.groups[j], p, p_raw=p, diff=float(diff),
                                   ci=(float(diff - tcrit * se), float(diff + tcrit * se)),
                                   n_a=len(va), n_b=len(vb), within=a)
                    r.comparisons.append(c)
                    comps.append(c)
            out[a] = r
        m = len(comps)
        for c in comps:                      # Šídák over all comparisons
            c.p = float(1 - (1 - c.p_raw) ** m) if m > 1 else c.p_raw
        for r in out.values():
            r.table = table
        return table, out

    # ------------------------------------------------------------------
    # Outliers
    # ------------------------------------------------------------------

    @staticmethod
    def grubbs(values, alpha=0.05):
        """Indices of outliers by the iterative two-sided Grubbs test
        (one at a time while n >= 3)."""
        v = np.asarray(values, float)
        idx = list(range(len(v)))
        out = []
        while len(idx) >= 3:
            x = v[idx]
            sd = x.std(ddof=1)
            if sd == 0:
                break
            dev = np.abs(x - x.mean())
            k = int(np.argmax(dev))
            n = len(x)
            g = dev[k] / sd
            t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            gcrit = (n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2))
            if g <= gcrit:
                break
            out.append(idx.pop(k))
        return out

    @staticmethod
    def find_outliers(df, x_col, y_col, hue_col=None, alpha=0.05):
        """Boolean Series (df's index): Grubbs outliers within each group
        (x group, and Color-by level when given)."""
        flag = pd.Series(False, index=df.index)
        if df is None or df.empty or y_col not in df or x_col not in df:
            return flag
        y = pd.to_numeric(df[y_col], errors="coerce")
        keys = [df[x_col].astype(str)] + ([df[hue_col].astype(str)]
                                          if hue_col and hue_col in df else [])
        for _, grp in y.groupby(keys):
            grp = grp.dropna()
            for i in StatisticsEngine.grubbs(grp.to_numpy(), alpha):
                flag[grp.index[i]] = True
        return flag

    @staticmethod
    def drop_outliers(df, x_col, y_col, hue_col=None, alpha=0.05):
        if df is None or df.empty:
            return df
        return df[~StatisticsEngine.find_outliers(df, x_col, y_col, hue_col, alpha)]
