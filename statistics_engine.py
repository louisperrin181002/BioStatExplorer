# src/statistics_engine.py
import numpy as np
import pandas as pd
import scipy.stats as stats
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


class StatisticsEngine:
    # Test names, exposed to the UI dropdown
    AUTO = "Auto (Prism Mode)"
    STUDENTS_T = "Student's t-test"
    WELCH_T = "Welch's t-test"
    MANN_WHITNEY = "Mann-Whitney U"
    ANOVA_TUKEY = "One-Way ANOVA + Tukey"
    KRUSKAL_DUNN = "Kruskal-Wallis + Dunn's"
    PAIRED_T = "Paired t-test"
    WILCOXON = "Wilcoxon signed-rank"
    NONE = "None"

    ALL_TESTS = [AUTO, STUDENTS_T, WELCH_T, MANN_WHITNEY, ANOVA_TUKEY,
                 KRUSKAL_DUNN, PAIRED_T, WILCOXON, NONE]

    @staticmethod
    def dunns_test(groups, group_names, p_adjust="bonferroni"):
        """
        Manual implementation of Dunn's post-hoc test (no external dependency
        on scikit-posthocs). Returns a list of dicts: group_a, group_b, p_value.
        Standard rank-sum z-test with tie correction, used after a
        significant Kruskal-Wallis result.
        """
        all_vals = np.concatenate(groups)
        N = len(all_vals)
        ranks = stats.rankdata(all_vals)

        _, counts = np.unique(all_vals, return_counts=True)
        tie_term = np.sum(counts**3 - counts) / (12.0 * (N - 1)) if N > 1 else 0.0

        rank_groups = []
        idx = 0
        for g in groups:
            n = len(g)
            rank_groups.append(ranks[idx: idx + n])
            idx += n

        results = []
        n_comparisons = len(groups) * (len(groups) - 1) // 2
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                n_i, n_j = len(groups[i]), len(groups[j])
                mean_rank_i = np.mean(rank_groups[i])
                mean_rank_j = np.mean(rank_groups[j])
                se = np.sqrt((N * (N + 1) / 12.0 - tie_term) * (1.0 / n_i + 1.0 / n_j))
                if se == 0:
                    z, p = 0.0, 1.0
                else:
                    z = (mean_rank_i - mean_rank_j) / se
                    p = 2 * (1 - stats.norm.cdf(abs(z)))
                if p_adjust == "bonferroni":
                    p = min(p * n_comparisons, 1.0)
                results.append({"group_a": group_names[i], "group_b": group_names[j], "p_value": p})
        return results

    @staticmethod
    def pair_values(df, x_col, y_col, group_a, group_b, pairing_col=None):
        """
        Builds matched (paired) value arrays for two groups using a pairing
        column (e.g. a replicate/sample ID shared across conditions). Falls
        back to row-order pairing if no pairing column is given and both
        groups have equal size. Returns (values_a, values_b, error_message).
        """
        cols = [x_col, y_col] + ([pairing_col] if pairing_col else [])
        sub = df[cols].dropna(subset=[x_col, y_col])
        a = sub[sub[x_col] == group_a]
        b = sub[sub[x_col] == group_b]

        if pairing_col and pairing_col in sub.columns:
            merged = a.merge(b, on=pairing_col, suffixes=("_a", "_b"))
            if merged.empty:
                return None, None, (
                    f"No matching '{pairing_col}' values found between "
                    f"'{group_a}' and '{group_b}'."
                )
            return merged[f"{y_col}_a"].values, merged[f"{y_col}_b"].values, None

        if len(a) != len(b):
            return None, None, (
                f"Cannot pair '{group_a}' (n={len(a)}) with '{group_b}' (n={len(b)}) — "
                f"unequal group sizes and no pairing column selected."
            )
        return a[y_col].values, b[y_col].values, None

    @staticmethod
    def _run_pairwise_test(test_name, g1, g2):
        """Runs a single named test on two independent-sample arrays."""
        if test_name == StatisticsEngine.STUDENTS_T:
            _, p = stats.ttest_ind(g1, g2, equal_var=True)
        elif test_name == StatisticsEngine.WELCH_T:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)
        elif test_name == StatisticsEngine.MANN_WHITNEY:
            _, p = stats.mannwhitneyu(g1, g2)
        else:
            return None
        return p

    @staticmethod
    def analyze_groups(df, x_col, y_col, test_override=None, reference=None, pairing_col=None):
        """
        Prism-style automated recommendation pipeline.
        Checks normality, variance, runs the correct test, and computes effect size.

        test_override: force a specific test name (see class constants) instead
            of the automatic normality/variance-driven pick. AUTO or None keeps
            the original automatic behavior.
        reference: if provided (and there are >2 groups, or a specific pairwise
            posthoc test is used), comparisons are reported as "each group vs
            reference" rather than every pairwise combination.
        pairing_col: optional column used to match samples for paired tests.
        """
        if df is None or df.empty or not x_col or not y_col:
            return "No data available for statistics."

        test_override = test_override or StatisticsEngine.AUTO

        # Clean data and group. Keep the pairing column around if a paired
        # test was requested, since pair_values() needs it to match rows.
        base_cols = [x_col, y_col]
        if test_override in (StatisticsEngine.PAIRED_T, StatisticsEngine.WILCOXON) and pairing_col:
            base_cols = base_cols + [pairing_col]
        df_clean = df[base_cols].dropna(subset=[x_col, y_col])
        group_names = list(df_clean[x_col].unique())
        groups = [df_clean[df_clean[x_col] == g][y_col].values for g in group_names]

        if len(groups) < 2:
            return "Need at least 2 distinct groups to run statistical comparisons."

        if reference is not None and reference not in group_names:
            reference = None  # ignore an invalid/stale reference selection

        report = "=== STATISTICAL ANALYSIS REPORT ===\n\n"

        # --- Paired tests take a completely different path (no normality/variance gate) ---
        if test_override in (StatisticsEngine.PAIRED_T, StatisticsEngine.WILCOXON):
            report += f"Test selected: {test_override} (paired comparison)\n"
            if pairing_col:
                report += f"Pairing column: {pairing_col}\n\n"
            else:
                report += "Pairing column: none provided — pairing by row order (requires equal group sizes)\n\n"

            targets = [reference] if reference else None
            pairs_done = 0
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    ga, gb = group_names[i], group_names[j]
                    if targets and reference not in (ga, gb):
                        continue
                    v1, v2, err = StatisticsEngine.pair_values(df_clean, x_col, y_col, ga, gb, pairing_col)
                    if err:
                        report += f"  - [{ga}] vs [{gb}]: {err}\n"
                        continue
                    if test_override == StatisticsEngine.PAIRED_T:
                        _, p = stats.ttest_rel(v1, v2)
                    else:
                        try:
                            _, p = stats.wilcoxon(v1, v2)
                        except ValueError as e:
                            report += f"  - [{ga}] vs [{gb}]: {e}\n"
                            continue
                    report += f"  - [{ga}] vs [{gb}] (n={len(v1)} pairs): p = {p:.5f}\n"
                    pairs_done += 1
            if pairs_done == 0:
                report += "  (No valid pairs could be formed.)\n"
            return report

        # 1. Normality Test (Shapiro-Wilk)
        all_normal = True
        report += "1. Normality Check (Shapiro-Wilk):\n"
        for name, grp in zip(group_names, groups):
            if len(grp) >= 3:
                stat, p = stats.shapiro(grp)
                status = "Normal" if p > 0.05 else "Non-Normal"
                report += f"  - Group [{name}]: p = {p:.4f} ({status})\n"
                if p <= 0.05: all_normal = False
            else:
                report += f"  - Group [{name}]: Too few replicates (n < 3) to test normality.\n"
                all_normal = False

        # 2. Variance Check (Levene's Test)
        stat, p_levene = stats.levene(*groups)
        equal_var = p_levene > 0.05
        report += f"\n2. Variance Homogeneity (Levene's Test):\n  - p = {p_levene:.4f} "
        report += " (Equal Variance)\n" if equal_var else " (Unequal Variance)\n"

        # 3. Test Execution
        report += "\n3. Executing Test:\n"

        if len(groups) == 2 and test_override in (
            StatisticsEngine.AUTO, StatisticsEngine.STUDENTS_T,
            StatisticsEngine.WELCH_T, StatisticsEngine.MANN_WHITNEY
        ):
            g1, g2 = groups[0], groups[1]
            chosen = test_override
            if chosen == StatisticsEngine.AUTO:
                if all_normal and equal_var:
                    chosen = StatisticsEngine.STUDENTS_T
                elif all_normal and not equal_var:
                    chosen = StatisticsEngine.WELCH_T
                else:
                    chosen = StatisticsEngine.MANN_WHITNEY

            p = StatisticsEngine._run_pairwise_test(chosen, g1, g2)
            report += f"  - Test used: {chosen}\n  - p-value: {p:.5f}\n"
            if chosen == StatisticsEngine.STUDENTS_T:
                cohens_d = (np.mean(g1) - np.mean(g2)) / np.sqrt((np.var(g1) + np.var(g2)) / 2)
                report += f"  - Effect Size (Cohen's d): {cohens_d:.4f}\n"

        else:
            # Multiple-group (or explicitly forced omnibus) tests
            use_parametric = test_override == StatisticsEngine.ANOVA_TUKEY or (
                test_override == StatisticsEngine.AUTO and all_normal and equal_var
            )
            use_nonparametric = test_override == StatisticsEngine.KRUSKAL_DUNN or (
                test_override == StatisticsEngine.AUTO and not (all_normal and equal_var)
            )

            if use_parametric:
                stat, p = stats.f_oneway(*groups)
                report += f"  - Omnibus test: One-Way ANOVA\n  - F-statistic: {stat:.4f}, p-value: {p:.5f}\n"

                if _HAS_STATSMODELS:
                    formula = f"Q('{y_col}') ~ C(Q('{x_col}'))"
                    model = ols(formula, data=df_clean).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    es = anova_table['sum_sq'].iloc[0] / (anova_table['sum_sq'].iloc[0] + anova_table['sum_sq'].iloc[1])
                    report += f"  - Effect Size (Eta-Squared eta^2): {es:.4f}\n"
                else:
                    report += "  - (Install statsmodels to also see Eta-Squared effect size)\n"

                report += "\n  Post-hoc: Tukey HSD" + (f" (vs reference '{reference}')" if reference else " (all pairs)") + ":\n"
                tukey_res = stats.tukey_hsd(*groups)
                for i in range(len(group_names)):
                    for j in range(i + 1, len(group_names)):
                        g1n, g2n = group_names[i], group_names[j]
                        if reference and reference not in (g1n, g2n):
                            continue
                        p_adj = tukey_res.pvalue[i, j]
                        sig = "significant" if p_adj < 0.05 else "not significant"
                        report += f"    - [{g1n}] vs [{g2n}]: p-adj = {p_adj:.5f} ({sig})\n"

            elif use_nonparametric:
                stat, p = stats.kruskal(*groups)
                report += f"  - Omnibus test: Kruskal-Wallis\n  - H-statistic: {stat:.4f}, p-value: {p:.5f}\n"

                report += "\n  Post-hoc: Dunn's test" + (f" (vs reference '{reference}')" if reference else " (all pairs)") + ", Bonferroni-corrected:\n"
                dunn_results = StatisticsEngine.dunns_test(groups, group_names)
                for r in dunn_results:
                    if reference and reference not in (r["group_a"], r["group_b"]):
                        continue
                    sig = "significant" if r["p_value"] < 0.05 else "not significant"
                    report += f"    - [{r['group_a']}] vs [{r['group_b']}]: p-adj = {r['p_value']:.5f} ({sig})\n"
            else:
                # Explicit two-sample test name forced onto a >2 group dataset:
                # fall back to reporting it against the reference (or all pairs).
                report += f"  - Running {test_override} across all group pairs" + (f" vs reference '{reference}'" if reference else "") + ":\n"
                for i in range(len(group_names)):
                    for j in range(i + 1, len(group_names)):
                        if reference and reference not in (group_names[i], group_names[j]):
                            continue
                        p = StatisticsEngine._run_pairwise_test(test_override, groups[i], groups[j])
                        if p is None:
                            continue
                        report += f"    - [{group_names[i]}] vs [{group_names[j]}]: p = {p:.5f}\n"

        return report
