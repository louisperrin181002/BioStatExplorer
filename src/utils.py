# src/utils.py
import numpy as np
import pandas as pd
import scipy.stats as stats

class SummaryStatistics:
    @staticmethod
    def generate_table(df, x_col, y_col, order=None):
        """Computes summary statistics, SEM, and 95% CI for each group."""
        if df is None or df.empty or not x_col or not y_col:
            return pd.DataFrame()
            
        df_clean = df[[x_col, y_col]].dropna()
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
        df_clean = df_clean.dropna()
        
        summary_data = []
        groups = dict(list(df_clean.groupby(x_col, sort=False)))
        names = list(groups)
        if order:   # same order as the plot and the report
            want = [n for o in order for n in names if str(n) == str(o)]
            names = want + [n for n in names if n not in want]
        for name in names:
            group = groups[name]
            values = group[y_col].values
            n = len(values)
            if n == 0: continue
            
            mean = np.mean(values)
            median = np.median(values)
            sd = np.std(values, ddof=1) if n > 1 else 0.0
            sem = stats.sem(values) if n > 1 else 0.0
            
            # 95% Confidence Interval
            if n > 1:
                ci_hbar = sem * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
                ci_lower = mean - ci_hbar
                ci_upper = mean + ci_hbar
            else:
                ci_lower, ci_upper = mean, mean

            summary_data.append({
                "Group": name,
                "N": n,
                "Mean": round(mean, 4),
                "Median": round(median, 4),
                "SD": round(sd, 4),
                "SEM": round(sem, 4),
                "95% CI Lower": round(ci_lower, 4),
                "95% CI Upper": round(ci_upper, 4)
            })
            
        return pd.DataFrame(summary_data)
