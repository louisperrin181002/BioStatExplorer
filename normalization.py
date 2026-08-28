# src/normalization.py
import pandas as pd

class NormalizationEngine:
    @staticmethod
    def background_subtraction(df, target_col, background_value):
        """Subtracts a static baseline value from a target column."""
        if target_col in df.columns:
            df[f"{target_col} (Bg Sub)"] = df[target_col] - background_value
        return df

    @staticmethod
    def od_normalization(df, signal_col, od_col):
        """Normalizes a raw signal (like GFP) by dividing it by the cell density (OD600)."""
        if signal_col in df.columns and od_col in df.columns:
            # Avoid division by zero
            df[f"{signal_col}/OD"] = df[signal_col] / df[od_col].replace(0, pd.NA)
        return df

    @staticmethod
    def wild_type_fold_change(df, value_col, group_col, wt_label="WT"):
        """Calculates fold change relative to the mean of the Wild Type (WT) control group."""
        if value_col not in df.columns or group_col not in df.columns:
            return df
        
        wt_mean = df[df[group_col] == wt_label][value_col].mean()
        if pd.isna(wt_mean) or wt_mean == 0:
            return df
            
        df[f"{value_col} (Fold Change)"] = df[value_col] / wt_mean
        return df
