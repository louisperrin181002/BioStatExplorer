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


def _labels(text):
    return {t.strip().lower() for t in str(text).split(",") if t.strip()}


def plate_normalise(df, readouts, blank_col=None, blank_label="", match_col=None,
                    plate_col=None, od=None, keep_blanks=False):
    """
    Plate-reader corrections, applied per plate (*plate_col*, e.g. one file
    of a time series):

    * blank subtraction — the mean of the wells whose *blank_col* is one of
      *blank_label* ("blank" or "blank, LB only") is subtracted from every
      readout; with *match_col* (e.g. Medium) each well gets the blank of
      its own medium. Raw values are kept as "<readout> (raw)".
    * OD normalisation — "<readout> / <od>" for every other readout.

    Returns (DataFrame, notes). Blank wells are dropped unless keep_blanks.
    """
    import numpy as np
    df = df.copy()
    notes = []
    readouts = [r for r in readouts if r in df.columns]
    is_blank = pd.Series(False, index=df.index)
    if blank_col and blank_col in df.columns and _labels(blank_label):
        want = _labels(blank_label)
        is_blank = df[blank_col].astype(str).str.strip().str.lower().isin(want)
        if not is_blank.any():
            notes.append(f"No wells labelled '{blank_label}' in '{blank_col}' — "
                         "nothing subtracted.")
        else:
            plates = df[plate_col] if plate_col and plate_col in df else pd.Series("", index=df.index)
            for r in readouts:
                df[f"{r} (raw)"] = df[r]
            for plate in plates.unique():
                on_plate = plates == plate
                tag = f"{plate}: " if plate_col else ""
                for r in readouts:
                    blanks = df.loc[on_plate & is_blank]
                    overall = pd.to_numeric(blanks[r], errors="coerce").mean()
                    if match_col and match_col in df.columns:
                        means = pd.to_numeric(blanks[r], errors="coerce").groupby(
                            blanks[match_col].astype(str)).mean()
                        key = df.loc[on_plate, match_col].astype(str)
                        sub = key.map(means)
                        missing = sorted(set(key[sub.isna() & ~is_blank[on_plate]]))
                        if missing and not np.isnan(overall):
                            notes.append(f"{tag}no '{blank_label}' wells for "
                                         f"{', '.join(missing)} ({match_col}) — used "
                                         f"the plate's mean blank for {r}")
                        sub = sub.fillna(overall)
                    else:
                        sub = pd.Series(overall, index=df.index[on_plate])
                    if sub.isna().all():
                        notes.append(f"{tag}no blank value for {r}")
                        continue
                    df.loc[on_plate, r] = pd.to_numeric(df.loc[on_plate, r],
                                                        errors="coerce") - sub
                if plate_col:
                    notes.append(f"{plate}: {int((on_plate & is_blank).sum())} blank wells")
            notes.insert(0, f"Blank-subtracted with {int(is_blank.sum())} '{blank_label}' "
                            "well(s)" + (f", matched by {match_col}" if match_col else "")
                         + "; raw values kept as '… (raw)'.")
    if od and od in df.columns:
        odv = pd.to_numeric(df[od], errors="coerce")
        low = int(((odv <= 0) & ~is_blank).sum())
        for r in readouts:
            if r == od:
                continue
            df[f"{r} / {od}"] = pd.to_numeric(df[r], errors="coerce") / odv.where(odv > 0)
        notes.append(f"Divided each readout by {od}"
                     + (f"; {low} well(s) with {od} ≤ 0 left empty" if low else "") + ".")
    if is_blank.any() and not keep_blanks:
        df = df[~is_blank].reset_index(drop=True)
    return df, notes
