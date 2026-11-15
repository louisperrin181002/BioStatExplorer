# src/normalization.py
"""Turning raw plate-reader numbers into comparable ones.

Two separate things live here, and they are used at different moments:

* NormalizationEngine — three one-column corrections applied to an
  already-built dataset, each adding a new column and leaving the
  original alone.
* plate_normalise()   — the whole-plate correction run at import time,
  before a dataset exists. This is the one that matters: it subtracts
  blanks per plate and divides by OD, and every later statistic inherits
  whatever it does.

The rule followed throughout is that a correction never destroys what it
corrects. Blank subtraction keeps the originals as "<readout> (raw)", OD
normalisation writes a new "<readout> / OD" column, and every decision
taken is appended to *notes* for the import report — because a blank that
was quietly not found would shift every number on the plate with nothing
on screen to say so.
"""
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
            # replace(0, pd.NA) rather than a guard: dividing by zero in
            # pandas gives inf, which then survives into means and plots
            # and poisons them silently. NA propagates instead and is
            # excluded from statistics by default.
            df[f"{signal_col}/OD"] = df[signal_col] / df[od_col].replace(0, pd.NA)
        return df

    @staticmethod
    def wild_type_fold_change(df, value_col, group_col, wt_label="WT"):
        """Calculates fold change relative to the mean of the Wild Type (WT) control group."""
        if value_col not in df.columns or group_col not in df.columns:
            return df
        
        wt_mean = df[df[group_col] == wt_label][value_col].mean()
        # No wild type found, or a mean of zero: return the frame
        # untouched rather than add a column of inf or NaN that would
        # look like a real result.
        if pd.isna(wt_mean) or wt_mean == 0:
            return df
            
        df[f"{value_col} (Fold Change)"] = df[value_col] / wt_mean
        return df


def _labels(text):
    """Split "blank, LB only" into {"blank", "lb only"}.

    A set of lower-cased, stripped labels, so the well annotation can be
    matched regardless of spacing or capitalisation, and so several
    labels can mark blanks on the same plate.
    """
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
    # --- outline ---------------------------------------------------------
    # 1. work out which rows are blanks
    # 2. for each plate, and each readout: find the blank mean (per medium
    #    if match_col is given), keep the raw column, subtract
    # 3. divide every readout by OD
    # 4. drop the blank wells
    # Steps 2 and 3 both write notes; step 1 writes one if it finds nothing.
    import numpy as np
    df = df.copy()          # never modify the caller's frame in place
    notes = []
    readouts = [r for r in readouts if r in df.columns]
    is_blank = pd.Series(False, index=df.index)
    if blank_col and blank_col in df.columns and _labels(blank_label):
        want = _labels(blank_label)
        # astype(str) first: the column may hold NaN or numbers, and
        # .str accessors return NaN rather than False for those, which
        # would make isin() produce a mask with holes in it.
        is_blank = df[blank_col].astype(str).str.strip().str.lower().isin(want)
        if not is_blank.any():
            notes.append(f"No wells labelled '{blank_label}' in '{blank_col}' — "
                         "nothing subtracted.")
        else:
            # One plate's blanks must not correct another's: a time
            # series is several files read hours apart, and the lamp and
            # the medium drift between them. With no plate column the
            # whole frame is treated as a single plate.
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
                        # A blank per medium. LB and M9 absorb differently,
                        # so one pooled blank would under-correct one and
                        # over-correct the other.
                        means = pd.to_numeric(blanks[r], errors="coerce").groupby(
                            blanks[match_col].astype(str)).mean()
                        key = df.loc[on_plate, match_col].astype(str)
                        sub = key.map(means)
                        # Media that have no blank of their own. The
                        # ~is_blank term stops the blank wells themselves
                        # being reported as unmatched.
                        missing = sorted(set(key[sub.isna() & ~is_blank[on_plate]]))
                        if missing and not np.isnan(overall):
                            notes.append(f"{tag}no '{blank_label}' wells for "
                                         f"{', '.join(missing)} ({match_col}) — used "
                                         f"the plate's mean blank for {r}")
                        # Fall back to the plate's overall blank, having
                        # just said so in the notes — better than leaving
                        # those wells uncorrected while their neighbours
                        # are corrected.
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
    # --- divide by OD -----------------------------------------------------
    if od and od in df.columns:
        odv = pd.to_numeric(df[od], errors="coerce")
        # Count the wells that cannot be normalised before doing it, so
        # the note can say how many: an OD at or below zero after blank
        # subtraction means a well dimmer than its own blank, which is
        # noise rather than negative growth.
        low = int(((odv <= 0) & ~is_blank).sum())
        for r in readouts:
            if r == od:
                continue
            # .where(odv > 0) turns the non-positive ODs into NaN before
            # the division, so those wells come out empty instead of as
            # a huge or negative ratio that would dominate every plot.
            df[f"{r} / {od}"] = pd.to_numeric(df[r], errors="coerce") / odv.where(odv > 0)
        notes.append(f"Divided each readout by {od}"
                     + (f"; {low} well(s) with {od} ≤ 0 left empty" if low else "") + ".")
    # The blanks have done their job and are not samples; left in, they
    # would appear as a group in every plot and drag every mean down.
    if is_blank.any() and not keep_blanks:
        df = df[~is_blank].reset_index(drop=True)
    return df, notes
