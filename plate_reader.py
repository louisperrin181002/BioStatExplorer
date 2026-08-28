# src/plate_reader.py
"""
Utilities for importing plate-reader-style data: a grid of wells
(rows A, B, C... x columns 1, 2, 3...) containing either raw
measurements (e.g. absorbance/fluorescence) or a "plate map" that
labels what's in each well (e.g. strain, media, replicate).
"""
import re
import string
import pandas as pd


PLATE_SHAPES = {
    "6-well (2x3)": (2, 3),
    "12-well (3x4)": (3, 4),
    "24-well (4x6)": (4, 6),
    "48-well (6x8)": (6, 8),
    "96-well (8x12)": (8, 12),
    "384-well (16x24)": (16, 24),
}


def well_id(row_idx, col_idx):
    """0-indexed (row, col) -> 'A1' style well ID."""
    letters = string.ascii_uppercase
    row_letter = letters[row_idx] if row_idx < 26 else (
        letters[row_idx // 26 - 1] + letters[row_idx % 26]
    )
    return f"{row_letter}{col_idx + 1}"


def make_empty_grid(n_rows, n_cols, fill=""):
    """A blank n_rows x n_cols grid of a given fill value (for a fresh editor grid)."""
    return [[fill for _ in range(n_cols)] for _ in range(n_rows)]


def detect_plate_grid(df):
    """
    Scans a raw dataframe (as read with header=None) for a block that looks
    like a plate: a row of column numbers (1, 2, 3, ...) immediately above
    rows starting with A, B, C, ... Returns (row_offset, col_offset, n_rows,
    n_cols) for the data block (excluding the header row/col), or None.
    """
    letters = set(string.ascii_uppercase)
    for r in range(min(len(df), 20)):
        row_vals = [str(v).strip() for v in df.iloc[r].tolist()]
        numeric_like = [v for v in row_vals if re.fullmatch(r"\d+(\.0+)?", v)]
        if len(numeric_like) >= 3:
            # Candidate header row of column numbers; find its starting column
            start_col = next((i for i, v in enumerate(row_vals) if re.fullmatch(r"\d+(\.0+)?", v)), None)
            if start_col is None:
                continue
            n_cols = len(numeric_like)
            # Look for consecutive row-letter labels starting just below/left
            row_letters = []
            for rr in range(r + 1, min(r + 1 + 26, len(df))):
                label = str(df.iat[rr, start_col - 1]).strip() if start_col > 0 else str(df.iat[rr, 0]).strip()
                if label and label.upper()[0] in letters and len(label) <= 2:
                    row_letters.append(label)
                else:
                    break
            if len(row_letters) >= 2:
                return (r + 1, start_col, len(row_letters), n_cols)
    return None


def grid_to_long(grid_values, value_name="Value", n_rows=None, n_cols=None):
    """
    grid_values: 2D list/array (rows x cols) of raw values, in plate order
    (row 0 = A, col 0 = well "..1"). Returns a long-format DataFrame with
    columns: Well, Row, Column, <value_name>.
    """
    n_rows = n_rows or len(grid_values)
    n_cols = n_cols or (len(grid_values[0]) if n_rows else 0)
    records = []
    for r in range(n_rows):
        for c in range(n_cols):
            val = grid_values[r][c] if r < len(grid_values) and c < len(grid_values[r]) else None
            records.append({
                "Well": well_id(r, c),
                "Row": string.ascii_uppercase[r] if r < 26 else well_id(r, 0)[:-len(str(c + 1))],
                "Column": c + 1,
                value_name: val,
            })
    df = pd.DataFrame(records)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df


def extract_plate_from_raw(raw_df, value_name="Value"):
    """
    Given a raw dataframe read from an uploaded plate-reader file (no
    assumptions about header row), auto-detects the plate block and returns
    a long-format DataFrame (Well, Row, Column, <value_name>). Raises
    ValueError if no plate-shaped block could be found.
    """
    loc = detect_plate_grid(raw_df)
    if loc is None:
        raise ValueError(
            "Could not find a plate-shaped grid (row letters A, B, C... beside "
            "column numbers 1, 2, 3...) in this file. You can still enter values "
            "manually in the plate editor."
        )
    row_start, col_start, n_rows, n_cols = loc
    block = raw_df.iloc[row_start:row_start + n_rows, col_start:col_start + n_cols].values
    return grid_to_long(block, value_name=value_name, n_rows=n_rows, n_cols=n_cols)


def map_grid_to_long(grid_values, field_name, n_rows, n_cols, split_delimiter=None, split_field_names=None):
    """
    Converts one plate-map layer (grid of text labels, e.g. strain names per
    well) into a long-format DataFrame with columns Well + the metadata
    field(s). If split_delimiter is given, each cell's text is split into
    multiple metadata columns (named by split_field_names).
    """
    records = []
    for r in range(n_rows):
        for c in range(n_cols):
            raw_val = grid_values[r][c] if r < len(grid_values) and c < len(grid_values[r]) else ""
            raw_val = "" if raw_val is None else str(raw_val).strip()
            rec = {"Well": well_id(r, c)}
            if raw_val == "":
                pass  # leave metadata fields blank/NaN for empty wells
            elif split_delimiter and split_field_names:
                parts = raw_val.split(split_delimiter)
                for i, fname in enumerate(split_field_names):
                    rec[fname] = parts[i].strip() if i < len(parts) else None
            else:
                rec[field_name] = raw_val
            records.append(rec)
    return pd.DataFrame(records)


def merge_well_layers(value_df, map_layer_dfs):
    """
    Merges the raw-value long DataFrame with one or more plate-map layer
    DataFrames (each indexed by 'Well') into a single analysis-ready
    DataFrame.
    """
    result = value_df
    for layer_df in map_layer_dfs:
        if layer_df is None or layer_df.empty:
            continue
        result = result.merge(layer_df, on="Well", how="left")
    return result
