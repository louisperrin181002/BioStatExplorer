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
                # zero-padded text: a category that sorts 01..12
                "Column": f"{c + 1:02d}",
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


# ---------------------------------------------------------------------------
# All plate blocks in an instrument export (e.g. BMG CLARIOstar: GFP and
# mKate grids, raw and OD-normalised, in one sheet)
# ---------------------------------------------------------------------------

def _int_like(v):
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def _text(v):
    return isinstance(v, str) and v.strip() != ""


def find_plate_blocks(raw_df):
    """
    Every plate-shaped block in a sheet read with header=None: a row of
    column numbers 1, 2, 3... with row letters A, B, C... down its left.
    Returns a list of dicts: title, n_rows, n_cols, values (2-D list of
    floats/None), cells (2-D list of the cell text), is_map (mostly text:
    a plate map of strains/conditions, not readings), overflow (count of non-numeric cells such as 'OVRFLW'),
    saturated (wells sharing the block's maximum value, a sign of a
    saturated detector). Titles come from the text just above each block
    ('1. Raw Data (485-15/528-20 1)'), prefixed with the section it sits
    in when that differs ('Normalized data with OD').
    """
    from src.excel_loader import OVERFLOW
    vals = raw_df.values
    n_r, n_c = vals.shape
    blocks, section = [], ""
    used_rows = set()
    for r in range(n_r - 1):
        for c in range(n_c - 2):
            if (r, c) in used_rows:
                continue
            # column numbers 1..k starting at (r, c+1); letters at (r+1.., c)
            k = 0
            while c + 1 + k < n_c and _int_like(vals[r, c + 1 + k]) == k + 1:
                k += 1
            if k < 3:
                continue
            rows = 0
            while (r + 1 + rows < n_r and _text(vals[r + 1 + rows, c]) and
                   vals[r + 1 + rows, c].strip().upper() ==
                   string.ascii_uppercase[rows % 26]):
                rows += 1
            if rows < 2:
                continue
            # title: nearest text above the header row, within 3 rows
            title = ""
            for rr in range(r - 1, max(-1, r - 4), -1):
                texts = [str(vals[rr, cc]).strip() for cc in range(c, min(n_c, c + k + 1))
                         if _text(vals[rr, cc])]
                if texts:
                    title = texts[0]
                    break
            # a section heading between the previous block and this one
            prev_end = blocks[-1]["_end"] if blocks else -1
            for rr in range(prev_end + 1, max(prev_end + 1, r - 1)):
                for cc in range(n_c):
                    t = vals[rr, cc]
                    if _text(t) and t.strip() != title and \
                            re.search(r"normali|blank|corrected|ratio|mean", t, re.I):
                        section = t.strip()
            block = vals[r + 1:r + 1 + rows, c + 1:c + 1 + k]
            cells = [["" if v is None or (isinstance(v, float) and pd.isna(v))
                      else (f"{v:g}" if isinstance(v, float) else str(v).strip())
                      for v in row] for row in block]
            out, overflow = [], 0
            for row in block:
                line = []
                for v in row:
                    if isinstance(v, (int, float)) and not pd.isna(v):
                        line.append(float(v))
                    elif _text(v) and OVERFLOW.match(v):
                        overflow += 1
                        line.append(None)
                    else:
                        try:
                            line.append(float(str(v).replace(",", ".")))
                        except (TypeError, ValueError):
                            line.append(None)
                out.append(line)
            flat = [v for line in out for v in line if v is not None]
            saturated = 0
            if len(flat) >= 4:
                top = max(flat)
                saturated = sum(1 for v in flat if v == top)
                saturated = saturated if saturated >= 2 else 0
            name = re.sub(r"^\s*\d+\.\s*", "", title) or f"Plate {len(blocks) + 1}"
            if section and not name.lower().startswith(section.lower()):
                short = re.sub(r"^\s*\d+\.\s*", "", section)
                name = f"{name} — {short}"
            n_text = sum(1 for row in block for v in row
                         if _text(v) and not OVERFLOW.match(v)
                         and _int_like(v) is None)
            blocks.append({"title": name, "n_rows": rows, "n_cols": k,
                           "values": out, "cells": cells,
                           "is_map": n_text > 0 and n_text >= len(flat),
                           "overflow": overflow,
                           "saturated": saturated, "_end": r + rows})
            for rr in range(r, r + 1 + rows):
                used_rows.add((rr, c))
    # unique titles
    seen = {}
    for b in blocks:
        t = b["title"]
        seen[t] = seen.get(t, 0) + 1
        if seen[t] > 1:
            b["title"] = f"{t} ({seen[t]})"
        b.pop("_end", None)
    return blocks


def shape_name(n_rows, n_cols):
    """The PLATE_SHAPES key that fits a block (smallest plate it fits)."""
    for name, (r, c) in sorted(PLATE_SHAPES.items(), key=lambda kv: kv[1][0] * kv[1][1]):
        if n_rows <= r and n_cols <= c:
            return name
    return "384-well (16x24)"
