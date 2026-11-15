# src/excel_loader.py
"""
Reads data tables from Excel (.xlsx/.xlsm) and CSV/TSV/TXT files.

* Merged cells (e.g. 'Experiment' or 'Strain' written once across a block
  of replicate rows) are filled into every row they cover.
* The header row is found automatically, so instrument preambles above the
  table (User:, Path:, Date: ...) are skipped.
* Empty rows/columns are dropped, duplicate column names made unique, and
  numbers stored as text converted.
Column names are kept exactly as written in the file.
"""
# ---------------------------------------------------------------------------
# HOW A FILE BECOMES A TABLE
# ---------------------------------------------------------------------------
# Three stages, and each is a separate function so the plate reader can
# stop after the first one:
#
#   1. read_grid()        file  ->  raw cells, no header, merged cells
#                                   filled in. Excel, .xls and text all
#                                   converge here on one shape.
#   2. find_header_row()  guesses which row holds the column names,
#                         skipping any instrument preamble above it.
#   3. table_from_grid()  cuts the body below that row, names the
#                         columns, drops the empty rows and columns and
#                         converts the text that is really numbers.
#
# DataLoader on the end just applies those to several files and stacks
# the results. Everything reports what it did into a LoadReport, because
# all three stages guess, and a guess the user cannot see is a guess they
# cannot correct.
# ---------------------------------------------------------------------------

import csv
import io
import os
import re

import pandas as pd

TEXT_EXT = (".csv", ".tsv", ".txt")
EXCEL_EXT = (".xlsx", ".xlsm")
#: Cell contents a plate reader writes instead of a number when the
#: detector saturates. Matched so the report can say "these look like
#: overflow" rather than the vaguer "could not read these as numbers".
#: ">.*" catches ">3.5"; "\*+" catches the bare asterisks some readers use.
OVERFLOW = re.compile(r"^\s*(ovrflw|overflow|over|sat(urated)?|>.*|\*+)\s*$",
                      re.I)


class LoadReport:
    """What the loader did to a file, for the status bar / a message."""

    def __init__(self, name):
        self.name = name
        self.notes = []

    def __str__(self):
        # Note the precedence: the conditional covers the whole
        # concatenation, so a report with no notes is the empty string
        # and not a bare "filename: " with nothing after it.
        return f"{self.name}: " + "; ".join(self.notes) if self.notes else ""


def sheet_names(path):
    """Sheet names of an Excel file ([] for text files)."""
    if not path.lower().endswith(EXCEL_EXT):
        return []
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def read_grid(path, sheet=None):
    """The raw cells of a sheet (or text file) as a DataFrame with no
    header, merged cells filled in. Used by the table and plate readers."""
    low = path.lower()
    if low.endswith(EXCEL_EXT):
        import openpyxl
        # data_only=True returns the last value Excel calculated instead
        # of the formula text. Without it a column of "=B2/C2" arrives as
        # strings and the whole column reads as non-numeric.
        wb = openpyxl.load_workbook(path, data_only=True)
        try:
            ws = wb[sheet] if sheet else wb.worksheets[0]
            # values_only=True yields plain tuples rather than Cell
            # objects — far less memory on a big sheet, and nothing below
            # needs the formatting.
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            # Rows can come back ragged. Pad them all to the widest one,
            # or DataFrame() below would align them by position and the
            # merged-cell loop could index past the end of a short row.
            width = max((len(r) for r in rows), default=0)
            rows = [r + [None] * (width - len(r)) for r in rows]
            # --- fill merged cells ------------------------------------
            # A merged block stores its value only in the top-left cell;
            # every other cell in it reads as None. That is how people
            # write "Experiment 1" once across six replicate rows, so the
            # value is copied into every cell of the block and those rows
            # become six ordinary rows that all know their experiment.
            for rng in ws.merged_cells.ranges:
                # openpyxl counts rows and columns from 1, the grid from 0.
                r0, c0 = rng.min_row - 1, rng.min_col - 1
                if r0 >= len(rows) or c0 >= width:
                    continue
                v = rows[r0][c0]
                # min(...) against the real size: a merge can be declared
                # over a region wider than the cells actually present.
                for r in range(r0, min(rng.max_row, len(rows))):
                    for c in range(c0, min(rng.max_col, width)):
                        rows[r][c] = v
            merged = len(ws.merged_cells.ranges)
        finally:
            wb.close()
        grid = pd.DataFrame(rows)
        # .attrs is pandas' free-form metadata dict. Used to carry the
        # merge count out to the caller without changing the return type,
        # which the plate reader also depends on.
        grid.attrs["merged"] = merged
        return grid
    if low.endswith(".xls"):
        try:
            return pd.read_excel(path, sheet_name=sheet or 0, header=None)
        except ImportError as exc:
            raise ValueError("Old .xls files need the 'xlrd' package — or "
                             "re-save the file as .xlsx in Excel.") from exc
    # --- text files -----------------------------------------------------
    # Read the bytes once, then work out the separator from the text. Two
    # encodings tried in order: utf-8-sig strips the byte-order mark Excel
    # writes when it saves a CSV, and latin-1 cannot fail, so it is the
    # catch-all that keeps a mis-encoded file openable.
    raw = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                raw = f.read()
            break
        except UnicodeDecodeError:
            continue
    sep = _sniff_separator(raw, low)
    # How many columns the widest line has. Counted with csv.reader so
    # that a separator inside a quoted field is not miscounted.
    #
    # This has to be worked out before the read, not left to pandas.
    # Without names=, pandas takes its column count from the first line,
    # and an instrument preamble's first line is usually narrower than
    # the table below it — "User: L Perrin" is one field where the table
    # has three. pandas then raises "Expected 1 fields in line 4, saw 2"
    # and the file cannot be opened at all, which is the exact case the
    # header search further down exists to handle.
    try:
        n_cols = max((len(r) for r in csv.reader(io.StringIO(raw),
                                                 delimiter=sep)), default=1)
    except csv.Error:
        n_cols = max((len(line.split(sep)) for line in raw.splitlines()),
                     default=1)
    # dtype=object keeps every cell as written, so the conversion later
    # is the only place that decides what is a number. skip_blank_lines
    # is off because a blank line above the table is part of the preamble
    # the header search has to count through. engine="python" because the
    # fast C parser rejects some of the separators Sniffer returns.
    # names=range(n_cols) fixes the width, so short lines are padded with
    # NaN instead of being treated as a different table.
    return pd.read_csv(io.StringIO(raw), sep=sep, header=None, dtype=object,
                       skip_blank_lines=False, engine="python",
                       names=range(max(n_cols, 1)))



def _sniff_separator(raw: str, low: str) -> str:
    """Which of comma, semicolon and tab separates the columns.

    csv.Sniffer is the obvious tool and it gets this wrong on exactly the
    files a plate reader produces in Europe: a semicolon-separated table
    whose numbers use the comma as the decimal mark. Sniffer sees commas
    everywhere and picks the comma, and "WT;0,25;ok" parses into the two
    columns "WT;0" and "25;ok".

    So each candidate is scored instead of guessed. A real separator
    makes most lines come out the *same* width, and wider than one; a
    decimal comma makes lines ragged. The score is therefore how much of
    the file agrees on the commonest field count, then how wide that
    count is, and the preamble lines that disagree simply lose a little
    consistency rather than deciding the answer.
    """
    from collections import Counter

    best, best_score = None, None
    for cand in (";", ",", "\t"):
        try:
            widths = [len(r) for r in csv.reader(io.StringIO(raw),
                                                 delimiter=cand)
                      if any(str(c).strip() for c in r)]
        except csv.Error:
            continue
        if not widths:
            continue
        modal, n = Counter(widths).most_common(1)[0]
        if modal < 2:
            continue                      # this separator splits nothing
        score = (n / len(widths), modal)  # agreement first, then width
        if best_score is None or score > best_score:
            best, best_score = cand, score
    if best:
        return best
    # Nothing split the file: it is a single column. The extension is the
    # only hint left, and for one column the choice changes nothing.
    return "\t" if low.endswith(".tsv") else ","


def _blank(v):
    """True for an empty cell, in any of the three forms one can take.

    None from openpyxl, NaN from pandas, or a string of spaces from
    someone pressing the space bar. The isinstance guard before pd.isna
    matters: pd.isna on a list returns an array, not a bool.
    """
    return v is None or (isinstance(v, float) and pd.isna(v)) or \
        (isinstance(v, str) and not v.strip())


def _is_number(v):
    """Whether a cell holds a number, including one written as text."""
    # bool is a subclass of int in Python, so True would otherwise count
    # as the number 1 and a column of TRUE/FALSE would convert to 1/0.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return not pd.isna(v)
    if isinstance(v, str):
        try:
            # Comma swapped for a point: "0,25" is how a quarter is
            # written across most of Europe, and float() will not take it.
            float(v.replace(",", "."))
            return True
        except ValueError:
            return False
    return False


def find_header_row(grid):
    """Index of the table's header row: the first row with at least two
    filled cells, mostly text, followed by a row of data."""
    # Three tests, all of which a header row passes and a preamble line
    # does not. Only the first 60 rows are searched: past that it is not
    # a preamble, it is a different kind of file.
    n = len(grid)
    for r in range(min(n, 60)):
        row = grid.iloc[r].tolist()
        filled = [v for v in row if not _blank(v)]
        # 1. At least two filled cells. "User: L. Perrin" is one label and
        #    one value, which is exactly the shape of a preamble line.
        if len(filled) < 2:
            continue
        texts = [v for v in filled if not _is_number(v)]
        # 2. Mostly text. A row of numbers is data, not column names —
        #    but a header may legitimately contain a few, as in a time
        #    course whose columns are "0", "3", "6", so it is a majority
        #    test rather than an absolute one.
        if len(texts) < max(2, 0.6 * len(filled)):
            continue
        if r + 1 >= n:
            return None
        # 3. Something underneath it. A title line sitting alone above a
        #    blank row looks like a header by the first two tests; what
        #    separates it from a real one is that a real one has data in
        #    the row below, roughly as wide as itself.
        nxt = [v for v in grid.iloc[r + 1].tolist() if not _blank(v)]
        if len(nxt) >= max(1, len(filled) // 2):
            return r
    # Nothing convincing: assume the first row rather than refuse the
    # file, since a wrong guess is visible and recoverable, and None here
    # would mean the user sees nothing at all.
    return 0 if n else None


def table_from_grid(grid, report=None):
    """Turns a raw grid into a clean table (header detected, empty
    rows/columns dropped, numbers converted)."""
    report = report or LoadReport("")
    h = find_header_row(grid)
    if h is None:
        return pd.DataFrame()
    if h > 0:
        report.notes.append(f"table starts at row {h + 1}")
    header = ["" if _blank(v) else str(v).strip() for v in grid.iloc[h]]
    # reset_index(drop=True) renumbers the rows from 0; without it the body
    # keeps the original sheet row numbers and every later positional
    # index is off by the height of the preamble.
    body = grid.iloc[h + 1:].reset_index(drop=True)
    # --- drop the empty columns ----------------------------------------
    # A column survives if it has a name *or* any content: a spacer column
    # between two blocks has neither, while an unnamed column of real data
    # is worth keeping under a made-up name.
    keep = [i for i, name in enumerate(header)
            if name or body.iloc[:, i].map(lambda v: not _blank(v)).any()]
    body = body.iloc[:, keep]
    # --- name the columns, uniquely -------------------------------------
    # pandas allows duplicate column names but almost nothing downstream
    # does: df[name] on a duplicate returns a DataFrame, not a Series, and
    # the plotting code then fails somewhere far from here. Repeats become
    # "OD600 (2)", "OD600 (3)".
    names, seen = [], {}
    for i in keep:
        name = header[i] or f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        names.append(name)
    body.columns = names
    # Drop rows that are blank all the way across — the spacer rows people
    # leave between blocks. axis=1 applies the test across each row; the ~
    # inverts the mask so the kept rows are the ones that are *not* empty.
    body = body[~body.apply(lambda r: all(_blank(v) for v in r), axis=1)]
    body = body.reset_index(drop=True)
    # --- convert the columns that are really numbers ---------------------
    # Everything arrived as text. A column becomes numeric only if nearly
    # all of its filled cells parse as numbers; the stragglers become
    # empty and are reported, rather than the whole column staying text
    # because of one "ovrflw" in it.
    for c in body.columns:
        col = body[c]
        if col.dtype == object:
            col = col.map(lambda v: v.strip() if isinstance(v, str) else v)
            col = col.map(lambda v: None if _blank(v) else v)
            body[c] = col
        vals = [v for v in body[c] if not _blank(v)]
        if not vals:
            continue
        nums = [v for v in vals if _is_number(v)]
        # All numeric, or at least 80% of a column long enough for that
        # proportion to mean something. The length test stops a 3-row
        # column with one stray word from being converted on the strength
        # of two cells.
        if len(nums) == len(vals) or (len(nums) >= 0.8 * len(vals)
                                      and len(vals) >= 5):
            bad = [v for v in vals if not _is_number(v)]
            conv = body[c].map(lambda v: float(v.replace(",", "."))
                               if isinstance(v, str) and _is_number(v) else v)
            # errors="coerce" turns whatever is left unparseable into
            # NaN instead of raising — which is the intended outcome here,
            # since the cells concerned were counted above and are about
            # to be described in the report.
            body[c] = pd.to_numeric(conv, errors="coerce")
            if bad:
                over = sum(1 for v in bad if OVERFLOW.match(str(v)))
                report.notes.append(
                    f"'{c}': {len(bad)} non-numeric cell(s) left empty"
                    + (f" ({over} look like detector overflow)" if over else
                       f" (e.g. '{bad[0]}')"))
    return body


class DataLoader:
    def load_table(self, path, sheet=None):
        """(DataFrame, LoadReport) for one file/sheet."""
        report = LoadReport(os.path.basename(path) + (f" [{sheet}]"
                                                      if sheet else ""))
        grid = read_grid(path, sheet)
        if grid.attrs.get("merged"):
            report.notes.append(f"{grid.attrs['merged']} merged cell "
                                "range(s) filled into every row")
        return table_from_grid(grid, report), report

    def load_and_merge(self, file_paths, sheets=None):
        """
        Loads each file (sheets: {path: sheet name}), tags rows with their
        'Source File', and stacks them. Returns (DataFrame, numeric columns,
        other columns, [LoadReport]).
        """
        sheets = sheets or {}
        frames, reports = [], []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            df, rep = self.load_table(path, sheets.get(path))
            if df.empty:
                rep.notes.append("no table found")
                reports.append(rep)
                continue
            df["Source File"] = os.path.basename(path)
            frames.append(df)
            reports.append(rep)
        if not frames:
            return pd.DataFrame(), [], [], reports
        # sort=False keeps the columns in the order the first file had
        # them, instead of alphabetically. Files that do not share every
        # column still stack: the missing cells become NaN.
        merged = pd.concat(frames, ignore_index=True, sort=False)
        # Split the columns into measurements and labels. This is what
        # lets the rest of the app offer the right columns in the right
        # places — a label cannot go on a Y axis, a measurement cannot
        # group the data.
        numeric = [c for c in merged.columns
                   if pd.api.types.is_numeric_dtype(merged[c])]
        other = [c for c in merged.columns if c not in numeric]
        return merged, numeric, other, reports
