# src/project.py
"""BioStat Explorer project files (.bse): the data, every plot setting,
plate grids, plate series and the figure layout, in one JSON file."""
import io
import json
from datetime import datetime

import pandas as pd

FORMAT = "biostatexplorer-project"
VERSION = 1
EXT = ".bse"
FILTER = "BioStat Explorer project (*.bse)"


def frame_to_json(df):
    """DataFrame -> JSON-safe dict. Numbers keep every digit (Python's
    float repr round-trips exactly); column types are restored on load."""
    if df is None:
        return None
    df = df.reset_index(drop=True)
    cols = [str(c) for c in df.columns]
    rows = df.astype(object).where(df.notna(), None).values.tolist()
    rows = [[v.item() if hasattr(v, "item") else v for v in r] for r in rows]
    return {"columns": cols, "dtypes": [str(t) for t in df.dtypes], "rows": rows}


def frame_from_json(d):
    if not d:
        return None
    if "rows" not in d:                       # (older 'table' format)
        df = pd.read_json(io.StringIO(json.dumps(d)), orient="table")
        return df.reset_index(drop=True)
    df = pd.DataFrame(d["rows"], columns=d["columns"])
    for c, t in zip(d["columns"], d["dtypes"]):
        try:
            if t.startswith(("float", "int")):
                df[c] = pd.to_numeric(df[c]).astype("float64" if df[c].isna().any()
                                                    or t.startswith("float") else t)
            elif t == "bool":
                df[c] = df[c].astype(bool)
            else:
                df[c] = df[c].astype(object)
        except (ValueError, TypeError):
            pass
    return df


def save(path, state):
    state = dict(state, format=FORMAT, version=VERSION,
                 saved=datetime.now().isoformat(timespec="seconds"))
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    import os
    os.replace(tmp, path)


def load(path):
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    if state.get("format") != FORMAT:
        raise ValueError("This is not a BioStat Explorer project file.")
    return state
