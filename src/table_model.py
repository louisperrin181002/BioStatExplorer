# src/table_model.py
"""A DataFrame shown in a QTableView without one widget per cell."""
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex


class PandasModel(QAbstractTableModel):
    """Shows a DataFrame in a QTableView without copying it into one
    widget per cell (fast for thousands of rows)."""

    # Qt asks the model for one cell at a time as it paints, so only the
    # visible rows are ever touched. The alternative — QTableWidget with
    # a QTableWidgetItem per cell — allocates a widget for all of them,
    # which is what makes a 50,000-row table take seconds to open.

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_frame(self, df):
        """Swap in a new DataFrame.

        begin/endResetModel bracket the change so every attached view
        drops its cached geometry and selection. Without them the view
        keeps painting the old shape and indexes past the end of the new
        frame.
        """
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        self.endResetModel()

    # A valid parent means Qt is asking about children of a cell. This is
    # a flat table, so the answer is always zero — without the guard, Qt
    # reads it as a tree and recurses.
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        # .iat is the positional single-cell accessor and the fastest one
        # pandas has; .iloc or .loc here would cost noticeably, since this
        # runs once per visible cell per repaint.
        v = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            # %g: six significant figures, switching to exponent form
            # when needed. Keeps 0.000123 and 1.23e+08 both readable in a
            # column of fixed width, which %f cannot.
            if isinstance(v, (float, np.floating)):
                return f"{v:.6g}"
            return str(v)
        # Numbers right-aligned so decimal points line up down a column.
        # int() because PyQt/PySide expect a plain int for this role, not
        # the flag enum.
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(
                v, (int, float, np.number)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        # Row numbers shown 1-based, as a spreadsheet does, rather than
        # the DataFrame index — which may be non-numeric or have gaps
        # after filtering.
        return str(section + 1)
