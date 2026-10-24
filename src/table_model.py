# src/table_model.py
"""A DataFrame shown in a QTableView without one widget per cell."""
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex


class PandasModel(QAbstractTableModel):
    """Shows a DataFrame in a QTableView without copying it into one
    widget per cell (fast for thousands of rows)."""

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_frame(self, df):
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        v = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if isinstance(v, (float, np.floating)):
                return f"{v:.6g}"
            return str(v)
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(
                v, (int, float, np.number)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)
