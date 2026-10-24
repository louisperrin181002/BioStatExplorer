# src/curves_tab.py
"""The Curves tab: growth curves and dose-response fits, one per well or
per group, with a table of parameters that can be sent to the Plotting
Dashboard to compare groups (e.g. growth rate per strain)."""
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QComboBox, QCheckBox,
    QPushButton, QLabel, QSplitter, QTableView, QHeaderView,
)

from src import curves
from src.plotting import palette_colors
from src.table_model import PandasModel

NONE = "(none)"


class CurvesTab(QWidget):
    """Signals: send_parameters(DataFrame) — use the fitted parameters as a
    dataset; export_requested(DataFrame)."""

    send_parameters = Signal(object)
    export_requested = Signal(object)

    def __init__(self, get_data, parent=None):
        super().__init__(parent)
        self._get_data = get_data             # () -> (DataFrame, metric cols)
        self.params = pd.DataFrame()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        side = QWidget()
        side.setObjectName("Card")
        side.setMinimumWidth(300)
        side.setMaximumWidth(360)
        f = QFormLayout(side)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setContentsMargins(12, 12, 12, 12)
        f.setVerticalSpacing(9)
        intro = QLabel("Fit a curve per well (or per group), then compare the "
                       "parameters between strains on the Plotting Dashboard.")
        intro.setWordWrap(True)
        intro.setObjectName("Hint")
        f.addRow(intro)
        self.model = QComboBox()
        self.model.addItems(curves.MODELS)
        self.model.currentTextChanged.connect(self._model_changed)
        f.addRow("Model", self.model)
        self.x = QComboBox()
        f.addRow("X (time / conc.)", self.x)
        self.y = QComboBox()
        f.addRow("Y", self.y)
        self.by = QComboBox()
        self.by.setToolTip("One curve per value of this column — e.g. Well "
                           "(then compare strains), or Strain (one curve each)")
        f.addRow("Fit each", self.by)
        self.color = QComboBox()
        self.color.setToolTip("Colours the curves, and is kept in the parameter table")
        f.addRow("Colour by", self.color)
        self.logx = QCheckBox("Log X axis")
        f.addRow("", self.logx)
        fit_btn = QPushButton("Fit")
        fit_btn.setObjectName("primaryBtn")
        fit_btn.clicked.connect(self.fit)
        f.addRow(fit_btn)
        self.send_btn = QPushButton("Compare Parameters on the Plotting Dashboard")
        self.send_btn.setToolTip("Makes the parameter table the dataset (Undo brings "
                                 "your data back), e.g. growth rate by strain with stars")
        self.send_btn.clicked.connect(lambda: self.send_parameters.emit(self.params.copy()))
        f.addRow(self.send_btn)
        exp_btn = QPushButton("Export Parameters…")
        exp_btn.clicked.connect(lambda: self.export_requested.emit(self.params.copy()))
        f.addRow(exp_btn)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setObjectName("Hint")
        f.addRow(self.info)
        lay.addWidget(side)

        split = QSplitter(Qt.Orientation.Vertical)
        card = QWidget()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        cl.addWidget(self.canvas)
        split.addWidget(card)
        self.table_model = PandasModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        split.addWidget(self.table)
        split.setSizes([600, 260])
        lay.addWidget(split, 1)
        self._model_changed()
        self._draw_empty("Choose a model and columns, then press Fit.")

    # ------------------------------------------------------------------

    def refresh_columns(self):
        df, metrics = self._get_data()
        if df is None:
            return
        cols = list(df.columns)
        others = [c for c in cols if c not in metrics]
        for combo, items, pref in (
                (self.x, metrics, ("time", "conc", "dose", "hour")),
                (self.y, metrics, ("od", "600", "gfp", "signal", "value")),
                (self.by, [NONE] + cols, ("well",)),
                (self.color, [NONE] + others, ("strain", "condition", "group"))):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if cur in items:
                combo.setCurrentText(cur)
            else:
                pick = next((c for c in items if any(k in c.lower() for k in pref)), None)
                if pick:
                    combo.setCurrentText(pick)
            combo.blockSignals(False)
        if self.y.currentText() == self.x.currentText():
            alt = [c for c in metrics if c != self.x.currentText()]
            if alt:
                self.y.setCurrentText(alt[0])

    def _model_changed(self, *_):
        self.logx.setChecked(self.model.currentText() == curves.DOSE)

    def _draw_empty(self, text):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color="#64748B")
        self.canvas.draw()

    def fit(self):
        df, _ = self._get_data()
        x, y, by = self.x.currentText(), self.y.currentText(), self.by.currentText()
        color = self.color.currentText()
        if df is None or df.empty or not x or not y:
            self._draw_empty("Open data with a time / concentration column first.")
            return
        if x == y:
            self._draw_empty("X and Y must be different columns.")
            return
        keys = [] if by == NONE else [by]
        keep = [c for c in df.columns if c not in (x, y)]
        model = self.model.currentText()
        self.params, fits = curves.fit_groups(df, x, y, keys, model, keep=keep)
        self.table_model.set_frame(self.params.round(6))
        self.table.resizeColumnsToContents()
        failed = int(self.params.get("Note", pd.Series(dtype=object)).notna().sum()) \
            if "Note" in self.params else 0
        self.info.setText(f"{len(self.params)} curve(s) fitted"
                          + (f"; {failed} could not be fitted (see Note)" if failed else "")
                          + ".")
        self._plot(df, x, y, keys, color, fits, model)

    def _plot(self, df, x, y, keys, color, fits, model):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        cats = []
        if color != NONE and color in df.columns:
            cats = list(dict.fromkeys(df[color].dropna().astype(str)))
        cols = dict(zip(cats, palette_colors("Modern", len(cats))))
        by_key = {}
        if keys and color != NONE and color in df.columns:
            first = df.drop_duplicates(keys).set_index(keys)[color].astype(str)
            by_key = {(k if isinstance(k, tuple) else (k,)): v for k, v in first.items()}
        seen = set()
        for key, (xs, ys, fn) in fits.items():
            cat = by_key.get(key) if by_key else (str(key[0]) if not keys else None)
            c = cols.get(cat, "#2563EB")
            label = cat if cat and cat not in seen else None
            if cat:
                seen.add(cat)
            ax.scatter(xs, ys, s=14, color=c, alpha=0.55, linewidths=0, label=label)
            if fn is not None:
                lo, hi = np.nanmin(xs), np.nanmax(xs)
                grid = (np.logspace(np.log10(lo), np.log10(hi), 200)
                        if self.logx.isChecked() and lo > 0 else np.linspace(lo, hi, 200))
                ax.plot(grid, fn(grid), color=c, lw=1.6)
        if self.logx.isChecked():
            ax.set_xscale("log")
        if model == curves.MAX_SLOPE:
            ax.set_yscale("log")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{model}: {y} vs {x}", fontsize=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.yaxis.grid(True, color="#E2E8F0")
        ax.set_axisbelow(True)
        if seen:
            ax.legend(frameon=False, fontsize=8, title=color, title_fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------

    def to_dict(self):
        return {"model": self.model.currentText(), "x": self.x.currentText(),
                "y": self.y.currentText(), "by": self.by.currentText(),
                "color": self.color.currentText(), "logx": self.logx.isChecked()}

    def from_dict(self, d):
        if not d:
            return
        self.refresh_columns()
        for combo, key in ((self.model, "model"), (self.x, "x"), (self.y, "y"),
                           (self.by, "by"), (self.color, "color")):
            if d.get(key) and combo.findText(d[key]) >= 0:
                combo.setCurrentText(d[key])
        self.logx.setChecked(bool(d.get("logx")))
