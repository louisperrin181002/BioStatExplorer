# src/layout_tab.py
"""The Layout tab: several graphs on one page at a journal size (mm),
with panel letters and one font size, exported as PDF/SVG/PNG/TIFF.

Each panel keeps its own copy of the rows it shows, so panels made from
different datasets (e.g. growth curves and the fitted growth rates) can
sit on the same page."""
import math
import string

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QListWidget, QListWidgetItem,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QLabel,
    QScrollArea, QAbstractItemView, QInputDialog,
)

from src import project

WIDTHS = {"Single column (85 mm)": 85.0, "1.5 column (120 mm)": 120.0,
          "Double column (174 mm)": 174.0, "Full page (190 mm)": 190.0}


class LayoutTab(QWidget):
    """*draw(ax, df, spec)* draws one panel. Signals: export_requested(fmt)."""

    export_requested = Signal(str)
    changed = Signal()

    def __init__(self, draw, parent=None):
        super().__init__(parent)
        self._draw = draw
        self.panels = []                 # {"spec": dict, "df": DataFrame, "name": str}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)

        side = QWidget()
        side.setObjectName("Card")
        side.setMinimumWidth(300)
        side.setMaximumWidth(360)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Panels")
        title.setObjectName("Title")
        sl.addWidget(title)
        hint = QLabel("On the Plotting Dashboard, set up a graph and press "
                      "＋ Add to Layout. Drag to reorder; double-click to rename.")
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        sl.addWidget(hint)
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.model().rowsMoved.connect(self._reordered)
        self.list.itemDoubleClicked.connect(self._rename)
        sl.addWidget(self.list, 1)
        row = QHBoxLayout()
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove)
        row.addWidget(rm)
        clr = QPushButton("Clear")
        clr.clicked.connect(self._clear)
        row.addWidget(clr)
        sl.addLayout(row)

        page = QLabel("Page")
        page.setObjectName("Section")
        sl.addWidget(page)
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.width_combo = QComboBox()
        self.width_combo.addItems(list(WIDTHS))
        self.width_combo.setCurrentText("Double column (174 mm)")
        f.addRow("Width", self.width_combo)
        self.cols = QSpinBox()
        self.cols.setRange(1, 6)
        self.cols.setValue(2)
        f.addRow("Columns", self.cols)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(20, 250)
        self.height_spin.setValue(60)
        self.height_spin.setSuffix(" mm")
        f.addRow("Panel height", self.height_spin)
        self.font_spin = QDoubleSpinBox()
        self.font_spin.setRange(5, 14)
        self.font_spin.setValue(7)
        self.font_spin.setSingleStep(0.5)
        self.font_spin.setSuffix(" pt")
        f.addRow("Font size", self.font_spin)
        self.letters = QCheckBox("Panel letters (A, B, C…)")
        self.letters.setChecked(True)
        f.addRow("", self.letters)
        sl.addLayout(f)
        for w in (self.width_combo, self.cols, self.height_spin, self.font_spin, self.letters):
            sig = getattr(w, "currentTextChanged", None) or getattr(w, "valueChanged", None) \
                or w.toggled
            sig.connect(self.render)
        exp = QLabel("Export")
        exp.setObjectName("Section")
        sl.addWidget(exp)
        er = QHBoxLayout()
        for fmt in ("pdf", "svg", "png", "tiff"):
            b = QPushButton(fmt.upper())
            b.clicked.connect(lambda _c=False, f_=fmt: self.export_requested.emit(f_))
            er.addWidget(b)
        sl.addLayout(er)
        lay.addWidget(side)

        self.figure = Figure(dpi=140)            # on-screen preview, a little enlarged
        self.canvas = FigureCanvas(self.figure)
        holder = QWidget()
        holder.setObjectName("Card")
        hl = QVBoxLayout(holder)
        hl.addWidget(self.canvas, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        lay.addWidget(scroll, 1)
        self.render()

    # ------------------------------------------------------------------

    def add_panel(self, spec, df):
        metric = spec["metrics"][0] if spec.get("metrics") else spec.get("y", "")
        name = f"{metric} vs {spec.get('x', '')} ({spec.get('chart', '')})"
        self.panels.append({"spec": dict(spec, metrics=[metric]), "df": df.copy(),
                            "name": name})
        self._refresh_list()
        self.render()
        self.changed.emit()

    def _refresh_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for i, p in enumerate(self.panels):
            it = QListWidgetItem(f"{string.ascii_uppercase[i % 26]}  {p['name']}")
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.list.addItem(it)
        self.list.blockSignals(False)

    def _reordered(self, *_):
        order = [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                 for i in range(self.list.count())]
        self.panels = [self.panels[i] for i in order]
        self._refresh_list()
        self.render()
        self.changed.emit()

    def _rename(self, item):
        i = item.data(Qt.ItemDataRole.UserRole)
        p = self.panels[i]
        metric = p["spec"]["metrics"][0]
        cur = p["spec"].get("titles", {}).get(metric, "")
        text, ok = QInputDialog.getText(self, "Panel Title", "Title (empty: automatic):",
                                        text=cur)
        if ok:
            p["spec"].setdefault("titles", {})
            if text.strip():
                p["spec"]["titles"][metric] = text.strip()
            else:
                p["spec"]["titles"].pop(metric, None)
            self.render()
            self.changed.emit()

    def _remove(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.panels):
            del self.panels[r]
            self._refresh_list()
            self.render()
            self.changed.emit()

    def _clear(self):
        self.panels = []
        self._refresh_list()
        self.render()
        self.changed.emit()

    # ------------------------------------------------------------------

    def page_size_mm(self):
        n = max(len(self.panels), 1)
        cols = min(self.cols.value(), n)
        rows = math.ceil(n / cols)
        return WIDTHS[self.width_combo.currentText()], rows * self.height_spin.value(), rows, cols

    def render(self, *_, figure=None):
        fig = figure or self.figure
        fig.clf()
        w_mm, h_mm, rows, cols = self.page_size_mm()
        fig.set_size_inches(w_mm / 25.4, h_mm / 25.4)
        if figure is None:
            dpi = fig.get_dpi()
            self.canvas.setFixedSize(int(w_mm / 25.4 * dpi), int(h_mm / 25.4 * dpi))
        if not self.panels:
            fig.text(0.5, 0.5, "No panels yet — add graphs from the Plotting Dashboard "
                     "(＋ Add to Layout).", ha="center", va="center", color="#64748B",
                     fontsize=9, wrap=True)
            if figure is None:
                self.canvas.draw()
            return
        size = self.font_spin.value()
        with matplotlib.rc_context({"font.size": size}):
            for i, p in enumerate(self.panels):
                ax = fig.add_subplot(rows, cols, i + 1)
                try:
                    self._draw(ax, p["df"], p["spec"])
                except Exception as exc:          # one bad panel shouldn't stop the page
                    ax.cla()
                    ax.axis("off")
                    ax.text(0.5, 0.5, f"Cannot draw:\n{exc}", ha="center", va="center",
                            fontsize=size, color="#DC2626", transform=ax.transAxes)
                    continue
                self._restyle(ax, size)
                if self.letters.isChecked():
                    ax.text(-0.02, 1.02, string.ascii_uppercase[i % 26], transform=ax.transAxes,
                            fontsize=size + 3, fontweight="bold", ha="right", va="bottom")
            try:
                fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=1.0)
            except Exception:
                pass
        if figure is None:
            self.canvas.draw()

    @staticmethod
    def _restyle(ax, size):
        """One journal font size for everything in a panel."""
        ax.title.set_fontsize(size + 1)
        ax.title.set_fontweight("semibold")
        ax.xaxis.label.set_fontsize(size)
        ax.yaxis.label.set_fontsize(size)
        ax.tick_params(labelsize=size - 0.5, length=2.5, width=0.6)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(size - 0.5)
            if leg.get_title():
                leg.get_title().set_fontsize(size - 0.5)
        for t in ax.texts:                       # stars and p-values
            txt = t.get_text()
            t.set_fontsize(size + 2 if set(txt) <= set("*") else
                           size - 1 if "p" in txt else size)

    def export_figure(self, path, fmt, dpi):
        fig = Figure()
        self.render(figure=fig)
        rc = {"svg.fonttype": "none", "pdf.fonttype": 42}
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "tiff" else {}
        with matplotlib.rc_context(rc):
            fig.savefig(path, format=fmt, dpi=dpi, **kw)

    # ------------------------------------------------------------------

    def to_list(self, for_file=False):
        page = {"width": self.width_combo.currentText(), "cols": self.cols.value(),
                "height": self.height_spin.value(), "font": self.font_spin.value(),
                "letters": self.letters.isChecked()}
        return [{"page": page}] + [
            {"spec": p["spec"], "name": p["name"],
             "data": project.frame_to_json(p["df"]) if for_file else p["df"].copy()}
            for p in self.panels]

    def from_list(self, items):
        self.panels = []
        for it in items or []:
            if "page" in it:
                pg = it["page"]
                for w, key, setter in ((self.width_combo, "width", "setCurrentText"),
                                       (self.cols, "cols", "setValue"),
                                       (self.height_spin, "height", "setValue"),
                                       (self.font_spin, "font", "setValue"),
                                       (self.letters, "letters", "setChecked")):
                    if key in pg:
                        w.blockSignals(True)
                        getattr(w, setter)(pg[key])
                        w.blockSignals(False)
                continue
            data = it.get("data")
            df = project.frame_from_json(data) if isinstance(data, dict) else data
            if df is not None:
                self.panels.append({"spec": it["spec"], "df": df, "name": it.get("name", "")})
        self._refresh_list()
        self.render()
