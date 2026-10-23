# run.py
import sys
import os
import pandas as pd
import html
import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSettings
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QDockWidget, QFileDialog,
    QLabel, QStatusBar, QListWidget, QListWidgetItem, QComboBox, QTabWidget, QPushButton,
    QAbstractItemView, QTextEdit, QGroupBox, QDoubleSpinBox, QCheckBox, QScrollArea,
    QSpinBox, QLineEdit, QInputDialog, QMessageBox, QGridLayout, QColorDialog, QMenu,
    QTableView, QToolBar, QToolButton, QDialog, QDialogButtonBox, QHeaderView
)
from PySide6.QtGui import QAction, QKeySequence, QColor, QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.excel_loader import DataLoader, sheet_names, read_grid
from src.plotting import PlottingEngine
from src.filters import FilterEngine
from src.normalization import NormalizationEngine
from src.statistics_engine import StatisticsEngine, format_p, stars
from src.utils import SummaryStatistics
from src.export_engine import ExportEngine
from src.report_generator import PDFReportGenerator
from src import plate_reader as pr
from src import theme

NO_SELECTION = "(none)"
DATA_FILTER = "Data files (*.xlsx *.xlsm *.xls *.csv *.tsv *.txt)"
DATA_EXT = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt")


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


class PasteableTableWidget(QTableWidget):
    """
    A QTableWidget that supports copy/paste of a rectangular block of cells
    using the standard clipboard shortcuts (Ctrl+V / Cmd+V to paste, Ctrl+C /
    Cmd+C to copy) — the same tab-separated format Excel/Sheets uses, so a
    block copied straight out of a spreadsheet pastes in starting at
    whichever cell is currently selected.
    """
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_from_clipboard()
        elif event.matches(QKeySequence.StandardKey.Copy):
            self.copy_to_clipboard()
        else:
            super().keyPressEvent(event)

    def paste_from_clipboard(self):
        text = QApplication.clipboard().text()
        if not text:
            return
        rows = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        while rows and rows[-1] == "":
            rows.pop()
        if not rows:
            return

        start_row = self.currentRow() if self.currentRow() >= 0 else 0
        start_col = self.currentColumn() if self.currentColumn() >= 0 else 0

        for r_off, row_text in enumerate(rows):
            cells = row_text.split("\t")
            for c_off, val in enumerate(cells):
                r, c = start_row + r_off, start_col + c_off
                if r < self.rowCount() and c < self.columnCount():
                    item = self.item(r, c)
                    if item is None:
                        item = QTableWidgetItem()
                        self.setItem(r, c, item)
                    item.setText(val.strip())

    def copy_to_clipboard(self):
        selected = self.selectedIndexes()
        if not selected:
            return
        rows = sorted(set(idx.row() for idx in selected))
        cols = sorted(set(idx.column() for idx in selected))
        lines = []
        for r in rows:
            line_vals = []
            for c in cols:
                item = self.item(r, c)
                line_vals.append(item.text() if item is not None else "")
            lines.append("\t".join(line_vals))
        QApplication.clipboard().setText("\n".join(lines))


class PlateGridEditor(QWidget):
    """
    An editable spreadsheet-like grid shaped like a well plate (rows A, B, C...
    columns 1, 2, 3...), used both for entering raw plate-reader values and
    for building plate-map metadata layers (e.g. which strain/media is in
    each well). Supports typing values directly, quick-filling a selection,
    and loading a plate-shaped block straight out of an uploaded file.
    """
    def __init__(self, n_rows=8, n_cols=12, parent=None):
        super().__init__(parent)
        self.n_rows = n_rows
        self.n_cols = n_cols

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Fill selected cells with:"))
        self.fill_value_edit = QLineEdit()
        toolbar.addWidget(self.fill_value_edit)
        fill_btn = QPushButton("Fill")
        fill_btn.clicked.connect(self.fill_selected)
        toolbar.addWidget(fill_btn)
        paste_btn = QPushButton("Paste")
        paste_btn.setToolTip("Paste a block copied from Excel/Sheets, starting at the selected cell (Ctrl/Cmd+V also works)")
        toolbar.addWidget(paste_btn)
        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy the selected cells (Ctrl/Cmd+C also works)")
        toolbar.addWidget(copy_btn)
        load_btn = QPushButton("Load From File...")
        load_btn.clicked.connect(self.load_from_file)
        toolbar.addWidget(load_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_grid)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)
        layout.addWidget(QLabel(
            "<i>Tip: click a starting cell, then paste a block copied from Excel — it fills outward from there.</i>"))

        self.table = PasteableTableWidget()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        layout.addWidget(self.table)
        self.set_shape(n_rows, n_cols)

        paste_btn.clicked.connect(self.table.paste_from_clipboard)
        copy_btn.clicked.connect(self.table.copy_to_clipboard)

    def set_shape(self, n_rows, n_cols):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.table.setRowCount(n_rows)
        self.table.setColumnCount(n_cols)
        self.table.setVerticalHeaderLabels([pr.well_id(r, 0)[:-1] for r in range(n_rows)])
        self.table.setHorizontalHeaderLabels([str(c + 1) for c in range(n_cols)])
        for r in range(n_rows):
            for c in range(n_cols):
                if self.table.item(r, c) is None:
                    self.table.setItem(r, c, QTableWidgetItem(""))

    def fill_selected(self):
        val = self.fill_value_edit.text()
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.information(self, "No cells selected", "Select one or more cells in the grid first.")
            return
        for item in selected:
            item.setText(val)

    def clear_grid(self):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                self.table.item(r, c).setText("")

    def load_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Plate Grid", "", DATA_FILTER)
        if not file_path:
            return
        try:
            blocks = pr.find_plate_blocks(read_grid(file_path))
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Could not read this file: {e}")
            return
        if not blocks:
            QMessageBox.warning(self, "Plate not found",
                "Could not find a plate layout (row letters beside column numbers) "
                "in this file. Please check the file or enter values manually.")
            return
        block = blocks[0]
        if len(blocks) > 1:
            titles = [f"{b['title']}  ({b['n_rows']}×{b['n_cols']})" for b in blocks]
            choice, ok = QInputDialog.getItem(self, "Choose Grid",
                                              "This file has several plate grids:",
                                              titles, 0, False)
            if not ok:
                return
            block = blocks[titles.index(choice)]
        self.set_shape(max(self.n_rows, block["n_rows"]), max(self.n_cols, block["n_cols"]))
        self.set_cells(block["cells"])

    def set_cells(self, cells):
        """Fill the grid from a 2-D list of text (an imported plate block)."""
        for r in range(min(self.n_rows, len(cells))):
            for c in range(min(self.n_cols, len(cells[r]))):
                self.table.item(r, c).setText(cells[r][c])

    def get_grid_values(self):
        grid = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                item = self.table.item(r, c)
                row.append(item.text() if item is not None else "")
            grid.append(row)
        return grid


def suggest_readout_name(title):
    """'Raw Data (485-15/528-20 1)' -> 'GFP (485/528)', '(600)' -> 'OD600'."""
    import re
    m = re.search(r"\((\d{3})(?:-\d+)?/(\d{3})", title)
    norm = " / OD" if re.search(r"normali", title, re.I) else ""
    if m:
        ex, em = int(m.group(1)), int(m.group(2))
        name = ("GFP" if 470 <= ex <= 500 and 500 <= em <= 545 else
                "mKate" if 560 <= ex <= 600 and em >= 610 else
                "RFP" if 540 <= ex <= 590 and em >= 590 else
                "BFP" if ex < 420 else f"Ex{ex}")
        return f"{name}{norm} ({ex}/{em})"
    m = re.search(r"\((\d{3})\)", title)
    if m:
        return f"OD{m.group(1)}{norm}"
    return title


class PlateImportDialog(QDialog):
    """Pick which grids of a plate-reader export to import, name them, and
    say whether each is a readout or a plate map."""

    KINDS = ["Readout (numbers)", "Plate map (labels)"]

    def __init__(self, blocks, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Plate-Reader File")
        self.resize(820, 360)
        self.blocks = blocks
        lay = QVBoxLayout(self)
        head = QLabel(f"<b>{len(blocks)} plate grid{'s' if len(blocks) != 1 else ''}</b> "
                      f"found in {html.escape(os.path.basename(path))}. Tick the ones to "
                      "import and give each a short name (it becomes a column).")
        head.setWordWrap(True)
        lay.addWidget(head)
        self.table = QTableWidget(len(blocks), 5)
        self.table.setHorizontalHeaderLabels(["Import", "Name", "Type", "Size", "Check"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for i, b in enumerate(blocks):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(i, 0, chk)
            name = QTableWidgetItem(b["title"] if b["is_map"] else
                                    suggest_readout_name(b["title"]))
            name.setToolTip(f"In the file: {b['title']}")
            self.table.setItem(i, 1, name)
            kind = QComboBox()
            kind.addItems(self.KINDS)
            kind.setCurrentIndex(1 if b["is_map"] else 0)
            self.table.setCellWidget(i, 2, kind)
            size = QTableWidgetItem(f"{b['n_rows']} × {b['n_cols']}")
            size.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, 3, size)
            notes = []
            if b["saturated"]:
                top = max(v for row in b["values"] for v in row if v is not None)
                notes.append(f"⚠ {b['saturated']} wells at the maximum ({top:g}) — "
                             "detector probably saturated")
            if b["overflow"]:
                notes.append(f"⚠ {b['overflow']} overflow cells (left empty)")
            note = QTableWidgetItem("; ".join(notes) or "OK")
            note.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if notes:
                note.setForeground(QColor("#B45309"))
            self.table.setItem(i, 4, note)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)
        hint = QLabel("Then add a plate map (what is in each well — strain, medium…) "
                      "and press Build Dataset From Plate.")
        hint.setObjectName("Hint")
        lay.addWidget(hint)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def chosen(self):
        """[(block, name, is_map)] for the ticked rows."""
        out = []
        for i, b in enumerate(self.blocks):
            if self.table.item(i, 0).checkState() != Qt.CheckState.Checked:
                continue
            name = self.table.item(i, 1).text().strip() or b["title"]
            out.append((b, name, self.table.cellWidget(i, 2).currentIndex() == 1))
        return out


class DropZoneWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel("Drop Excel or CSV files here\n\nor use Open Data… (⌘O)\n\n"
                            "Plate-reader exports: Import Plate… (⇧⌘I)")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("dropLabel")
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setAcceptDrops(True)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()
                 if url.toLocalFile().lower().endswith(DATA_EXT)]
        if paths: self.parent_window.process_files(paths)


class BioStatExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioStat Explorer")
        self.resize(1500, 950)

        self.data_loader = DataLoader()
        self.master_df = None
        self.filtered_df = None
        self.current_summary_df = None
        self.is_dark_mode = False
        # Manual color overrides set by clicking a group's x-tick label or
        # legend entry on the plot; keyed by group/hue-level name (str).
        # Cleared whenever a new file is loaded (see load_files/clear_data).
        self.group_colors = {}
        # Manual title/axis-label overrides set by clicking the title or an
        # axis label on the plot. Titles/ylabels are keyed by the metric
        # (y_col) being plotted, since each subplot can show a different one;
        # the x-axis label is shared across all subplots.
        self.custom_titles = {}
        self.custom_ylabels = {}
        self.custom_xlabel = None
        # Manual x-axis group ordering, set by right-clicking a group's
        # x-tick label or legend entry and choosing a "Move..." action.
        self.group_order = []

        # Populated on load: which columns are recognized signal/value columns
        # vs. which are metadata/categorical columns usable for filtering.
        self.recognized_cols = []
        self.metric_cols = []
        self.metadata_cols = []
        self.metadata_checklists = {}   # {column_name: QListWidget}

        self.init_central_widget()
        self.init_dock_panels()
        self.init_menu()
        self.apply_theme()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _action(self, menu, text, slot, shortcut=None, tip=None):
        act = QAction(text, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if tip:
            act.setToolTip(tip)
        menu.addAction(act)
        return act

    def init_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        self._action(file_menu, "Open Data…", self.open_file_dialog,
                     QKeySequence.StandardKey.Open)
        self._action(file_menu, "Add Data (append)…",
                     lambda: self.open_file_dialog(append=True))
        self._action(file_menu, "Import Plate-Reader File…",
                     self.import_plate_file, "Ctrl+Shift+I")
        file_menu.addSeparator()
        self.export_menu = file_menu.addMenu("Export")
        self._action(self.export_menu, "Figure (PNG, 600 dpi)…",
                     lambda: self.export_current_plot("png"), "Ctrl+Shift+E")
        self._action(self.export_menu, "Figure (SVG — editable in Inkscape)…",
                     lambda: self.export_current_plot("svg"))
        self._action(self.export_menu, "Figure (PDF)…",
                     lambda: self.export_current_plot("pdf"))
        self._action(self.export_menu, "Figure (TIFF, 600 dpi)…",
                     lambda: self.export_current_plot("tiff"))
        self.export_menu.addSeparator()
        self._action(self.export_menu, "Statistics (Excel)…",
                     self.export_stats_data, "Ctrl+E")
        self._action(self.export_menu, "PDF Report…", self.export_pdf_report)
        self._action(self.export_menu, "Data Table (CSV)…", self.export_data_csv)
        file_menu.addSeparator()
        self._action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        view_menu = menu_bar.addMenu("&View")
        self.dark_action = QAction("Dark Mode", self, checkable=True)
        self.dark_action.toggled.connect(self.toggle_dark_theme)
        view_menu.addAction(self.dark_action)
        view_menu.addAction(self.dock.toggleViewAction() if hasattr(self, "dock")
                            else QAction("Filters", self))

        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(tb)
        brand = QLabel("  BioStat Explorer   ")
        brand.setObjectName("Title")
        tb.addWidget(brand)
        open_btn = QToolButton()
        open_btn.setText("＋ Open Data")
        open_btn.setObjectName("Primary")
        open_btn.clicked.connect(self.open_file_dialog)
        tb.addWidget(open_btn)
        for text, slot in (("Import Plate", self.import_plate_file),
                           ("Run Statistics", self._run_stats_from_toolbar)):
            b = QToolButton()
            b.setText(text)
            b.clicked.connect(slot)
            tb.addWidget(b)
        exp = QToolButton()
        exp.setText("Export ▾")
        exp.setMenu(self.export_menu)
        exp.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tb.addWidget(exp)

    def init_central_widget(self):
        self.tabs = QTabWidget()

        # Tab 1: Data View
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.drop_zone = DropZoneWidget(self)
        self.table_model = PandasModel()
        self.table_viewer = QTableView()
        self.table_viewer.setModel(self.table_model)
        self.table_viewer.setAlternatingRowColors(True)
        self.table_viewer.setSortingEnabled(False)
        self.table_viewer.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.table_viewer.hide()
        self.data_layout.addWidget(self.drop_zone)
        self.data_layout.addWidget(self.table_viewer)

        # Tab 2: Graphing View
        self.graph_tab = QWidget()
        self.graph_layout = QHBoxLayout(self.graph_tab)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setMinimumWidth(330)
        control_host = QWidget()
        self.control_panel = QVBoxLayout(control_host)
        control_scroll.setWidget(control_host)

        # --- Styling ---
        style_group = QGroupBox("Look")
        style_box = QVBoxLayout()
        self.dark_mode_cb = QCheckBox("Enable Dark Mode")
        self.dark_mode_cb.toggled.connect(self.toggle_dark_theme)
        style_box.addWidget(self.dark_mode_cb)

        style_box.addWidget(QLabel("Plot Color Palette:"))
        self.palette_selector = QComboBox()
        self.palette_selector.addItems(["muted", "deep", "colorblind", "rocket", "crest"])
        self.palette_selector.currentTextChanged.connect(self.update_plot)
        style_box.addWidget(self.palette_selector)

        self.bw_mode_cb = QCheckBox("Black && white (grey + hatching)")
        self.bw_mode_cb.setToolTip(
            "Overrides colors with grayscale shades, and adds hatch patterns "
            "(bars/histograms) or distinct line styles/markers (line plots) "
            "so groups stay distinguishable without color - handy for print.")
        self.bw_mode_cb.toggled.connect(self.update_plot)
        style_box.addWidget(self.bw_mode_cb)

        reset_colors_btn = QPushButton("Reset custom colors")
        reset_colors_btn.setToolTip("Click a bar/box's x-axis label (or a legend entry) on the graph to recolor it. This clears all such overrides.")
        reset_colors_btn.clicked.connect(self.reset_group_colors)
        style_box.addWidget(reset_colors_btn)

        reset_order_btn = QPushButton("Reset sample order")
        reset_order_btn.setToolTip("Right-click a group's x-axis label (or a legend entry) on the graph to move it left/right. This clears all such overrides.")
        reset_order_btn.clicked.connect(self.reset_group_order)
        style_box.addWidget(reset_order_btn)

        reset_labels_btn = QPushButton("Reset title / axis labels")
        reset_labels_btn.setToolTip("Click the graph title or an axis label to edit it. This restores the auto-generated text.")
        reset_labels_btn.clicked.connect(self.reset_custom_labels)
        style_box.addWidget(reset_labels_btn)

        style_group.setLayout(style_box)
        self.control_panel.addWidget(style_group)

        # --- Data selection: axis, metric, grouping ---
        axis_group = QGroupBox("Data To Plot")
        axis_box = QVBoxLayout()

        axis_box.addWidget(QLabel("<b>X Axis (category):</b>"))
        self.x_selector = QComboBox()
        self.x_selector.currentTextChanged.connect(self.on_x_axis_changed)
        axis_box.addWidget(self.x_selector)

        axis_box.addWidget(QLabel("<b>Metric (Y axis):</b>"))
        self.y_selector = QComboBox()
        self.y_selector.currentTextChanged.connect(self.update_plot)
        axis_box.addWidget(self.y_selector)

        self.show_all_metrics_cb = QCheckBox("Show all recognized metrics (stacked subplots)")
        self.show_all_metrics_cb.toggled.connect(self.update_plot)
        axis_box.addWidget(self.show_all_metrics_cb)

        axis_box.addWidget(QLabel("<b>Group / Color By (optional):</b>"))
        self.hue_selector = QComboBox()
        self.hue_selector.currentTextChanged.connect(self.update_plot)
        axis_box.addWidget(self.hue_selector)

        axis_group.setLayout(axis_box)
        self.control_panel.addWidget(axis_group)

        # --- Chart type ---
        self.control_panel.addWidget(QLabel("<b>Graph Type:</b>"))
        self.chart_selector = QComboBox()
        self.chart_selector.addItems(["Bar", "Line", "Boxplot", "Violin", "Beeswarm", "Box + Swarm", "Scatter",
                                      "Heatmap", "Volcano Plot", "Dendrogram", "Network Graph", "Stacked Histogram",
                                      "Stacked Bar"])
        self.chart_selector.currentTextChanged.connect(self.update_plot)
        self.control_panel.addWidget(self.chart_selector)

        self.chart_help_label = QLabel(
            "<i>Heatmap: X=rows, Group/Color-By=columns (or leave Group/Color-By empty to "
            "profile all numeric metrics per X group).<br>"
            "Volcano: X=log2 fold-change column, Metric=p-value column, "
            "Group/Color-By (optional)=point label.<br>"
            "Dendrogram/Network: use every numeric metric column automatically.<br>"
            "Stacked Histogram: bins the Metric column's values; stacks one layer per "
            "Group/Color-By level (or per X category if Group/Color-By is empty).<br>"
            "Stacked Bar: one bar per X category, height = sum of Metric values in that "
            "category, shown as stacked segments (per Group/Color-By level if set, "
            "otherwise one segment per row).</i>"
        )
        self.chart_help_label.setWordWrap(True)
        self.chart_help_label.setStyleSheet("color: gray; font-size: 10px;")
        self.chart_help_label.setVisible(False)
        self.control_panel.addWidget(self.chart_help_label)
        self.chart_selector.currentTextChanged.connect(self._update_chart_help_visibility)

        # --- Significance overlay ---
        stats_overlay_group = QGroupBox("Significance Overlay")
        stats_overlay_box = QVBoxLayout()

        stats_overlay_box.addWidget(QLabel("Statistical Test:"))
        self.stat_mode_selector = QComboBox()
        self.stat_mode_selector.addItems(StatisticsEngine.ALL_TESTS)
        self.stat_mode_selector.currentTextChanged.connect(self.update_plot)
        stats_overlay_box.addWidget(self.stat_mode_selector)

        stats_overlay_box.addWidget(QLabel("Compare To (reference group):"))
        self.reference_selector = QComboBox()
        self.reference_selector.currentTextChanged.connect(self.update_plot)
        stats_overlay_box.addWidget(self.reference_selector)

        stats_overlay_box.addWidget(QLabel("Pairing Column (for paired tests only):"))
        self.pairing_selector = QComboBox()
        self.pairing_selector.currentTextChanged.connect(self.update_plot)
        stats_overlay_box.addWidget(self.pairing_selector)

        self.show_pvalue_cb = QCheckBox("Show p-values on plot")
        self.show_pvalue_cb.toggled.connect(self.update_plot)
        stats_overlay_box.addWidget(self.show_pvalue_cb)
        self.show_ns_cb = QCheckBox("Also show non-significant (ns)")
        self.show_ns_cb.toggled.connect(self.update_plot)
        stats_overlay_box.addWidget(self.show_ns_cb)
        overlay_hint = QLabel("Stars use the same tests and corrections as the "
                              "Statistical Analysis tab. With a Group / Color By "
                              "column, colours are compared within each X group.")
        overlay_hint.setWordWrap(True)
        overlay_hint.setObjectName("Hint")
        stats_overlay_box.addWidget(overlay_hint)

        stats_overlay_group.setLayout(stats_overlay_box)
        self.control_panel.addWidget(stats_overlay_group)

        # --- Options ---
        options_group = QGroupBox("Options")
        options_box = QVBoxLayout()
        self.log_scale_cb = QCheckBox("Log scale")
        self.log_scale_cb.toggled.connect(self.update_plot)
        options_box.addWidget(self.log_scale_cb)
        self.normalize_ref_cb = QCheckBox("Normalize to reference group (fold change)")
        self.normalize_ref_cb.toggled.connect(self.update_plot)
        options_box.addWidget(self.normalize_ref_cb)

        self.manual_yrange_cb = QCheckBox("Manual Y-axis range")
        self.manual_yrange_cb.toggled.connect(self.update_plot)
        options_box.addWidget(self.manual_yrange_cb)
        yrange_row = QHBoxLayout()
        yrange_row.addWidget(QLabel("Min:"))
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1e9, 1e9)
        self.y_min_spin.setDecimals(3)
        self.y_min_spin.valueChanged.connect(self.update_plot)
        yrange_row.addWidget(self.y_min_spin)
        yrange_row.addWidget(QLabel("Max:"))
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-1e9, 1e9)
        self.y_max_spin.setDecimals(3)
        self.y_max_spin.setValue(100.0)
        self.y_max_spin.valueChanged.connect(self.update_plot)
        yrange_row.addWidget(self.y_max_spin)
        options_box.addLayout(yrange_row)

        options_box.addWidget(QLabel("Histogram bins (Stacked Histogram only):"))
        self.hist_bins_spin = QSpinBox()
        self.hist_bins_spin.setRange(2, 200)
        self.hist_bins_spin.setValue(20)
        self.hist_bins_spin.valueChanged.connect(self.update_plot)
        options_box.addWidget(self.hist_bins_spin)

        options_group.setLayout(options_box)
        self.control_panel.addWidget(options_group)

        # --- Axis break (gap) ---
        axis_break_group = QGroupBox("X-Axis Break (Gap)")
        axis_break_box = QVBoxLayout()
        self.axis_break_cb = QCheckBox("Enable X-axis break")
        self.axis_break_cb.setToolTip(
            "Splits the plot into two side-by-side panels with a gap between them - "
            "e.g. show 1->2, skip a gap, then show 3->4 - instead of one continuous axis. "
            "Only applies to Line plots (a numeric, continuous X axis).")
        self.axis_break_cb.toggled.connect(self.update_plot)
        axis_break_box.addWidget(self.axis_break_cb)

        left_range_row = QHBoxLayout()
        left_range_row.addWidget(QLabel("Left panel X range:"))
        self.axis_break_left_min = QDoubleSpinBox()
        self.axis_break_left_min.setRange(-1e9, 1e9)
        self.axis_break_left_min.setValue(1.0)
        self.axis_break_left_min.valueChanged.connect(self.update_plot)
        left_range_row.addWidget(self.axis_break_left_min)
        left_range_row.addWidget(QLabel("to"))
        self.axis_break_left_max = QDoubleSpinBox()
        self.axis_break_left_max.setRange(-1e9, 1e9)
        self.axis_break_left_max.setValue(2.0)
        self.axis_break_left_max.valueChanged.connect(self.update_plot)
        left_range_row.addWidget(self.axis_break_left_max)
        axis_break_box.addLayout(left_range_row)

        right_range_row = QHBoxLayout()
        right_range_row.addWidget(QLabel("Right panel X range:"))
        self.axis_break_right_min = QDoubleSpinBox()
        self.axis_break_right_min.setRange(-1e9, 1e9)
        self.axis_break_right_min.setValue(3.0)
        self.axis_break_right_min.valueChanged.connect(self.update_plot)
        right_range_row.addWidget(self.axis_break_right_min)
        right_range_row.addWidget(QLabel("to"))
        self.axis_break_right_max = QDoubleSpinBox()
        self.axis_break_right_max.setRange(-1e9, 1e9)
        self.axis_break_right_max.setValue(4.0)
        self.axis_break_right_max.valueChanged.connect(self.update_plot)
        right_range_row.addWidget(self.axis_break_right_max)
        axis_break_box.addLayout(right_range_row)

        axis_break_group.setLayout(axis_break_box)
        self.control_panel.addWidget(axis_break_group)

        # --- Data Selection (per-value checkboxes, like Strains/Organism/Media) ---
        data_selection_group = QGroupBox("Filter Values")
        self.data_selection_layout = QVBoxLayout()
        data_selection_group.setLayout(self.data_selection_layout)
        self.control_panel.addWidget(data_selection_group)

        self.plot_btn = QPushButton("Refresh Graph Layout")
        self.plot_btn.setObjectName("primaryBtn")
        self.plot_btn.clicked.connect(self.update_plot)
        self.control_panel.addWidget(self.plot_btn)

        self.control_panel.addWidget(QLabel("<br><b>Export Graphics:</b>"))
        self.btn_export_png = QPushButton("Export Figure (PNG)")
        self.btn_export_png.clicked.connect(lambda: self.export_current_plot("png"))
        self.control_panel.addWidget(self.btn_export_png)
        self.btn_export_svg = QPushButton("Export Graphic (SVG)")
        self.btn_export_svg.clicked.connect(lambda: self.export_current_plot("svg"))
        self.control_panel.addWidget(self.btn_export_svg)
        self.control_panel.addStretch()

        self.plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_canvas.mpl_connect("pick_event", self.on_plot_pick)
        self.ax = self.plot_figure.add_subplot(111)
        self.axes = [self.ax]
        PlottingEngine.generate_plot(None, "", "", "", self.ax)
        self.graph_layout.addWidget(control_scroll, stretch=1)
        self.graph_layout.addWidget(self.plot_canvas, stretch=3)

        # Tab 3: Analytics View
        self.analysis_tab = QWidget()
        self.analysis_layout = QHBoxLayout(self.analysis_tab)
        analysis_controls = QVBoxLayout()

        norm_group = QGroupBox("Data Normalization")
        norm_box = QVBoxLayout()
        norm_box.addWidget(QLabel("Background Baseline:"))
        self.bg_spin = QDoubleSpinBox()
        self.bg_spin.setRange(-100000, 100000)
        norm_box.addWidget(self.bg_spin)
        self.btn_bg_sub = QPushButton("Apply Subtraction")
        self.btn_bg_sub.clicked.connect(self.run_bg_sub)
        norm_box.addWidget(self.btn_bg_sub)
        self.btn_wt_fold = QPushButton("Calculate Fold Change vs Reference")
        self.btn_wt_fold.clicked.connect(self.run_wt_fold)
        norm_box.addWidget(self.btn_wt_fold)
        norm_group.setLayout(norm_box)
        analysis_controls.addWidget(norm_group)

        stat_group = QGroupBox("Statistics")
        stat_box = QVBoxLayout()
        self.btn_run_stats = QPushButton("Run Statistical Test")
        self.btn_run_stats.setObjectName("successBtn")
        self.btn_run_stats.clicked.connect(self.run_statistical_pipeline)
        stat_box.addWidget(self.btn_run_stats)

        self.btn_export_stats = QPushButton("Export to Excel…")
        self.btn_export_stats.clicked.connect(self.export_stats_data)
        stat_box.addWidget(self.btn_export_stats)

        self.btn_export_pdf = QPushButton("PDF Report…")
        self.btn_export_pdf.setObjectName("dangerBtn")
        self.btn_export_pdf.clicked.connect(self.export_pdf_report)
        stat_box.addWidget(self.btn_export_pdf)

        stat_group.setLayout(stat_box)
        analysis_controls.addWidget(stat_group)
        analysis_controls.addStretch()

        right_analysis_layout = QVBoxLayout()
        right_analysis_layout.addWidget(QLabel("<b>Descriptive Group Summaries:</b>"))
        self.summary_table_view = QTableWidget()
        right_analysis_layout.addWidget(self.summary_table_view, stretch=1)

        right_analysis_layout.addWidget(QLabel("<b>Report</b> — test, post-hoc comparisons, effect sizes"))
        self.stats_output = QTextEdit()
        self.stats_output.setReadOnly(True)
        right_analysis_layout.addWidget(self.stats_output, stretch=2)

        self.analysis_layout.addLayout(analysis_controls, stretch=1)
        self.analysis_layout.addLayout(right_analysis_layout, stretch=2)

        self.init_plate_reader_tab()

        self.tabs.addTab(self.data_tab, "Data Viewer")
        self.tabs.addTab(self.plate_tab, "Plate Reader Import")
        self.tabs.addTab(self.graph_tab, "Plotting Dashboard")
        self.tabs.addTab(self.analysis_tab, "Statistical Analysis")
        self.setCentralWidget(self.tabs)

    def init_plate_reader_tab(self):
        self.plate_tab = QWidget()
        outer_layout = QVBoxLayout(self.plate_tab)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("<b>Plate Size:</b>"))
        self.plate_size_selector = QComboBox()
        self.plate_size_selector.addItems(list(pr.PLATE_SHAPES.keys()))
        self.plate_size_selector.setCurrentText("96-well (8x12)")
        self.plate_size_selector.currentTextChanged.connect(self.on_plate_size_changed)
        top_bar.addWidget(self.plate_size_selector)
        import_btn = QPushButton("Import Plate-Reader File…")
        import_btn.setObjectName("primaryBtn")
        import_btn.setToolTip("Reads every plate grid in an instrument export (e.g. BMG "
                              "CLARIOstar: GFP, mKate, OD, normalised) — or a plate map")
        import_btn.clicked.connect(self.import_plate_file)
        top_bar.addWidget(import_btn)
        top_bar.addStretch()
        outer_layout.addLayout(top_bar)

        splitter_layout = QHBoxLayout()

        # Left: raw plate values — one grid per readout (GFP plate, mKate plate,
        # a timepoint, etc.), addable/removable just like the map layers on the right.
        values_group = QGroupBox("1. Raw Plate Values (add one grid per readout / metric)")
        values_layout = QVBoxLayout()
        value_controls = QHBoxLayout()
        self.value_name_edit = QLineEdit()
        self.value_name_edit.setPlaceholderText("e.g. GFP, mKate, OD600, Timepoint_1...")
        value_controls.addWidget(self.value_name_edit)
        add_value_btn = QPushButton("Add Value Grid")
        add_value_btn.clicked.connect(self.add_plate_value_layer)
        value_controls.addWidget(add_value_btn)
        remove_value_btn = QPushButton("Remove Current Grid")
        remove_value_btn.clicked.connect(self.remove_plate_value_layer)
        value_controls.addWidget(remove_value_btn)
        values_layout.addLayout(value_controls)

        self.value_layers_tabs = QTabWidget()
        values_layout.addWidget(self.value_layers_tabs)
        values_group.setLayout(values_layout)
        splitter_layout.addWidget(values_group, stretch=1)

        # Right: plate map layers (metadata per well, e.g. Strain, Media)
        map_group = QGroupBox("2. Plate Map (what's in each well — add one layer per metadata field)")
        map_layout = QVBoxLayout()
        layer_controls = QHBoxLayout()
        self.layer_name_edit = QLineEdit()
        self.layer_name_edit.setPlaceholderText("e.g. Strain, Media, Organism...")
        layer_controls.addWidget(self.layer_name_edit)
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_plate_map_layer)
        layer_controls.addWidget(add_layer_btn)
        remove_layer_btn = QPushButton("Remove Current Layer")
        remove_layer_btn.clicked.connect(self.remove_plate_map_layer)
        layer_controls.addWidget(remove_layer_btn)
        map_layout.addLayout(layer_controls)

        self.map_layers_tabs = QTabWidget()
        map_layout.addWidget(self.map_layers_tabs)
        map_group.setLayout(map_layout)
        splitter_layout.addWidget(map_group, stretch=1)

        outer_layout.addLayout(splitter_layout, stretch=1)

        build_bar = QHBoxLayout()
        build_btn = QPushButton("Build Dataset From Plate")
        build_btn.setObjectName("primaryBtn")
        build_btn.clicked.connect(self.build_dataset_from_plate)
        build_bar.addWidget(build_btn)
        build_bar.addWidget(QLabel(
            "Tip: add a value grid per readout (e.g. 'GFP', 'mKate') and a map layer per "
            "metadata field (e.g. 'Strain') so wells can be grouped/filtered afterwards."))
        outer_layout.addLayout(build_bar)

        self.value_layer_editors = {}
        self.map_layer_editors = {}
        # Start with one sensible default of each so the tab isn't empty
        self._new_plate_grid_tab(self.value_layers_tabs, self.value_layer_editors, "Value")
        self._new_plate_grid_tab(self.map_layers_tabs, self.map_layer_editors, "Strain")

    def on_plate_size_changed(self, size_name):
        n_rows, n_cols = pr.PLATE_SHAPES[size_name]
        for editor in self.value_layer_editors.values():
            editor.set_shape(n_rows, n_cols)
        for editor in self.map_layer_editors.values():
            editor.set_shape(n_rows, n_cols)

    def _new_plate_grid_tab(self, tabs_widget, editors_dict, name):
        """Shared helper: adds a new named PlateGridEditor tab to either the
        value-layers or map-layers QTabWidget."""
        if not name or name in editors_dict:
            return None
        n_rows, n_cols = pr.PLATE_SHAPES[self.plate_size_selector.currentText()]
        editor = PlateGridEditor(n_rows, n_cols)
        editors_dict[name] = editor
        tabs_widget.addTab(editor, name)
        return editor

    def add_plate_value_layer(self):
        name = self.value_name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Name needed", "Type a name for this readout first (e.g. 'GFP').")
            return
        if name in self.value_layer_editors:
            QMessageBox.information(self, "Grid exists", f"A value grid named '{name}' already exists.")
            return
        self._new_plate_grid_tab(self.value_layers_tabs, self.value_layer_editors, name)
        self.value_name_edit.clear()

    def remove_plate_value_layer(self):
        idx = self.value_layers_tabs.currentIndex()
        if idx < 0:
            return
        name = self.value_layers_tabs.tabText(idx)
        self.value_layers_tabs.removeTab(idx)
        self.value_layer_editors.pop(name, None)

    def add_plate_map_layer(self):
        name = self.layer_name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Layer name needed", "Type a name for this metadata field first (e.g. 'Strain').")
            return
        if name in self.map_layer_editors:
            QMessageBox.information(self, "Layer exists", f"A layer named '{name}' already exists.")
            return
        self._new_plate_grid_tab(self.map_layers_tabs, self.map_layer_editors, name)
        self.layer_name_edit.clear()

    def remove_plate_map_layer(self):
        idx = self.map_layers_tabs.currentIndex()
        if idx < 0:
            return
        name = self.map_layers_tabs.tabText(idx)
        self.map_layers_tabs.removeTab(idx)
        self.map_layer_editors.pop(name, None)

    @staticmethod
    def _grid_is_empty(editor):
        return all(not v.strip() for row in editor.get_grid_values() for v in row)

    def import_plate_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Plate-Reader File",
                                              self._last_dir(), DATA_FILTER)
        if not path:
            return
        self._remember_dir(path)
        sheet, ok = self._choose_sheet(path, "plate grids")
        if not ok:
            return
        try:
            blocks = pr.find_plate_blocks(read_grid(path, sheet))
        except Exception as e:
            QMessageBox.warning(self, "Could not read", str(e))
            return
        if not blocks:
            QMessageBox.information(
                self, "No plate grid found",
                "No plate-shaped grid (column numbers 1, 2, 3… with row letters A, B, "
                "C… beside them) was found. If this file is a plain table (one row "
                "per sample), use Open Data instead.")
            return
        dlg = PlateImportDialog(blocks, path, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dlg.chosen()
        if not chosen:
            return
        shape = pr.shape_name(max(b["n_rows"] for b, _, _ in chosen),
                              max(b["n_cols"] for b, _, _ in chosen))
        if shape != self.plate_size_selector.currentText():
            self.plate_size_selector.setCurrentText(shape)
        for block, name, is_map in chosen:
            editors = self.map_layer_editors if is_map else self.value_layer_editors
            tabs = self.map_layers_tabs if is_map else self.value_layers_tabs
            # an untouched default grid makes way for the imported ones
            for old in [n for n, e in editors.items() if n in ("Value", "Strain")
                        and n != name and self._grid_is_empty(e)]:
                tabs.removeTab(tabs.indexOf(editors.pop(old)))
            editor = editors.get(name) or self._new_plate_grid_tab(tabs, editors, name)
            editor.set_cells(block["cells"])
            tabs.setCurrentWidget(editor)
        self.tabs.setCurrentWidget(self.plate_tab)
        warn = [f"{n}: {b['saturated']} saturated wells" for b, n, _ in chosen if b["saturated"]]
        self.status_bar.showMessage(
            f"Imported {len(chosen)} grid(s) from {os.path.basename(path)}"
            + (" — " + "; ".join(warn) if warn else ""), 10000)

    def build_dataset_from_plate(self):
        n_rows, n_cols = pr.PLATE_SHAPES[self.plate_size_selector.currentText()]

        if not self.value_layer_editors:
            QMessageBox.warning(self, "No value grids", "Add at least one raw value grid first (e.g. 'GFP').")
            return

        value_dfs = []
        for name, editor in self.value_layer_editors.items():
            grid = editor.get_grid_values()
            value_dfs.append(pr.grid_to_long(grid, value_name=name, n_rows=n_rows, n_cols=n_cols))

        merged_values = value_dfs[0]
        for vdf in value_dfs[1:]:
            merged_values = merged_values.merge(vdf, on=["Well", "Row", "Column"], how="outer")

        metric_names = list(self.value_layer_editors.keys())
        if merged_values[metric_names].dropna(how="all").empty:
            QMessageBox.warning(self, "No values found",
                "All raw plate value grids are empty. Type values in, or use 'Load From File...' first.")
            return

        layer_dfs = []
        for name, editor in self.map_layer_editors.items():
            grid = editor.get_grid_values()
            layer_dfs.append(pr.map_grid_to_long(grid, name, n_rows, n_cols))

        merged = pr.merge_well_layers(merged_values, layer_dfs)
        # empty wells (no reading at all) aren't samples
        merged = merged.dropna(subset=metric_names, how="all").reset_index(drop=True)

        append = False
        if self.master_df is not None:
            choice = QMessageBox.question(
                self, "Existing data found",
                "You already have data loaded. Append this plate as additional rows, "
                "or replace the current dataset entirely?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            append = (choice == QMessageBox.StandardButton.Yes)

        self.ingest_dataframe(merged, append=append, source_label="Plate Reader Import")
        self.tabs.setCurrentWidget(self.graph_tab)

    def init_dock_panels(self):
        self.dock = QDockWidget("Filters", self)
        dock_contents = QWidget()
        dock_layout = QVBoxLayout(dock_contents)
        dock_layout.addWidget(QLabel("<b>1. Select Metadata Column:</b>"))
        self.metadata_col_list = QListWidget()
        self.metadata_col_list.currentTextChanged.connect(self.populate_sub_filter_options)
        dock_layout.addWidget(self.metadata_col_list, stretch=2)
        dock_layout.addWidget(QLabel("<b>2. Filter Unique Values:</b>"))
        self.value_filter_list = QListWidget()
        self.value_filter_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.value_filter_list.itemSelectionChanged.connect(self.execute_live_filter)
        dock_layout.addWidget(self.value_filter_list, stretch=3)

        reset_btn = QPushButton("Reset All Filters")
        reset_btn.clicked.connect(self.reset_all_filters)
        dock_layout.addWidget(reset_btn)

        self.dock.setWidget(dock_contents)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        # Remembers each column's selected values so switching which column
        # you're looking at doesn't discard the filter you already set on
        # another column — {column_name: [selected value strings]}.
        self.dock_filter_selections = {}

    def apply_theme(self):
        self.setStyleSheet(theme.stylesheet(self.is_dark_mode))

    def toggle_dark_theme(self, checked):
        self.is_dark_mode = checked
        for w in (getattr(self, "dark_mode_cb", None), getattr(self, "dark_action", None)):
            if w is not None and w.isChecked() != checked:
                w.blockSignals(True)
                w.setChecked(checked)
                w.blockSignals(False)
        self.apply_theme()
        self.update_plot()

    def open_file_dialog(self, append=False):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Data File(s)",
                                                self._last_dir(), DATA_FILTER)
        if files:
            self._remember_dir(files[0])
            self.process_files(files, append=bool(append))

    def _last_dir(self):
        return QSettings("BioStatExplorer", "BioStatExplorer").value("last_dir", "")

    def _remember_dir(self, path):
        QSettings("BioStatExplorer", "BioStatExplorer").setValue(
            "last_dir", os.path.dirname(path))

    def _choose_sheet(self, path, purpose="data"):
        """The sheet to read: asks when an Excel file has several."""
        try:
            names = sheet_names(path)
        except Exception:
            return None, True
        if len(names) <= 1:
            return (names[0] if names else None), True
        name, ok = QInputDialog.getItem(
            self, "Choose Sheet",
            f"{os.path.basename(path)} has {len(names)} sheets.\n"
            f"Which one holds the {purpose}?", names, 0, False)
        return name, ok

    def process_files(self, file_paths, append=False):
        sheets = {}
        for path in file_paths:
            sheet, ok = self._choose_sheet(path)
            if not ok:
                return
            if sheet:
                sheets[path] = sheet
        try:
            df, numeric, other, reports = self.data_loader.load_and_merge(
                file_paths, sheets)
        except Exception as e:
            QMessageBox.warning(self, "Could not open", str(e))
            return
        notes = [str(r) for r in reports if r.notes]
        if df.empty:
            QMessageBox.warning(self, "No data found",
                                "No table was found in:\n" + "\n".join(
                                    os.path.basename(p) for p in file_paths)
                                + ("\n\n" + "\n".join(notes) if notes else ""))
            return
        self.recognized_cols = numeric
        self.ingest_dataframe(df, append=append)
        if notes:
            self.status_bar.showMessage("Loaded — " + " | ".join(notes), 12000)

    def ingest_dataframe(self, df, append=False, source_label=None):
        """
        Shared finalization path for any newly-loaded dataset, whether it came
        from Excel/CSV files (process_files) or from the Plate Reader Import
        tab (build_dataset_from_plate). Classifies columns by dtype, and
        either replaces or appends to the current master dataset.
        """
        if df is None or df.empty:
            return
        if source_label and "Source File" not in df.columns:
            df = df.copy()
            df["Source File"] = source_label

        if append and self.master_df is not None:
            combined = pd.concat([self.master_df, df], ignore_index=True, sort=False)
        else:
            combined = df.copy()

        self.master_df = combined.copy()
        self.filtered_df = combined.copy()
        # Classify by actual dtype: any numeric column is a plottable metric,
        # everything else (Strain, Medium, Well...) is filterable metadata.
        self.metric_cols = [c for c in combined.columns if pd.api.types.is_numeric_dtype(combined[c])]
        self.metadata_cols = [c for c in combined.columns if c not in self.metric_cols]
        if "Source File" not in self.metadata_cols and "Source File" in combined.columns:
            self.metadata_cols.append("Source File")
        if not append:
            # a new dataset: forget the previous one's filters and labels
            self.dock_filter_selections = {}
            self.value_filter_list.clear()
            self.group_order = []
            self.group_colors = {}
        self.display_data(self.filtered_df)
        self.metadata_col_list.clear()
        self.metadata_col_list.addItems(self.metadata_cols)
        self.rebuild_data_selection_panel()      # before plotting with it
        self.update_selectors(choose_defaults=not append)
        self.status_bar.showMessage(f"Loaded {combined.shape[0]} rows, {combined.shape[1]} columns.", 4000)

    # ---------------------------------------------------------------
    # Selector population
    # ---------------------------------------------------------------
    def _default_x(self):
        """A column that groups the rows: text, 2-50 distinct values, not an
        ID (well, file, replicate)."""
        df = self.master_df
        skip = {"Well", "Row", "Column", "Source File"}
        best = None
        for c in self.metadata_cols:
            if c in skip:
                continue
            n = df[c].nunique()
            if 2 <= n <= 50 and n < len(df):
                if best is None:
                    best = c
                # a column where every group has replicates is the best guess
                if df[c].value_counts().min() >= 2:
                    return c
        return best

    def update_selectors(self, choose_defaults=False):
        cols = list(self.filtered_df.columns) if self.filtered_df is not None else []
        old_x, old_y = self.x_selector.currentText(), self.y_selector.currentText()
        self.x_selector.blockSignals(True)
        self.y_selector.blockSignals(True)
        self.x_selector.clear()
        self.y_selector.clear()
        self.x_selector.addItems(cols)
        # Prefer numeric metric columns for the Metric dropdown, if any exist
        metric_candidates = [c for c in getattr(self, 'metric_cols', []) if c in cols] or cols
        self.y_selector.addItems(metric_candidates)
        x = old_x if (old_x in cols and not choose_defaults) else (
            self._default_x() if self.master_df is not None else None)
        if x:
            self.x_selector.setCurrentText(x)
        if old_y in metric_candidates and not choose_defaults:
            self.y_selector.setCurrentText(old_y)
        self.x_selector.blockSignals(False)
        self.y_selector.blockSignals(False)
        self.rebuild_hue_and_pairing_selectors()
        self.rebuild_reference_selector()
        self.update_plot()

    def _update_chart_help_visibility(self, chart_type):
        self.chart_help_label.setVisible(chart_type in ("Heatmap", "Volcano Plot", "Dendrogram", "Network Graph", "Stacked Bar"))

    def on_x_axis_changed(self):
        self.rebuild_hue_and_pairing_selectors()
        self.rebuild_reference_selector()
        self.update_plot()

    def rebuild_hue_and_pairing_selectors(self):
        x_col = self.x_selector.currentText()
        candidates = [c for c in self.metadata_cols if c != x_col]

        for combo in (self.hue_selector, self.pairing_selector):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(NO_SELECTION)
            combo.addItems(candidates)
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def rebuild_reference_selector(self):
        x_col = self.x_selector.currentText()
        current = self.reference_selector.currentText()
        self.reference_selector.blockSignals(True)
        self.reference_selector.clear()
        self.reference_selector.addItem(NO_SELECTION)
        if self.master_df is not None and x_col in self.master_df.columns:
            self.reference_selector.addItems(FilterEngine.get_unique_values(self.master_df, x_col))
        idx = self.reference_selector.findText(current)
        self.reference_selector.setCurrentIndex(idx if idx >= 0 else 0)
        self.reference_selector.blockSignals(False)

    def rebuild_data_selection_panel(self):
        self._clear_layout(self.data_selection_layout)
        self.metadata_checklists = {}
        if self.master_df is None:
            return
        for col in self.metadata_cols:
            if col not in self.master_df.columns:
                continue
            group = QGroupBox(col)
            v = QVBoxLayout()
            lst = QListWidget()
            lst.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            for val in FilterEngine.get_unique_values(self.master_df, col):
                item = QListWidgetItem(val)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                lst.addItem(item)
            lst.itemChanged.connect(self.update_plot)
            v.addWidget(lst)
            group.setLayout(v)
            self.data_selection_layout.addWidget(group)
            self.metadata_checklists[col] = lst

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    self._clear_layout(sub_layout)

    def get_plot_filter_map(self):
        filter_map = {}
        for col, lst in self.metadata_checklists.items():
            checked = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.CheckState.Checked]
            filter_map[col] = checked
        return filter_map

    def get_active_plot_df(self):
        if self.filtered_df is None:
            return None
        return FilterEngine.apply_multi_filter(self.filtered_df, self.get_plot_filter_map())

    def get_selected_metrics(self):
        if self.show_all_metrics_cb.isChecked() and getattr(self, 'metric_cols', None):
            df_cols = list(self.filtered_df.columns) if self.filtered_df is not None else []
            return [c for c in self.metric_cols if c in df_cols]
        current = self.y_selector.currentText()
        return [current] if current else []

    def get_reference(self):
        ref = self.reference_selector.currentText()
        return None if (not ref or ref == NO_SELECTION) else ref

    def get_hue(self):
        hue = self.hue_selector.currentText()
        return None if (not hue or hue == NO_SELECTION) else hue

    def get_pairing_col(self):
        p = self.pairing_selector.currentText()
        return None if (not p or p == NO_SELECTION) else p

    # ---------------------------------------------------------------
    # Filtering (single-column dock, kept for backward compatibility)
    # ---------------------------------------------------------------
    def populate_sub_filter_options(self, target_column):
        if not target_column or self.master_df is None: return
        self.value_filter_list.blockSignals(True)
        self.value_filter_list.clear()
        self.value_filter_list.addItems(FilterEngine.get_unique_values(self.master_df, target_column))
        # Restore this column's previously-selected values, so switching to
        # another column and back doesn't lose what you'd already set here.
        previously_selected = set(self.dock_filter_selections.get(target_column, []))
        if previously_selected:
            for i in range(self.value_filter_list.count()):
                item = self.value_filter_list.item(i)
                if item.text() in previously_selected:
                    item.setSelected(True)
        self.value_filter_list.blockSignals(False)

    def execute_live_filter(self):
        if self.master_df is None: return
        current_item = self.metadata_col_list.currentItem()
        if current_item is None: return
        selected_column = current_item.text()
        selected_items = [item.text() for item in self.value_filter_list.selectedItems()]

        # Remember this column's selection (even once you switch to another
        # column), and combine every column's selection together (AND across
        # columns) rather than only ever applying the last one touched.
        if selected_items:
            self.dock_filter_selections[selected_column] = selected_items
        else:
            self.dock_filter_selections.pop(selected_column, None)

        self.filtered_df = FilterEngine.apply_multi_filter(self.master_df, self.dock_filter_selections)
        self.display_data(self.filtered_df)
        self.update_summary_table_view()
        self.update_plot()

    def reset_all_filters(self):
        """Clears every filter: the column/value dock selections above, and
        every checkbox in the Plotting Dashboard's Filter Values panel."""
        self.dock_filter_selections = {}
        self.value_filter_list.blockSignals(True)
        self.value_filter_list.clearSelection()
        self.value_filter_list.blockSignals(False)

        for lst in self.metadata_checklists.values():
            lst.blockSignals(True)
            for i in range(lst.count()):
                lst.item(i).setCheckState(Qt.CheckState.Checked)
            lst.blockSignals(False)

        self.filtered_df = self.master_df.copy() if self.master_df is not None else None
        if self.filtered_df is not None:
            self.display_data(self.filtered_df)
        self.update_summary_table_view()
        self.update_plot()

    def display_data(self, df):
        if df is None: return
        self.drop_zone.hide()
        self.table_viewer.show()
        self.table_model.set_frame(df)
        if len(df) <= 2000:
            self.table_viewer.resizeColumnsToContents()

    # ---------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------
    def reset_group_colors(self):
        self.group_colors = {}
        self.update_plot()

    def reset_group_order(self):
        self.group_order = []
        self.update_plot()

    def reset_custom_labels(self):
        self.custom_titles = {}
        self.custom_ylabels = {}
        self.custom_xlabel = None
        self.update_plot()

    def on_plot_pick(self, event):
        """
        Routes clicks on plot elements tagged with a URL (set in
        PlottingEngine.generate_plot) to the right editor:
          - 'group_color:<name>'  (x-tick label / legend entry)
              left-click  -> color picker (existing behavior)
              right-click -> reorder menu (move this group left/right/etc.)
          - 'edit_title:<metric>' / 'edit_xlabel' / 'edit_ylabel:<metric>'
              any click   -> text-entry dialog to override that label
        """
        artist = event.artist
        url = artist.get_url() if hasattr(artist, "get_url") else None
        if not url:
            return

        mouse_event = getattr(event, "mouseevent", None)
        button = mouse_event.button if mouse_event is not None else 1

        if url.startswith("group_color:"):
            group_name = url[len("group_color:"):]
            if button == 3:  # right-click
                self._show_group_reorder_menu(group_name)
            else:
                current = self.group_colors.get(group_name)
                initial = QColor(current) if current else QColor("white")
                chosen = QColorDialog.getColor(initial, self, f"Choose color for '{group_name}'")
                if chosen.isValid():
                    self.group_colors[group_name] = chosen.name()
                    self.update_plot()

        elif url.startswith("edit_title:"):
            metric = url[len("edit_title:"):]
            current = self.custom_titles.get(metric, artist.get_text())
            text, ok = QInputDialog.getText(self, "Edit Graph Title", "Title:", text=current)
            if ok:
                if text.strip():
                    self.custom_titles[metric] = text
                else:
                    self.custom_titles.pop(metric, None)  # blank -> revert to auto title
                self.update_plot()

        elif url == "edit_xlabel":
            current = self.custom_xlabel if self.custom_xlabel is not None else artist.get_text()
            text, ok = QInputDialog.getText(self, "Edit X-Axis Label", "X-axis label:", text=current)
            if ok:
                self.custom_xlabel = text if text.strip() else None
                self.update_plot()

        elif url.startswith("edit_ylabel:"):
            metric = url[len("edit_ylabel:"):]
            current = self.custom_ylabels.get(metric, artist.get_text())
            text, ok = QInputDialog.getText(self, "Edit Y-Axis Label", "Y-axis label:", text=current)
            if ok:
                if text.strip():
                    self.custom_ylabels[metric] = text
                else:
                    self.custom_ylabels.pop(metric, None)
                self.update_plot()

    def _current_group_order(self):
        """The active x-axis category order: manual overrides layered on top
        of the natural (first-seen-in-data) order, for whatever the plot is
        currently showing."""
        df = self.get_active_plot_df()
        x_col = self.x_selector.currentText()
        if df is None or not x_col or x_col not in df.columns:
            return []
        natural = list(df[x_col].dropna().unique().astype(str))
        if not self.group_order:
            return natural
        ordered = [g for g in self.group_order if g in natural]
        remaining = [g for g in natural if g not in ordered]
        return ordered + remaining

    def _show_group_reorder_menu(self, group_name):
        """Right-click context menu on a group's label/legend entry, offering
        to move it earlier/later along the x-axis (or a hue category's
        position within the legend/color cycle)."""
        order = self._current_group_order()
        if group_name not in order:
            return
        idx = order.index(group_name)

        menu = QMenu(self)
        act_left = menu.addAction("Move Left")
        act_left.setEnabled(idx > 0)
        act_right = menu.addAction("Move Right")
        act_right.setEnabled(idx < len(order) - 1)
        menu.addSeparator()
        act_start = menu.addAction("Move to Start")
        act_end = menu.addAction("Move to End")

        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return

        if chosen is act_left:
            order[idx - 1], order[idx] = order[idx], order[idx - 1]
        elif chosen is act_right:
            order[idx + 1], order[idx] = order[idx], order[idx + 1]
        elif chosen is act_start:
            order.insert(0, order.pop(idx))
        elif chosen is act_end:
            order.append(order.pop(idx))
        else:
            return

        self.group_order = order
        self.update_plot()

    def _apply_axis_break(self, ax_left, ax_right):
        """
        Standard matplotlib broken-axis technique: two panels share a y-axis,
        each showing a different X range, with the shared spine hidden and
        small diagonal marks drawn across the gap to indicate the break.
        """
        ax_left.set_xlim(self.axis_break_left_min.value(), self.axis_break_left_max.value())
        ax_right.set_xlim(self.axis_break_right_min.value(), self.axis_break_right_max.value())

        ax_left.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.tick_params(labelleft=False)
        ax_right.set_ylabel("")
        ax_right.set_title("")
        # Both panels draw their own legend (from _draw_line) since they're
        # independent generate_plot calls - keep only the right one so the
        # gap between panels doesn't end up with two overlapping legends.
        left_legend = ax_left.get_legend()
        if left_legend is not None:
            left_legend.remove()

        d = 0.4
        break_kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
                             color='k', mec='k', mew=1, clip_on=False)
        ax_left.plot([1], [0], transform=ax_left.transAxes, **break_kwargs)
        ax_left.plot([1], [1], transform=ax_left.transAxes, **break_kwargs)
        ax_right.plot([0], [0], transform=ax_right.transAxes, **break_kwargs)
        ax_right.plot([0], [1], transform=ax_right.transAxes, **break_kwargs)

    def update_plot(self):
        df = self.get_active_plot_df()
        x_col = self.x_selector.currentText()
        metrics = self.get_selected_metrics()

        if df is None or df.empty or not x_col or not metrics:
            self.plot_figure.clf()
            self.ax = self.plot_figure.add_subplot(111)
            self.axes = [self.ax]
            PlottingEngine.generate_plot(None, "", "", "", self.ax)
            self.plot_canvas.draw()
            return

        hue_col = self.get_hue()
        reference = self.get_reference()
        pairing_col = self.get_pairing_col()
        normalize = self.normalize_ref_cb.isChecked()
        log_scale = self.log_scale_cb.isChecked()
        chart_type = self.chart_selector.currentText()
        stat_mode = self.stat_mode_selector.currentText()
        bw_mode = self.bw_mode_cb.isChecked()
        hist_bins = self.hist_bins_spin.value()
        # Axis break (gap) only makes sense on a continuous, numeric X axis -
        # i.e. Line plots - so it's ignored for categorical chart types.
        use_axis_break = self.axis_break_cb.isChecked() and chart_type == "Line"

        self.plot_figure.clf()

        if use_axis_break:
            gs = self.plot_figure.add_gridspec(nrows=len(metrics), ncols=2, width_ratios=[1, 1], wspace=0.08)
            self.axes = []
            axis_pairs = []
            for row in range(len(metrics)):
                ax_left = self.plot_figure.add_subplot(gs[row, 0])
                ax_right = self.plot_figure.add_subplot(gs[row, 1], sharey=ax_left)
                self.axes.append(ax_left)
                axis_pairs.append((ax_left, ax_right))
        else:
            self.axes = self.plot_figure.subplots(nrows=len(metrics), ncols=1, squeeze=False)[:, 0].tolist()
        self.ax = self.axes[0]

        for idx, metric in enumerate(metrics):
            plot_df = df
            y_col = metric
            if normalize:
                if reference:
                    plot_df = NormalizationEngine.wild_type_fold_change(df.copy(), metric, x_col, wt_label=reference)
                    candidate_col = f"{metric} (Fold Change)"
                    if candidate_col in plot_df.columns:
                        y_col = candidate_col
                    else:
                        self.status_bar.showMessage(
                            f"Could not normalize '{metric}' to '{reference}' (no data / zero mean) — showing raw values.", 5000)
                else:
                    self.status_bar.showMessage("Select a reference group in 'Compare To' to normalize.", 5000)

            manual_yrange = self.manual_yrange_cb.isChecked()
            plot_kwargs = dict(
                dark_mode=self.is_dark_mode, palette=self.palette_selector.currentText(),
                stat_mode=stat_mode, hue_col=hue_col, log_scale=log_scale,
                reference=reference, pairing_col=pairing_col,
                show_pvalue=self.show_pvalue_cb.isChecked(),
                y_min=self.y_min_spin.value() if manual_yrange else None,
                y_max=self.y_max_spin.value() if manual_yrange else None,
                group_colors=self.group_colors, bw_mode=bw_mode, hist_bins=hist_bins,
                custom_title=self.custom_titles.get(y_col), custom_xlabel=self.custom_xlabel,
                custom_ylabel=self.custom_ylabels.get(y_col), group_order=self.group_order,
                show_ns=self.show_ns_cb.isChecked(),
            )

            if use_axis_break:
                ax_left, ax_right = axis_pairs[idx]
                PlottingEngine.generate_plot(plot_df, chart_type, x_col, y_col, ax_left, **plot_kwargs)
                PlottingEngine.generate_plot(plot_df, chart_type, x_col, y_col, ax_right, **plot_kwargs)
                self._apply_axis_break(ax_left, ax_right)
            else:
                PlottingEngine.generate_plot(plot_df, chart_type, x_col, y_col, self.axes[idx], **plot_kwargs)

        self.plot_figure.tight_layout()
        self.plot_canvas.draw()

    def update_summary_table_view(self):
        df = self.get_active_plot_df()
        x_var = self.x_selector.currentText()
        metrics = self.get_selected_metrics()
        y_var = metrics[0] if metrics else None
        if not x_var or not y_var or df is None: return
        self.current_summary_df = SummaryStatistics.generate_table(
            df, x_var, y_var, order=self._current_group_order())
        if self.current_summary_df.empty: return
        self.summary_table_view.setRowCount(self.current_summary_df.shape[0])
        self.summary_table_view.setColumnCount(self.current_summary_df.shape[1])
        self.summary_table_view.setHorizontalHeaderLabels(self.current_summary_df.columns)
        for r in range(self.current_summary_df.shape[0]):
            for c in range(self.current_summary_df.shape[1]):
                self.summary_table_view.setItem(r, c, QTableWidgetItem(str(self.current_summary_df.iat[r, c])))
        self.summary_table_view.resizeColumnsToContents()

    def _add_column_to_data(self, make):
        """Apply *make(df) -> df* (which adds a column) to the full dataset,
        so the new column survives changing filters, then re-filter."""
        self.master_df = make(self.master_df.copy())
        self.filtered_df = FilterEngine.apply_multi_filter(
            self.master_df, self.dock_filter_selections)
        self.metric_cols = [c for c in self.master_df.columns
                            if pd.api.types.is_numeric_dtype(self.master_df[c])]
        self.display_data(self.filtered_df)
        self.update_selectors()

    def run_bg_sub(self):
        if self.master_df is None: return
        metrics = self.get_selected_metrics()
        if not metrics: return
        bg = self.bg_spin.value()
        self._add_column_to_data(
            lambda df: NormalizationEngine.background_subtraction(df, metrics[0], bg))
        self.status_bar.showMessage(f"Added '{metrics[0]} (Bg Sub)' (minus {bg:g}).", 6000)

    def run_wt_fold(self):
        if self.filtered_df is None: return
        metrics = self.get_selected_metrics()
        reference = self.get_reference()
        if not metrics or not reference:
            self.status_bar.showMessage("Select a Metric and a reference group ('Compare To') first.", 5000)
            return
        x = self.x_selector.currentText()
        self._add_column_to_data(
            lambda df: NormalizationEngine.wild_type_fold_change(df, metrics[0], x, wt_label=reference))
        self.status_bar.showMessage(f"Added '{metrics[0]} (Fold Change)' relative to '{reference}'.", 6000)

    def _run_stats_from_toolbar(self):
        self.tabs.setCurrentWidget(self.analysis_tab)
        self.run_statistical_pipeline()

    def run_statistical_pipeline(self):
        df = self.get_active_plot_df()
        x_var = self.x_selector.currentText()
        metrics = self.get_selected_metrics()
        if not x_var or not metrics or df is None:
            self.stats_output.setPlainText("Open data and choose an X axis and a "
                                           "Metric on the Plotting Dashboard first.")
            return
        self.update_summary_table_view()

        test = self.stat_mode_selector.currentText()
        if test == StatisticsEngine.NONE:
            test = StatisticsEngine.AUTO
        hue = self.get_hue()
        order = self._current_group_order()
        texts, rows = [], []
        for y_var in metrics:
            if hue:
                results = StatisticsEngine.compare_within(
                    df, x_var, hue, y_var, test, self.get_reference(),
                    self.get_pairing_col())
                for g, r in results.items():
                    texts.append(StatisticsEngine.format_report(r, hue, y_var, within=f"{x_var} = {g}"))
                    rows += self._comparison_rows(r, y_var, f"{x_var} = {g}")
                if len(results) > 1:
                    texts.append("Note: p-values are also Holm-corrected across the "
                                 f"{len(results)} '{x_var}' groups.")
            else:
                r = StatisticsEngine.compare(df, x_var, y_var, test, self.get_reference(),
                                             self.get_pairing_col(), group_order=order)
                texts.append(StatisticsEngine.format_report(r, x_var, y_var))
                rows += self._comparison_rows(r, y_var)
        self.current_comparisons = pd.DataFrame(rows)
        self.stats_output.setPlainText(("\n\n" + "-" * 60 + "\n\n").join(texts))

    @staticmethod
    def _comparison_rows(r, y_var, within=None):
        out = []
        for c in r.comparisons:
            out.append({
                "Metric": y_var, "Within": within or "", "Group A": c.a,
                "Group B": c.b, "n A": c.n_a, "n B": c.n_b,
                "Difference (A-B)": c.diff,
                "95% CI low": c.ci[0] if c.ci else None,
                "95% CI high": c.ci[1] if c.ci else None,
                "p (adjusted)": c.p, "Significance": stars(c.p),
                "p (unadjusted)": c.p_raw, "Effect size": c.effect,
                "Test": r.test, "Post-hoc / correction": r.posthoc,
                "Omnibus": r.omnibus})
        return out

    def _save_path(self, title, default_name, ext, filt):
        start = os.path.join(self._last_dir(), default_name)
        path, _ = QFileDialog.getSaveFileName(self, title, start, filt)
        if not path:
            return None
        if not path.lower().endswith("." + ext):
            path += "." + ext
        self._remember_dir(path)
        return path

    def export_current_plot(self, format_type):
        labels = {"png": "PNG image", "svg": "SVG (editable text)", "pdf": "PDF",
                  "tiff": "TIFF image"}
        path = self._save_path("Export Figure", f"figure.{format_type}", format_type,
                               f"{labels[format_type]} (*.{format_type})")
        if path:
            ExportEngine.save_figure_to(self.plot_figure, path, format_type)
            self.status_bar.showMessage(f"Figure saved: {path}", 6000)

    def _ensure_stats(self):
        if not self.stats_output.toPlainText().strip() or getattr(
                self, "current_comparisons", None) is None:
            self.run_statistical_pipeline()
        return self.current_summary_df is not None and not self.current_summary_df.empty

    def export_stats_data(self):
        if not self._ensure_stats():
            QMessageBox.information(self, "Export Statistics", "Nothing to export yet — "
                                    "open data and choose an X axis and a Metric.")
            return
        path = self._save_path("Export Statistics", "statistics.xlsx", "xlsx", "Excel (*.xlsx)")
        if path:
            ExportEngine.save_excel_summary_to(
                path, self.current_summary_df, self.stats_output.toPlainText(),
                getattr(self, "current_comparisons", None))
            self.status_bar.showMessage(f"Statistics saved: {path}", 6000)

    def export_pdf_report(self):
        if not self._ensure_stats():
            QMessageBox.information(self, "PDF Report", "Nothing to report yet — "
                                    "open data and choose an X axis and a Metric.")
            return
        path = self._save_path("PDF Report", "report.pdf", "pdf", "PDF (*.pdf)")
        if not path:
            return
        import tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "plot.png")
        try:
            ExportEngine.save_figure_to(self.plot_figure, tmp, "png", dpi=300)
            PDFReportGenerator.generate_pdf(path, self.current_summary_df,
                                            self.stats_output.toPlainText(), tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.status_bar.showMessage(f"Report saved: {path}", 6000)

    def export_data_csv(self):
        df = self.get_active_plot_df()
        if df is None or df.empty:
            return
        path = self._save_path("Export Data", "data.csv", "csv", "CSV (*.csv)")
        if path:
            df.to_csv(path, index=False)
            self.status_bar.showMessage(f"{len(df)} rows saved: {path}", 6000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BioStatExplorer()
    window.show()
    sys.exit(app.exec())
