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
    QTableView, QToolBar, QToolButton, QDialog, QDialogButtonBox, QHeaderView,
    QFormLayout
)
from PySide6.QtGui import QAction, QKeySequence, QColor, QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.excel_loader import DataLoader, sheet_names, read_grid
from src.plotting import PlottingEngine, PALETTE_NAMES, palette_colors
from src.groups_panel import GroupsPanel
from src.curves_tab import CurvesTab
from src.layout_tab import LayoutTab
from src.filters import FilterEngine
from src.normalization import NormalizationEngine, plate_normalise
from src.statistics_engine import StatisticsEngine, format_p, stars
from src.utils import SummaryStatistics
from src.export_engine import ExportEngine
from src.report_generator import PDFReportGenerator
from src import plate_reader as pr
from src import theme
from src.table_model import PandasModel
from src import project

NO_SELECTION = "(none)"
DATA_FILTER = "Data files (*.xlsx *.xlsm *.xls *.csv *.tsv *.txt)"
DATA_EXT = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt")


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
        # every column visible without scrolling sideways
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(28)
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
        # Order of the Color-By series (legend / bars within a group)
        self.hue_order = []

        # Populated on load: which columns are recognized signal/value columns
        # vs. which are metadata/categorical columns usable for filtering.
        self.recognized_cols = []
        self.metric_cols = []
        self.metadata_cols = []
        self.metadata_checklists = {}   # {column_name: QListWidget}

        self.init_central_widget()
        self.init_dock_panels()
        self._undo_stack, self._redo_stack = [], []
        self._project_path = None
        self._dirty = False
        self.init_menu()
        self._building = False
        self._update_title()
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
        self._action(file_menu, "Open Project…", self.open_project, "Ctrl+Shift+O")
        self._action(file_menu, "Save Project", self.save_project,
                     QKeySequence.StandardKey.Save)
        self._action(file_menu, "Save Project As…", self.save_project_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._action(file_menu, "Open Data…", self.open_file_dialog,
                     QKeySequence.StandardKey.Open)
        self._action(file_menu, "Add Data (append)…",
                     lambda: self.open_file_dialog(append=True))
        self._action(file_menu, "Import Plate-Reader File…",
                     self.import_plate_file, "Ctrl+Shift+I")
        file_menu.addSeparator()
        self.export_menu = file_menu.addMenu("Export")
        self._action(self.export_menu, "Figure (PNG)…",
                     lambda: self.export_current_plot("png"), "Ctrl+Shift+E")
        self._action(self.export_menu, "Figure (SVG — editable in Inkscape)…",
                     lambda: self.export_current_plot("svg"))
        self._action(self.export_menu, "Figure (PDF)…",
                     lambda: self.export_current_plot("pdf"))
        self._action(self.export_menu, "Figure (TIFF)…",
                     lambda: self.export_current_plot("tiff"))
        res = self.export_menu.addMenu("Figure Resolution (PNG / TIFF)")
        from PySide6.QtGui import QActionGroup
        grp = QActionGroup(res)
        for d, note in ((300, ""), (600, "  (journals)"), (1200, "  (line art)")):
            a = res.addAction(f"{d} dpi{note}")
            a.setCheckable(True)
            a.setChecked(d == self._export_dpi())
            a.triggered.connect(lambda _c=False, d=d: QSettings(
                "BioStatExplorer", "BioStatExplorer").setValue("export_dpi", d))
            grp.addAction(a)
        res.addSeparator()
        res.addAction("SVG and PDF are vector: sharp at any size").setEnabled(False)
        self.export_menu.addSeparator()
        self._action(self.export_menu, "Statistics (Excel)…",
                     self.export_stats_data, "Ctrl+E")
        self._action(self.export_menu, "PDF Report…", self.export_pdf_report)
        self._action(self.export_menu, "Data Table (CSV)…", self.export_data_csv)
        file_menu.addSeparator()
        self._action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        edit_menu = menu_bar.addMenu("&Edit")
        self.undo_action = self._action(edit_menu, "Undo", self.undo,
                                        QKeySequence.StandardKey.Undo)
        self.redo_action = self._action(edit_menu, "Redo", self.redo, "Ctrl+Shift+Z")
        self._sync_undo_actions()

        view_menu = menu_bar.addMenu("&View")
        self.dark_action = QAction("Dark Mode", self, checkable=True)
        self.dark_action.toggled.connect(self.toggle_dark_theme)
        view_menu.addAction(self.dark_action)
        filt = self.dock.toggleViewAction()
        filt.setText("Filters Panel (filters every tab)")
        view_menu.addAction(filt)

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
        self._building = True          # no redraws until everything exists
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

        # Controls: three compact tabs; only what applies is shown
        def form_page():
            page = QWidget()
            form = QFormLayout(page)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setVerticalSpacing(8)
            form.setContentsMargins(10, 12, 10, 10)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            return scroll, form

        def section(form, text):
            lab = QLabel(text)
            lab.setObjectName("Section")
            form.addRow(lab)

        def combo(slot):
            c = QComboBox()
            c.currentTextChanged.connect(slot)
            return c

        def check(text, slot=None):
            c = QCheckBox(text)
            c.toggled.connect(slot or self.update_plot)
            return c

        def spin(lo, hi, val, decimals=3):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(decimals)
            sb.setValue(val)
            sb.valueChanged.connect(self.update_plot)
            return sb

        self.control_tabs = QTabWidget()
        self.control_tabs.setObjectName("ControlTabs")
        self.control_tabs.setMinimumWidth(320)
        self.control_tabs.setMaximumWidth(380)

        # --- Data ---
        data_scroll, f = form_page()
        self.chart_selector = combo(self.update_plot)
        self.chart_selector.addItems(["Bar", "Box + Swarm", "Boxplot", "Violin", "Beeswarm",
                                      "Scatter", "Line", "Stacked Bar", "Stacked Histogram",
                                      "Heatmap", "Volcano Plot", "Dendrogram", "Network Graph"])
        f.addRow("Graph", self.chart_selector)
        self.x_selector = combo(self.on_x_axis_changed)
        f.addRow("X axis", self.x_selector)
        self.y_selector = combo(self.update_plot)
        f.addRow("Metric", self.y_selector)
        self.hue_selector = combo(self.update_plot)
        self.hue_selector.setToolTip("Splits each X group into coloured bars/boxes")
        f.addRow("Color by", self.hue_selector)
        self.show_all_metrics_cb = check("All metrics (one panel each)")
        f.addRow("", self.show_all_metrics_cb)
        self.chart_help_label = QLabel()
        self.chart_help_label.setWordWrap(True)
        self.chart_help_label.setObjectName("Hint")
        f.addRow(self.chart_help_label)
        section(f, "Show only")
        self.data_selection_layout = QVBoxLayout()
        self.data_selection_layout.setSpacing(6)
        f.addRow(self.data_selection_layout)
        self.control_tabs.addTab(data_scroll, "Data")

        # --- Style ---
        style_scroll, f = form_page()
        self.palette_selector = combo(self.update_plot)
        self.palette_selector.addItems(PALETTE_NAMES)
        f.addRow("Colours", self.palette_selector)
        self.bw_mode_cb = check("Black && white (grey + hatching)")
        self.bw_mode_cb.setToolTip("Grey shades with hatching / line styles, for print")
        f.addRow("", self.bw_mode_cb)
        self.dark_mode_cb = check("Dark mode", self.toggle_dark_theme)
        f.addRow("", self.dark_mode_cb)
        section(f, "Y axis")
        self.log_scale_cb = check("Log scale")
        f.addRow("", self.log_scale_cb)
        self.normalize_ref_cb = check("Fold change vs reference group")
        self.normalize_ref_cb.setToolTip("Divides by the mean of the 'Compare to' group (Stats tab)")
        f.addRow("", self.normalize_ref_cb)
        self.manual_yrange_cb = check("Fixed range")
        self.manual_yrange_cb.toggled.connect(self._sync_control_visibility)
        f.addRow("", self.manual_yrange_cb)
        yr = QHBoxLayout()
        self.y_min_spin = spin(-1e9, 1e9, 0.0)
        self.y_max_spin = spin(-1e9, 1e9, 100.0)
        yr.addWidget(self.y_min_spin)
        yr.addWidget(QLabel("to"))
        yr.addWidget(self.y_max_spin)
        self._yrange_row = QWidget()
        self._yrange_row.setLayout(yr)
        yr.setContentsMargins(0, 0, 0, 0)
        f.addRow("", self._yrange_row)
        self.hist_bins_spin = QSpinBox()
        self.hist_bins_spin.setRange(2, 200)
        self.hist_bins_spin.setValue(20)
        self.hist_bins_spin.valueChanged.connect(self.update_plot)
        self._bins_label = QLabel("Bins")
        f.addRow(self._bins_label, self.hist_bins_spin)
        section(f, "Labels")
        lab_hint = QLabel("Click the title or an axis label on the graph to edit it.")
        lab_hint.setObjectName("Hint")
        lab_hint.setWordWrap(True)
        f.addRow(lab_hint)
        reset_labels_btn = QPushButton("Reset title and axis labels")
        reset_labels_btn.clicked.connect(self.reset_custom_labels)
        f.addRow(reset_labels_btn)
        # axis break (Line plots)
        self._break_section = QLabel("X-axis break (Line)")
        self._break_section.setObjectName("Section")
        f.addRow(self._break_section)
        self.axis_break_cb = check("Split the X axis into two ranges")
        f.addRow("", self.axis_break_cb)
        self.axis_break_left_min = spin(-1e9, 1e9, 1.0, 2)
        self.axis_break_left_max = spin(-1e9, 1e9, 2.0, 2)
        self.axis_break_right_min = spin(-1e9, 1e9, 3.0, 2)
        self.axis_break_right_max = spin(-1e9, 1e9, 4.0, 2)
        self._break_rows = []
        for label, a, b in (("Left", self.axis_break_left_min, self.axis_break_left_max),
                            ("Right", self.axis_break_right_min, self.axis_break_right_max)):
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(a)
            hl.addWidget(QLabel("to"))
            hl.addWidget(b)
            lab = QLabel(label)
            f.addRow(lab, row)
            self._break_rows += [lab, row]
        self.control_tabs.addTab(style_scroll, "Style")

        # --- Stats ---
        stats_scroll, f = form_page()
        self.stat_mode_selector = combo(self.update_plot)
        self.stat_mode_selector.addItems(StatisticsEngine.ALL_TESTS)
        f.addRow("Test", self.stat_mode_selector)
        self.reference_selector = combo(self.update_plot)
        self.reference_selector.setToolTip("Compare every group with this one only "
                                           "(Dunnett / Dunn vs control)")
        f.addRow("Compare to", self.reference_selector)
        self.pairing_selector = combo(self.update_plot)
        self.pairing_selector.setToolTip("The column that matches samples across groups "
                                         "(e.g. replicate ID)")
        self._pairing_label = QLabel("Pair by")
        f.addRow(self._pairing_label, self.pairing_selector)
        self.show_pvalue_cb = check("Show p-values")
        f.addRow("", self.show_pvalue_cb)
        self.show_ns_cb = check("Show non-significant (ns)")
        f.addRow("", self.show_ns_cb)
        self.exclude_outliers_cb = check("Leave out outliers (Grubbs)")
        self.exclude_outliers_cb.setToolTip(
            "Grubbs' test (alpha 0.05) within each group; outliers are listed in "
            "the report either way")
        f.addRow("", self.exclude_outliers_cb)
        overlay_hint = QLabel("Stars use the same tests and corrections as the Statistical "
                              "Analysis tab. With Color by, colours are compared within "
                              "each X group.")
        overlay_hint.setWordWrap(True)
        overlay_hint.setObjectName("Hint")
        f.addRow(overlay_hint)
        run_btn = QPushButton("Open Full Report")
        run_btn.setObjectName("primaryBtn")
        run_btn.clicked.connect(self._run_stats_from_toolbar)
        f.addRow(run_btn)
        self.control_tabs.addTab(stats_scroll, "Stats")

        for w in (self.chart_selector, self.stat_mode_selector):
            w.currentTextChanged.connect(self._sync_control_visibility)

        # Graph + Groups panel
        self.plot_figure = Figure(figsize=(6, 5), dpi=100)
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_canvas.mpl_connect("pick_event", self.on_plot_pick)
        self.ax = self.plot_figure.add_subplot(111)
        self.axes = [self.ax]
        PlottingEngine.generate_plot(None, "", "", "", self.ax)
        canvas_card = QWidget()
        canvas_card.setObjectName("Card")
        cl = QVBoxLayout(canvas_card)
        cl.setContentsMargins(8, 8, 8, 8)
        head = QHBoxLayout()
        head.addStretch()
        add_layout_btn = QPushButton("＋ Add to Layout")
        add_layout_btn.setToolTip("Put this graph on the figure page (Layout tab)")
        add_layout_btn.clicked.connect(self._add_to_layout)
        head.addWidget(add_layout_btn)
        cl.addLayout(head)
        cl.addWidget(self.plot_canvas)

        self.groups_panel = GroupsPanel()
        self.groups_panel.order_changed.connect(self._set_group_order)
        self.groups_panel.series_order_changed.connect(self._set_series_order)
        self.groups_panel.color_changed.connect(self._set_group_color)
        self.groups_panel.colors_reset.connect(self.reset_group_colors)
        self.groups_panel.order_reset.connect(self.reset_group_order)

        self.graph_layout.setContentsMargins(10, 10, 10, 10)
        self.graph_layout.setSpacing(10)
        self.graph_layout.addWidget(self.control_tabs)
        self.graph_layout.addWidget(canvas_card, stretch=1)
        self.graph_layout.addWidget(self.groups_panel)
        self._sync_control_visibility()

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
        self.tabs.addTab(self.plate_tab, "Plate")
        self.tabs.addTab(self.graph_tab, "Plotting Dashboard")
        self.tabs.addTab(self.analysis_tab, "Statistical Analysis")
        self.curves_tab = CurvesTab(lambda: (self.get_active_plot_df(),
                                             list(getattr(self, "metric_cols", []))))
        self.curves_tab.send_parameters.connect(self._use_curve_parameters)
        self.curves_tab.export_requested.connect(self._export_curve_parameters)
        self.tabs.addTab(self.curves_tab, "Curves")
        self.layout_tab = LayoutTab(lambda ax, df, spec: BioStatExplorer.draw_metric(
            ax, df, spec, spec["metrics"][0]))
        self.layout_tab.export_requested.connect(self._export_layout)
        self.layout_tab.changed.connect(self._mark_dirty)
        self.tabs.addTab(self.layout_tab, "Layout")
        self.setCentralWidget(self.tabs)

    def init_plate_reader_tab(self):
        self.plate_tab = QWidget()
        outer_layout = QVBoxLayout(self.plate_tab)

        intro = QLabel("<b>Two ways to fill a plate:</b> type or paste the values by hand "
                       "in the grids below (click a well and type, or paste a block from "
                       "Excel), or <b>Import Plate-Reader File…</b> to fill them from an "
                       "instrument export. You can mix both, and edit imported values.")
        intro.setWordWrap(True)
        outer_layout.addWidget(intro)
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

        # 3. series + corrections
        extra = QHBoxLayout()
        series_box = QGroupBox("Plates")
        sl = QHBoxLayout(series_box)
        self.series_label = QLabel("One plate (the grids above).")
        self.series_label.setWordWrap(True)
        sl.addWidget(self.series_label, 1)
        self.series_combo = QComboBox()
        self.series_combo.setToolTip("Which plate of the series the grids show")
        self.series_combo.currentIndexChanged.connect(self._show_series_plate)
        sl.addWidget(self.series_combo)
        times_btn = QPushButton("Names && Times…")
        times_btn.clicked.connect(self._edit_series_times)
        self._series_widgets = [self.series_combo, times_btn]
        sl.addWidget(times_btn)
        extra.addWidget(series_box, 1)

        norm_box = QGroupBox("3. Corrections (optional)")
        nl = QGridLayout(norm_box)
        self.blank_cb = QCheckBox("Subtract blanks: wells labelled")
        self.blank_label_edit = QLineEdit("blank")
        self.blank_label_edit.setMaximumWidth(120)
        self.blank_label_edit.setToolTip("Label of the blank wells in the plate map "
                                         "(several: separate with commas)")
        self.blank_layer_combo = QComboBox()
        self.blank_match_combo = QComboBox()
        self.blank_match_combo.setToolTip("e.g. Medium: each well gets the blank of its own "
                                          "medium")
        nl.addWidget(self.blank_cb, 0, 0)
        nl.addWidget(self.blank_label_edit, 0, 1)
        nl.addWidget(QLabel("in"), 0, 2)
        nl.addWidget(self.blank_layer_combo, 0, 3)
        nl.addWidget(QLabel("matched by"), 0, 4)
        nl.addWidget(self.blank_match_combo, 0, 5)
        self.od_cb = QCheckBox("Divide every readout by")
        self.od_combo = QComboBox()
        nl.addWidget(self.od_cb, 1, 0)
        nl.addWidget(self.od_combo, 1, 1, 1, 3)
        self.keep_blanks_cb = QCheckBox("Keep blank wells in the data")
        nl.addWidget(self.keep_blanks_cb, 1, 4, 1, 2)
        extra.addWidget(norm_box, 2)
        outer_layout.addLayout(extra)

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
        self.plate_series = []
        self._series_index = 0
        self._refresh_series_label()

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
        self._refresh_norm_choices()
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
        self._refresh_norm_choices()

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
        self._refresh_norm_choices()

    @staticmethod
    def _grid_is_empty(editor):
        return all(not v.strip() for row in editor.get_grid_values() for v in row)

    def import_plate_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Plate-Reader File(s) — select several for a time series",
            self._last_dir(), DATA_FILTER)
        if not paths:
            return
        self._remember_dir(paths[0])
        paths = sorted(paths, key=lambda p_: (pr.hours_in_name(os.path.basename(p_)) is None,
                                              pr.hours_in_name(os.path.basename(p_)) or 0,
                                              os.path.basename(p_).lower()))
        sheet, ok = self._choose_sheet(paths[0], "plate grids")
        if not ok:
            return
        try:
            blocks = pr.find_plate_blocks(read_grid(paths[0], sheet))
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
        dlg = PlateImportDialog(blocks, paths[0], self)
        if len(paths) > 1:
            dlg.setWindowTitle(f"Import {len(paths)} Plates — choose the grids "
                               "(the same ones are taken from every file)")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dlg.chosen()
        if not chosen:
            return
        self._push_undo()
        shape = pr.shape_name(max(b["n_rows"] for b, _, _ in chosen),
                              max(b["n_cols"] for b, _, _ in chosen))
        if shape != self.plate_size_selector.currentText():
            self.plate_size_selector.setCurrentText(shape)
        # plate maps go straight into the map grids
        for block, name, is_map in chosen:
            if is_map:
                for old in [n for n, e in self.map_layer_editors.items()
                            if n == "Strain" and n != name and self._grid_is_empty(e)]:
                    self.map_layers_tabs.removeTab(
                        self.map_layers_tabs.indexOf(self.map_layer_editors.pop(old)))
                ed = self.map_layer_editors.get(name) or self._new_plate_grid_tab(
                    self.map_layers_tabs, self.map_layer_editors, name)
                ed.set_cells(block["cells"])
        readouts = [(b, n) for b, n, is_map in chosen if not is_map]
        series, problems, matched = [], [], []
        for i, path in enumerate(paths):
            if i == 0:
                file_blocks = [b for b, _ in readouts]
            else:
                try:
                    file_blocks = pr.match_blocks([b for b, _ in readouts],
                                                  pr.find_plate_blocks(read_grid(path, sheet)
                                                                       if sheet in sheet_names(path)
                                                                       else read_grid(path)))
                except Exception as e:
                    problems.append(f"{os.path.basename(path)}: {e}")
                    continue
            vals = {}
            for (ref, name), b in zip(readouts, file_blocks):
                if b is None:
                    problems.append(f"{os.path.basename(path)}: no grid like '{ref['title']}'")
                    continue
                if b["title"] != ref["title"]:
                    matched.append(f"{os.path.basename(path)}: '{b['title']}' used as "
                                   f"{name} (first file: '{ref['title']}')")
                vals[name] = b["cells"]
            stem = os.path.splitext(os.path.basename(path))[0]
            series.append({"name": stem, "time": pr.hours_in_name(stem), "values": vals,
                           "saturated": sum(b["saturated"] for b in file_blocks if b)})
        # more readouts for plates already loaded (e.g. the OD files after the
        # GFP/mKate files of the same time points)?
        new_names = {name for _, name in readouts}
        old_times = {p_.get("time") for p_ in self.plate_series}
        if len(series) > 1 and self.plate_series and (
                {p_["time"] for p_ in series} <= old_times or len(series) == len(self.plate_series)):
            have = new_names & set(self.plate_series[0]["values"])
            ans = QMessageBox.question(
                self, "Add to the plates already loaded?",
                f"{len(self.plate_series)} plates are already loaded. Add "
                f"{', '.join(sorted(new_names))} to them (matched by time point, or "
                "by order), instead of starting a new series?"
                + (f"\n\n{', '.join(sorted(have))} will be replaced by the new files' "
                   "values." if have else ""),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if ans == QMessageBox.StandardButton.Yes:
                self._store_series_plate()
                unmatched = []
                for i, new in enumerate(series):
                    old = next((o_ for o_ in self.plate_series if new["time"] is not None
                                and o_.get("time") == new["time"]), None)
                    if old is None and new["time"] is None and i < len(self.plate_series):
                        old = self.plate_series[i]
                    if old is None:
                        unmatched.append(new["name"])
                        continue
                    old["values"].update(new["values"])
                for name in sorted(new_names):
                    if name not in self.value_layer_editors:
                        self._new_plate_grid_tab(self.value_layers_tabs,
                                                 self.value_layer_editors, name)
                self._load_series_plate(self._series_index,
                                        self.plate_series[self._series_index]["values"])
                self._refresh_norm_choices()
                self.tabs.setCurrentWidget(self.plate_tab)
                self.status_bar.showMessage(
                    f"Added {', '.join(sorted(new_names))} to {len(series) - len(unmatched)} "
                    "plates", 10000)
                if unmatched:
                    QMessageBox.warning(self, "Not matched",
                                        "No loaded plate with the same time point for:\n"
                                        + "\n".join(unmatched))
                return
        for name in [n for n, e in self.value_layer_editors.items() if n == "Value"
                     and self._grid_is_empty(e) and n not in new_names]:
            self.value_layers_tabs.removeTab(
                self.value_layers_tabs.indexOf(self.value_layer_editors.pop(name)))
        for _, name in readouts:
            if name not in self.value_layer_editors:
                self._new_plate_grid_tab(self.value_layers_tabs, self.value_layer_editors, name)
        self.plate_series = series if len(series) > 1 else []
        self._series_index = 0
        self._load_series_plate(0, series[0]["values"] if series else {})
        self._refresh_series_label()
        self._refresh_norm_choices()
        self.tabs.setCurrentWidget(self.plate_tab)
        warn = [f"{p_['name']}: {p_['saturated']} saturated wells" for p_ in series
                if p_["saturated"]]
        self.status_bar.showMessage(
            f"Imported {len(chosen)} grid(s) from {len(paths)} file(s)"
            + (" — " + "; ".join(warn) if warn else ""), 12000)
        if problems:
            QMessageBox.warning(self, "Some grids were not found", "\n".join(problems[:20]))
        if matched:
            QMessageBox.information(
                self, "Check the matched grids",
                "These files name their grids differently, so they were matched by "
                "position — please check they measure the same thing:\n\n"
                + "\n".join(matched[:20]))

    # --- plate series (several files, e.g. 8 h / 16 h / 24 h) ---
    def _load_series_plate(self, index, values):
        for name, cells in values.items():
            ed = self.value_layer_editors.get(name)
            if ed is not None:
                ed.clear_grid()
                ed.set_cells(cells)

    def _store_series_plate(self):
        """Keep edits made in the grids for the plate shown."""
        if self.plate_series and 0 <= self._series_index < len(self.plate_series):
            self.plate_series[self._series_index]["values"] = {
                n: e.get_grid_values() for n, e in self.value_layer_editors.items()}

    def _show_series_plate(self, index):
        if not self.plate_series or index < 0 or index == self._series_index:
            return
        self._store_series_plate()
        self._series_index = index
        self._load_series_plate(index, self.plate_series[index]["values"])

    @staticmethod
    def _series_title(p_):
        return p_["name"] + (f"  ({p_['time']:g} h)" if p_.get("time") is not None else "")

    def _refresh_series_label(self):
        n = len(self.plate_series)
        for w in self._series_widgets:
            w.setVisible(n > 1)
        if n > 1:
            times = [p_["time"] for p_ in self.plate_series if p_.get("time") is not None]
            self.series_label.setText(
                f"<b>{n} plates</b> — one plate map for all; each plate becomes a "
                "'Plate' value" + (", with a 'Time (h)' column" if times else "")
                + ". The grids show:")
            self.series_combo.blockSignals(True)
            self.series_combo.clear()
            self.series_combo.addItems([self._series_title(p_) for p_ in self.plate_series])
            self.series_combo.setCurrentIndex(self._series_index)
            self.series_combo.blockSignals(False)
        else:
            self.series_label.setText("One plate (the grids above). To analyse several "
                                      "plates (e.g. 8 h, 16 h, 24 h), select all their files "
                                      "in Import Plate-Reader File….")

    def _edit_series_times(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Plate Names and Times")
        lay = QVBoxLayout(dlg)
        t = QTableWidget(len(self.plate_series), 2)
        t.setHorizontalHeaderLabels(["Plate", "Time (h) — empty if none"])
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i, p_ in enumerate(self.plate_series):
            t.setItem(i, 0, QTableWidgetItem(p_["name"]))
            t.setItem(i, 1, QTableWidgetItem("" if p_.get("time") is None else f"{p_['time']:g}"))
        lay.addWidget(t)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.resize(520, 320)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._push_undo()
        for i, p_ in enumerate(self.plate_series):
            p_["name"] = t.item(i, 0).text().strip() or p_["name"]
            txt = t.item(i, 1).text().strip().replace(",", ".")
            try:
                p_["time"] = float(txt) if txt else None
            except ValueError:
                pass
        self._refresh_series_label()

    # --- corrections ---
    def _refresh_norm_choices(self):
        if not hasattr(self, "blank_layer_combo"):
            return
        maps = list(self.map_layer_editors)
        for combo, items in ((self.blank_layer_combo, maps),
                             (self.blank_match_combo, ["(whole plate)"] + maps),
                             (self.od_combo, list(self.value_layer_editors))):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            od_like = lambda n_: "od" in n_.lower() or "600" in n_
            if cur in items and not (combo is self.od_combo and not od_like(cur)):
                combo.setCurrentText(cur)
            elif combo is self.od_combo:
                od = next((n for n in items if "od" in n.lower() or "600" in n), None)
                if od:
                    combo.setCurrentText(od)
            combo.blockSignals(False)

    def _plate_norm_state(self):
        return {"blank": self.blank_cb.isChecked(), "label": self.blank_label_edit.text(),
                "layer": self.blank_layer_combo.currentText(),
                "match": self.blank_match_combo.currentText(),
                "od": self.od_cb.isChecked(), "od_col": self.od_combo.currentText(),
                "keep_blanks": self.keep_blanks_cb.isChecked()}

    def _restore_plate_norm(self, st):
        self._refresh_norm_choices()
        self.blank_cb.setChecked(bool(st.get("blank")))
        self.blank_label_edit.setText(st.get("label", "blank"))
        for combo, key in ((self.blank_layer_combo, "layer"), (self.blank_match_combo, "match"),
                           (self.od_combo, "od_col")):
            if st.get(key) and combo.findText(st[key]) >= 0:
                combo.setCurrentText(st[key])
        self.od_cb.setChecked(bool(st.get("od")))
        self.keep_blanks_cb.setChecked(bool(st.get("keep_blanks")))

    def build_dataset_from_plate(self):
        n_rows, n_cols = pr.PLATE_SHAPES[self.plate_size_selector.currentText()]

        if not self.value_layer_editors:
            QMessageBox.warning(self, "No value grids", "Add at least one raw value grid first (e.g. 'GFP').")
            return
        metric_names = list(self.value_layer_editors.keys())
        self._store_series_plate()
        plates = self.plate_series or [{"name": None, "time": None, "values": {
            n: e.get_grid_values() for n, e in self.value_layer_editors.items()}}]

        layer_dfs = []
        for name, editor in self.map_layer_editors.items():
            grid = editor.get_grid_values()
            layer_dfs.append(pr.map_grid_to_long(grid, name, n_rows, n_cols))

        frames = []
        for plate in plates:
            value_dfs = [pr.grid_to_long(plate["values"].get(name, [[""] * n_cols] * n_rows),
                                         value_name=name, n_rows=n_rows, n_cols=n_cols)
                         for name in metric_names]
            merged_values = value_dfs[0]
            for vdf in value_dfs[1:]:
                merged_values = merged_values.merge(vdf, on=["Well", "Row", "Column"], how="outer")
            merged = pr.merge_well_layers(merged_values, layer_dfs)
            if plate["name"] is not None:
                merged.insert(0, "Plate", plate["name"])
                if any(p_.get("time") is not None for p_ in plates):
                    merged.insert(1, "Time (h)", plate.get("time"))
            frames.append(merged)
        merged = pd.concat(frames, ignore_index=True)
        if merged[metric_names].dropna(how="all").empty:
            QMessageBox.warning(self, "No values found",
                "All raw plate value grids are empty. Type values in, or use 'Load From File...' first.")
            return

        notes = []
        blank_col = self.blank_layer_combo.currentText() if self.blank_cb.isChecked() else None
        od = self.od_combo.currentText() if self.od_cb.isChecked() else None
        if blank_col or od:
            match = self.blank_match_combo.currentText()
            merged, notes = plate_normalise(
                merged, metric_names, blank_col, self.blank_label_edit.text(),
                match_col=None if match == "(whole plate)" else match,
                plate_col="Plate" if "Plate" in merged else None, od=od,
                keep_blanks=self.keep_blanks_cb.isChecked())
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
        if notes:
            self.status_bar.showMessage(" ".join(notes[:3]), 15000)
        self.last_plate_notes = notes

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
        # the Plotting Dashboard's "Show only" lists cover everyday filtering;
        # this panel stays available under View
        self.dock.hide()
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

    @staticmethod
    def _export_dpi():
        try:
            d = int(QSettings("BioStatExplorer", "BioStatExplorer").value("export_dpi", 600))
        except (TypeError, ValueError):
            d = 600
        return d if d in (300, 600, 1200) else 600

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
        if self.master_df is not None and not getattr(self, "_restoring", False):
            self._push_undo()
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
            self.hue_order = []
            self.group_colors = {}
        self.display_data(self.filtered_df)
        self.metadata_col_list.clear()
        self.metadata_col_list.addItems(self.metadata_cols)
        self.rebuild_data_selection_panel()      # before plotting with it
        self.update_selectors(choose_defaults=not append)
        if hasattr(self, "curves_tab"):
            self.curves_tab.refresh_columns()
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
        # readings first; time / position / raw columns last
        late = lambda c: (c.lower().startswith(("time", "row", "column", "replicate", "rep"))
                          or c.endswith("(raw)"))
        metric_candidates = [c for c in metric_candidates if not late(c)] + \
            [c for c in metric_candidates if late(c)]
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

    CHART_HELP = {
        "Heatmap": "Rows = X groups; columns = Color by (or every metric if empty).",
        "Volcano Plot": "X = log2 fold change column, Metric = p-value column; "
                        "Color by labels the points.",
        "Dendrogram": "Clusters the X groups using every numeric column.",
        "Network Graph": "Correlations between all numeric columns.",
        "Stacked Histogram": "Distribution of the Metric, stacked by Color by "
                             "(or by X).",
        "Stacked Bar": "Sum of the Metric per X group, stacked by Color by.",
        "Line": "Mean ± SEM against a numeric X (time, concentration…).",
    }

    def _sync_control_visibility(self, *_):
        """Show only the controls that apply to the chosen graph / test."""
        chart = self.chart_selector.currentText()
        help_text = self.CHART_HELP.get(chart, "")
        self.chart_help_label.setText(help_text)
        self.chart_help_label.setVisible(bool(help_text))
        for w in [self._bins_label, self.hist_bins_spin]:
            w.setVisible(chart == "Stacked Histogram")
        for w in [self._break_section, self.axis_break_cb] + self._break_rows:
            w.setVisible(chart == "Line")
        paired = self.stat_mode_selector.currentText() in StatisticsEngine.PAIRED
        self._pairing_label.setVisible(paired)
        self.pairing_selector.setVisible(paired)
        self._yrange_row.setEnabled(self.manual_yrange_cb.isChecked())

    def on_x_axis_changed(self):
        if getattr(self, "_building", False):
            return
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
            rows = min(lst.count(), 5)
            lst.setFixedHeight(rows * 27 + 8)
            v.setContentsMargins(6, 4, 6, 6)
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
        self._push_undo()
        self.group_colors = {}
        self.update_plot()

    def reset_group_order(self):
        self._push_undo()
        self.group_order = []
        self.hue_order = []
        self.update_plot()

    def _set_group_order(self, names):
        self._push_undo()
        self.group_order = list(names)
        self.update_plot()

    def _set_series_order(self, names):
        self._push_undo()
        self.hue_order = list(names)
        self.update_plot()

    def _set_group_color(self, name, color):
        self._push_undo()
        if color:
            self.group_colors[name] = color
        else:
            self.group_colors.pop(name, None)
        self.update_plot()

    def _refresh_groups_panel(self, df, x_col, metric, hue_col, chart):
        """Groups panel = what the graph shows, with the colours drawn."""
        if df is None or df.empty or not x_col or x_col not in df.columns:
            self.groups_panel.set_groups([])
            return
        x_groups = self._current_group_order()
        series = None
        if hue_col and hue_col in df.columns and hue_col != x_col:
            levels = [str(v) for v in df[hue_col].dropna().unique()]
            first = [h for h in self.hue_order if h in levels]
            series = first + [h for h in levels if h not in first]
            cats = series
        elif chart == "Line":
            cats = [metric]
        else:
            cats = x_groups
        bw = self.bw_mode_cb.isChecked()
        if bw:
            n = max(len(cats), 1)
            shades = np.linspace(0.75, 0.05, n) if n > 1 else [0.2]
            colors = ["#%02x%02x%02x" % ((int(v * 255),) * 3) for v in shades]
        else:
            colors = palette_colors(self.palette_selector.currentText(), len(cats))
        shown = {str(c): (colors[i] if bw else self.group_colors.get(str(c), colors[i]))
                 for i, c in enumerate(cats)}
        if chart == "Line" and not series:
            series = [metric]
        self.groups_panel.set_groups(x_groups, series, shown, self.group_colors)

    def reset_custom_labels(self):
        self._push_undo()
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
                    self._push_undo()
                    self.group_colors[group_name] = chosen.name()
                    self.update_plot()

        elif url.startswith("edit_title:"):
            metric = url[len("edit_title:"):]
            current = self.custom_titles.get(metric, artist.get_text())
            text, ok = QInputDialog.getText(self, "Edit Graph Title", "Title:", text=current)
            if ok:
                self._push_undo()
                if text.strip():
                    self.custom_titles[metric] = text
                else:
                    self.custom_titles.pop(metric, None)  # blank -> revert to auto title
                self.update_plot()

        elif url == "edit_xlabel":
            current = self.custom_xlabel if self.custom_xlabel is not None else artist.get_text()
            text, ok = QInputDialog.getText(self, "Edit X-Axis Label", "X-axis label:", text=current)
            if ok:
                self._push_undo()
                self.custom_xlabel = text if text.strip() else None
                self.update_plot()

        elif url.startswith("edit_ylabel:"):
            metric = url[len("edit_ylabel:"):]
            current = self.custom_ylabels.get(metric, artist.get_text())
            text, ok = QInputDialog.getText(self, "Edit Y-Axis Label", "Y-axis label:", text=current)
            if ok:
                self._push_undo()
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

        self._push_undo()
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

    # ---------------------------------------------------------------
    # The plot as a "spec": every setting in one dict. The live plot,
    # layout panels, saved projects and undo all use it.
    # ---------------------------------------------------------------
    def plot_spec(self):
        manual = self.manual_yrange_cb.isChecked()
        brk = self.axis_break_cb.isChecked()
        return {
            "chart": self.chart_selector.currentText(),
            "x": self.x_selector.currentText(),
            "metrics": self.get_selected_metrics(),
            "y": self.y_selector.currentText(),
            "all_metrics": self.show_all_metrics_cb.isChecked(),
            "hue": self.get_hue(),
            "reference": self.get_reference(),
            "pairing": self.get_pairing_col(),
            "normalize": self.normalize_ref_cb.isChecked(),
            "log": self.log_scale_cb.isChecked(),
            "stat": self.stat_mode_selector.currentText(),
            "bw": self.bw_mode_cb.isChecked(),
            "bins": self.hist_bins_spin.value(),
            "palette": self.palette_selector.currentText(),
            "show_p": self.show_pvalue_cb.isChecked(),
            "show_ns": self.show_ns_cb.isChecked(),
            "y_range": [self.y_min_spin.value(), self.y_max_spin.value()] if manual else None,
            "y_range_values": [self.y_min_spin.value(), self.y_max_spin.value()],
            "axis_break": [self.axis_break_left_min.value(), self.axis_break_left_max.value(),
                           self.axis_break_right_min.value(),
                           self.axis_break_right_max.value()] if brk else None,
            "group_colors": dict(self.group_colors),
            "group_order": list(self.group_order),
            "hue_order": list(self.hue_order),
            "titles": dict(self.custom_titles),
            "xlabel": self.custom_xlabel,
            "ylabels": dict(self.custom_ylabels),
            "filters": self.get_plot_filter_map(),
            "exclude_outliers": getattr(self, "exclude_outliers_cb", None) is not None
                                and self.exclude_outliers_cb.isChecked(),
        }

    @staticmethod
    def draw_metric(ax, df, spec, metric, dark=False, status=None):
        """Draw one metric of *spec* on *ax* (df already filtered).
        Returns the y column actually drawn."""
        x_col, y_col = spec["x"], metric
        plot_df = df
        if spec.get("exclude_outliers"):
            plot_df = StatisticsEngine.drop_outliers(plot_df, x_col, metric,
                                                     spec.get("hue"))
        if spec.get("normalize"):
            ref = spec.get("reference")
            if ref:
                plot_df = NormalizationEngine.wild_type_fold_change(
                    plot_df.copy(), metric, x_col, wt_label=ref)
                cand = f"{metric} (Fold Change)"
                if cand in plot_df.columns:
                    y_col = cand
                elif status:
                    status(f"Could not normalize '{metric}' to '{ref}' (no data / zero "
                           "mean) — showing raw values.")
            elif status:
                status("Select a reference group in 'Compare to' to normalize.")
        rng = spec.get("y_range")
        PlottingEngine.generate_plot(
            plot_df, spec["chart"], x_col, y_col, ax, dark_mode=dark,
            palette=spec["palette"], stat_mode=spec["stat"], hue_col=spec.get("hue"),
            log_scale=spec["log"], reference=spec.get("reference"),
            pairing_col=spec.get("pairing"), show_pvalue=spec["show_p"],
            y_min=rng[0] if rng else None, y_max=rng[1] if rng else None,
            group_colors=spec["group_colors"], bw_mode=spec["bw"], hist_bins=spec["bins"],
            custom_title=spec["titles"].get(y_col), custom_xlabel=spec.get("xlabel"),
            custom_ylabel=spec["ylabels"].get(y_col), group_order=spec["group_order"],
            show_ns=spec["show_ns"], hue_order=spec["hue_order"])
        return y_col

    def update_plot(self):
        if getattr(self, "_building", False):
            return
        df = self.get_active_plot_df()
        spec = self.plot_spec()
        x_col, metrics = spec["x"], spec["metrics"]

        if df is None or df.empty or not x_col or not metrics:
            self.plot_figure.clf()
            self.ax = self.plot_figure.add_subplot(111)
            self.axes = [self.ax]
            PlottingEngine.generate_plot(None, "", "", "", self.ax)
            self.plot_canvas.draw()
            return

        # Axis break (gap) only makes sense on a numeric X axis (Line plots)
        use_axis_break = spec["axis_break"] is not None and spec["chart"] == "Line"
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
        status = lambda m: self.status_bar.showMessage(m, 5000)
        for idx, metric in enumerate(metrics):
            if use_axis_break:
                ax_left, ax_right = axis_pairs[idx]
                self.draw_metric(ax_left, df, spec, metric, self.is_dark_mode, status)
                self.draw_metric(ax_right, df, spec, metric, self.is_dark_mode, status)
                self._apply_axis_break(ax_left, ax_right)
            else:
                self.draw_metric(self.axes[idx], df, spec, metric, self.is_dark_mode, status)

        self.plot_figure.tight_layout()
        self.plot_canvas.draw()
        self._refresh_groups_panel(df, x_col, metrics[0], spec["hue"], spec["chart"])
        self._mark_dirty()

    def update_summary_table_view(self, df=None):
        df = self.get_active_plot_df() if df is None else df
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
        self._push_undo()
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
        test = self.stat_mode_selector.currentText()
        if test == StatisticsEngine.NONE:
            test = StatisticsEngine.AUTO
        hue = self.get_hue()
        order = self._current_group_order()
        exclude = self.exclude_outliers_cb.isChecked()
        texts, rows, tables = [], [], []
        for y_var in metrics:
            flags = StatisticsEngine.find_outliers(df, x_var, y_var, hue)
            out_txt = self._outlier_text(df, flags, x_var, y_var, hue, exclude)
            data = df[~flags] if exclude else df
            if y_var == metrics[0]:
                self.update_summary_table_view(data)
            if test == StatisticsEngine.TWO_WAY and hue:
                table, results = StatisticsEngine.two_way(data, x_var, hue, y_var, self.hue_order)
                lines = ["=== TWO-WAY ANOVA ===", "",
                         f"'{y_var}' by '{x_var}' and '{hue}' (type II sums of squares)", ""]
                for t in table[:3]:
                    lines.append(f"  - {t['Effect']}: F({t['df']:g}, {table[3]['df']:g}) = "
                                 f"{t['F']:.3f}, {format_p(t['p'])} {stars(t['p'])}; "
                                 f"partial eta² = {t['partial eta²']:.3f}")
                lines += ["", f"Comparisons of '{hue}' within each '{x_var}' — pooled error "
                          "from the model, Šídák-corrected for all of them:"]
                for g, r in results.items():
                    for c in r.comparisons:
                        lines.append(f"  - {x_var} = {g}: [{c.a}] vs [{c.b}]: {format_p(c.p)} "
                                     f"{stars(c.p)}; difference = {c.diff:.4g} "
                                     f"(95% CI {c.ci[0]:.4g} to {c.ci[1]:.4g})")
                    rows += self._comparison_rows(r, y_var, f"{x_var} = {g}")
                tables.append(pd.DataFrame(table).assign(Metric=y_var))
                texts.append("\n".join(lines) + out_txt)
            elif hue:
                results = StatisticsEngine.compare_within(
                    data, x_var, hue, y_var, test, self.get_reference(),
                    self.get_pairing_col(), self.hue_order)
                for g, r in results.items():
                    texts.append(StatisticsEngine.format_report(r, hue, y_var, within=f"{x_var} = {g}"))
                    rows += self._comparison_rows(r, y_var, f"{x_var} = {g}")
                note = (f"Note: p-values are also Holm-corrected across the {len(results)} "
                        f"'{x_var}' groups." if len(results) > 1 else "")
                texts.append(note + out_txt)
            else:
                r = StatisticsEngine.compare(data, x_var, y_var, test, self.get_reference(),
                                             self.get_pairing_col(), group_order=order)
                texts.append(StatisticsEngine.format_report(r, x_var, y_var) + out_txt)
                rows += self._comparison_rows(r, y_var)
        self.current_comparisons = pd.DataFrame(rows)
        self.current_anova_tables = pd.concat(tables) if tables else None
        self.stats_output.setPlainText(("\n\n" + "-" * 60 + "\n\n").join(t for t in texts if t))

    @staticmethod
    def _outlier_text(df, flags, x_var, y_var, hue, excluded):
        if not flags.any():
            return "\n\nOutliers (Grubbs, alpha = 0.05): none."
        lines = ["", "", f"Outliers (Grubbs, alpha = 0.05) — "
                 + ("LEFT OUT of the test and plot:" if excluded else
                    "flagged only (tick 'Leave out outliers' to exclude):")]
        for i in flags[flags].index:
            grp = str(df.at[i, x_var]) + (f" / {df.at[i, hue]}" if hue else "")
            lines.append(f"  - {grp}: {df.at[i, y_var]:.6g} (row {i + 1})")
        return "\n".join(lines)

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

    # ---------------------------------------------------------------
    # Projects (.bse) and undo: both use the full state
    # ---------------------------------------------------------------
    def _plate_state(self):
        return {"size": self.plate_size_selector.currentText(),
                "values": {n: e.get_grid_values() for n, e in self.value_layer_editors.items()},
                "maps": {n: e.get_grid_values() for n, e in self.map_layer_editors.items()},
                "series": [dict(p) for p in getattr(self, "plate_series", [])],
                "norm": self._plate_norm_state() if hasattr(self, "_plate_norm_state") else {}}

    def _restore_plate(self, st):
        if not st:
            return
        for tabs, editors in ((self.value_layers_tabs, self.value_layer_editors),
                              (self.map_layers_tabs, self.map_layer_editors)):
            while tabs.count():
                tabs.removeTab(0)
            editors.clear()
        self.plate_size_selector.blockSignals(True)
        self.plate_size_selector.setCurrentText(st.get("size", "96-well (8x12)"))
        self.plate_size_selector.blockSignals(False)
        for key, tabs, editors in (("values", self.value_layers_tabs, self.value_layer_editors),
                                   ("maps", self.map_layers_tabs, self.map_layer_editors)):
            for name, cells in st.get(key, {}).items():
                ed = self._new_plate_grid_tab(tabs, editors, name)
                if ed is not None:
                    ed.set_cells(cells)
        self.plate_series = [dict(p) for p in st.get("series", [])]
        if hasattr(self, "_restore_plate_norm"):
            self._restore_plate_norm(st.get("norm", {}))
        if hasattr(self, "_refresh_series_label"):
            self._refresh_series_label()

    def state(self, for_file=False):
        """Everything needed to rebuild this analysis."""
        df = self.master_df
        return {
            "data": project.frame_to_json(df) if for_file else (None if df is None else df.copy()),
            "spec": self.plot_spec(),
            "dock_filters": {k: list(v) for k, v in self.dock_filter_selections.items()},
            "plate": self._plate_state(),
            "layout": self.layout_tab.to_list(for_file) if hasattr(self, "layout_tab") else [],
            "curves": self.curves_tab.to_dict() if hasattr(self, "curves_tab") else {},
        }

    def apply_state(self, st):
        self._restoring = True
        try:
            data = st.get("data")
            df = project.frame_from_json(data) if isinstance(data, dict) else data
            if df is not None and not df.empty:
                self.ingest_dataframe(df.copy(), append=False)
            else:
                self.master_df = self.filtered_df = None
            self.dock_filter_selections = {k: list(v) for k, v in
                                           st.get("dock_filters", {}).items()}
            if self.master_df is not None:
                self.filtered_df = FilterEngine.apply_multi_filter(
                    self.master_df, self.dock_filter_selections)
                self.display_data(self.filtered_df)
            self._apply_spec(st.get("spec", {}))
            self._restore_plate(st.get("plate"))
            if hasattr(self, "layout_tab"):
                self.layout_tab.from_list(st.get("layout", []))
            if hasattr(self, "curves_tab"):
                self.curves_tab.from_dict(st.get("curves", {}))
        finally:
            self._restoring = False
        self.update_plot()

    def _apply_spec(self, sp):
        if not sp:
            return
        widgets = [self.chart_selector, self.x_selector, self.y_selector, self.hue_selector,
                   self.reference_selector, self.pairing_selector, self.palette_selector,
                   self.stat_mode_selector]
        for w in widgets:
            w.blockSignals(True)
        try:
            for combo, key in ((self.chart_selector, "chart"), (self.x_selector, "x"),
                               (self.y_selector, "y"), (self.palette_selector, "palette"),
                               (self.stat_mode_selector, "stat")):
                if sp.get(key) and combo.findText(sp[key]) >= 0:
                    combo.setCurrentText(sp[key])
            self.rebuild_hue_and_pairing_selectors()
            self.rebuild_reference_selector()
            for combo, key in ((self.hue_selector, "hue"), (self.reference_selector, "reference"),
                               (self.pairing_selector, "pairing")):
                v = sp.get(key) or NO_SELECTION
                combo.setCurrentText(v if combo.findText(v) >= 0 else NO_SELECTION)
        finally:
            for w in widgets:
                w.blockSignals(False)
        checks = ((self.show_all_metrics_cb, "all_metrics"), (self.normalize_ref_cb, "normalize"),
                  (self.log_scale_cb, "log"), (self.bw_mode_cb, "bw"),
                  (self.show_pvalue_cb, "show_p"), (self.show_ns_cb, "show_ns"),
                  (self.exclude_outliers_cb, "exclude_outliers"))
        for cb, key in checks:
            cb.blockSignals(True)
            cb.setChecked(bool(sp.get(key)))
            cb.blockSignals(False)
        self.hist_bins_spin.blockSignals(True)
        self.hist_bins_spin.setValue(int(sp.get("bins", 20)))
        self.hist_bins_spin.blockSignals(False)
        rng = sp.get("y_range")
        vals = rng or sp.get("y_range_values") or [0.0, 100.0]
        self.manual_yrange_cb.blockSignals(True)
        self.manual_yrange_cb.setChecked(rng is not None)
        self.manual_yrange_cb.blockSignals(False)
        for spin, v in zip((self.y_min_spin, self.y_max_spin), vals):
            spin.blockSignals(True)
            spin.setValue(float(v))
            spin.blockSignals(False)
        brk = sp.get("axis_break")
        self.axis_break_cb.blockSignals(True)
        self.axis_break_cb.setChecked(brk is not None)
        self.axis_break_cb.blockSignals(False)
        if brk:
            for spin, v in zip((self.axis_break_left_min, self.axis_break_left_max,
                                self.axis_break_right_min, self.axis_break_right_max), brk):
                spin.blockSignals(True)
                spin.setValue(float(v))
                spin.blockSignals(False)
        self.group_colors = dict(sp.get("group_colors", {}))
        self.group_order = list(sp.get("group_order", []))
        self.hue_order = list(sp.get("hue_order", []))
        self.custom_titles = dict(sp.get("titles", {}))
        self.custom_xlabel = sp.get("xlabel")
        self.custom_ylabels = dict(sp.get("ylabels", {}))
        filters = sp.get("filters", {})
        for col, lst in self.metadata_checklists.items():
            if col not in filters:
                continue
            keep = set(filters[col])
            lst.blockSignals(True)
            for i in range(lst.count()):
                it = lst.item(i)
                it.setCheckState(Qt.CheckState.Checked if it.text() in keep
                                 else Qt.CheckState.Unchecked)
            lst.blockSignals(False)
        self._sync_control_visibility()

    # --- undo ---
    UNDO_LIMIT = 30

    def _push_undo(self):
        if getattr(self, "_restoring", False) or getattr(self, "_building", False):
            return
        self._undo_stack.append(self.state())
        del self._undo_stack[:-self.UNDO_LIMIT]
        self._redo_stack.clear()
        self._sync_undo_actions()
        self._mark_dirty()

    def _sync_undo_actions(self):
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(bool(self._undo_stack))
            self.redo_action.setEnabled(bool(self._redo_stack))

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self.state())
        self.apply_state(self._undo_stack.pop())
        self._sync_undo_actions()
        self.status_bar.showMessage("Undone", 2500)

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self.state())
        self.apply_state(self._redo_stack.pop())
        self._sync_undo_actions()
        self.status_bar.showMessage("Redone", 2500)

    # --- project files ---
    def _update_title(self):
        name = os.path.basename(self._project_path) if self._project_path else "Untitled"
        self.setWindowTitle(f"{name}[*] — BioStat Explorer")
        self.setWindowModified(self._dirty)

    def _mark_dirty(self):
        if getattr(self, "_restoring", False) or getattr(self, "_building", False):
            return
        if self.master_df is not None and not self._dirty:
            self._dirty = True
            self._update_title()

    def save_project(self):
        if not self._project_path:
            return self.save_project_as()
        try:
            project.save(self._project_path, self.state(for_file=True))
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return False
        self._dirty = False
        self._update_title()
        self.status_bar.showMessage(f"Saved {os.path.basename(self._project_path)}", 4000)
        return True

    def save_project_as(self):
        path = self._save_path("Save Project", "analysis.bse", "bse", project.FILTER)
        if not path:
            return False
        self._project_path = path
        return self.save_project()

    def open_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Project", self._last_dir(),
                                                  project.FILTER)
            if not path:
                return
        if not self._confirm_discard():
            return
        try:
            st = project.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not open", str(e))
            return
        self._remember_dir(path)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.apply_state(st)
        self._project_path = path
        self._dirty = False
        self._sync_undo_actions()
        self._update_title()
        self.tabs.setCurrentWidget(self.graph_tab)
        self.status_bar.showMessage(f"Opened {os.path.basename(path)}", 4000)

    def _confirm_discard(self):
        if not self._dirty or self.master_df is None:
            return True
        name = os.path.basename(self._project_path) if self._project_path else "this analysis"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Save changes?")
        box.setText(f"Save the changes to {name}?")
        box.setInformativeText("Your data, graph settings, colours, plate grids and layout "
                               "are kept in a project file you can reopen later.")
        box.setStandardButtons(QMessageBox.StandardButton.Save |
                               QMessageBox.StandardButton.Discard |
                               QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        answer = box.exec()
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        super().closeEvent(event)

    # --- figure layout ---
    def _add_to_layout(self):
        df = self.get_active_plot_df()
        spec = self.plot_spec()
        if df is None or df.empty or not spec["metrics"]:
            QMessageBox.information(self, "Layout", "Set up a graph first.")
            return
        self._push_undo()
        for metric in spec["metrics"]:
            self.layout_tab.add_panel(dict(spec, metrics=[metric]), df)
        n = len(self.layout_tab.panels)
        self.status_bar.showMessage(f"Added to the Layout tab ({n} panel"
                                    f"{'s' if n != 1 else ''})", 5000)

    def _export_layout(self, fmt):
        if not self.layout_tab.panels:
            QMessageBox.information(self, "Layout", "Add some graphs to the layout first.")
            return
        names = {"pdf": "PDF", "svg": "SVG (editable text)", "png": "PNG image",
                 "tiff": "TIFF image"}
        path = self._save_path("Export Figure Page", f"figure.{fmt}", fmt,
                               f"{names[fmt]} (*.{fmt})")
        if path:
            self.layout_tab.export_figure(path, fmt, self._export_dpi())
            w, h, _, _ = self.layout_tab.page_size_mm()
            self.status_bar.showMessage(f"Saved {path} ({w:.0f} × {h:.0f} mm)", 8000)

    # --- curves ---
    def _use_curve_parameters(self, params):
        if params is None or params.empty:
            QMessageBox.information(self, "Curves", "Fit some curves first.")
            return
        group = self.curves_tab.color.currentText()
        self.ingest_dataframe(params.drop(columns=["Note"], errors="ignore"),
                              append=False, source_label="Curve fits")
        if group in self.metadata_cols:
            self.x_selector.setCurrentText(group)
        self.tabs.setCurrentWidget(self.graph_tab)
        self.status_bar.showMessage("Now plotting the fitted parameters — Undo (⌘Z) "
                                    "brings your data back", 10000)

    def _export_curve_parameters(self, params):
        if params is None or params.empty:
            return
        path = self._save_path("Export Parameters", "curve_parameters.xlsx", "xlsx",
                               "Excel (*.xlsx)")
        if path:
            params.to_excel(path, index=False)
            self.status_bar.showMessage(f"Saved {path}", 6000)

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
        dpi = self._export_dpi()
        labels = {"png": f"PNG image, {dpi} dpi", "svg": "SVG (editable text)",
                  "pdf": "PDF", "tiff": f"TIFF image, {dpi} dpi"}
        path = self._save_path("Export Figure", f"figure.{format_type}", format_type,
                               f"{labels[format_type]} (*.{format_type})")
        if path:
            ExportEngine.save_figure_to(self.plot_figure, path, format_type, dpi=dpi)
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
                getattr(self, "current_comparisons", None),
                getattr(self, "current_anova_tables", None))
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
            ExportEngine.save_figure_to(self.plot_figure, tmp, "png", dpi=self._export_dpi())
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


def selftest():
    """Checks a built app: statistics, plate parsing and exports work."""
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    checks = {}
    df = pd.DataFrame({"g": list("AAAABBBBCCCC"),
                       "v": [1, 1.2, .9, 1.1, 1.6, 1.5, 1.9, 1.4, 1.2, 1.3, 1.25, 1.35],
                       "rep": [1, 2, 3, 4] * 3})
    SE = StatisticsEngine
    checks["anova"] = abs(SE.compare(df, "g", "v", SE.ANOVA_TUKEY).omnibus_p - 0.00203) < 1e-3
    checks["dunnett"] = len(SE.compare(df, "g", "v", SE.ANOVA_TUKEY, reference="A").comparisons) == 2
    checks["dunn"] = len(SE.compare(df, "g", "v", SE.KRUSKAL_DUNN).comparisons) == 3
    checks["rm_anova"] = SE.compare(df, "g", "v", SE.PAIRED_T, pairing_col="rep").omnibus_p is not None
    grid = pd.DataFrame([[None, 1, 2, 3, 4], ["A", 1, 2, 3, 4], ["B", 5, 6, 7, 8]])
    checks["plate"] = len(pr.find_plate_blocks(grid)) == 1
    fig, ax = plt.subplots()
    from src.plotting import PlottingEngine
    PlottingEngine.generate_plot(df, "Box + Swarm", "g", "v", ax, stat_mode=SE.AUTO)
    d = tempfile.mkdtemp()
    for fmt in ("png", "svg", "pdf", "tiff"):
        ExportEngine.save_figure_to(fig, os.path.join(d, "f." + fmt), fmt, dpi=100)
    checks["export"] = all(os.path.getsize(os.path.join(d, "f." + f)) > 500
                           for f in ("png", "svg", "pdf", "tiff"))
    import openpyxl, reportlab, statsmodels  # noqa: F401  (bundled?)
    ok = all(checks.values())
    print(("selftest OK " if ok else "selftest FAILED ") + str(checks))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    app = QApplication(sys.argv)
    window = BioStatExplorer()
    window.show()
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".bse"):
            window.open_project(arg)
    sys.exit(app.exec())
