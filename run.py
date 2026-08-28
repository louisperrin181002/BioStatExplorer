# run.py
import sys
import os
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QDockWidget, QFileDialog,
    QLabel, QStatusBar, QListWidget, QListWidgetItem, QComboBox, QTabWidget, QPushButton,
    QAbstractItemView, QTextEdit, QGroupBox, QDoubleSpinBox, QCheckBox, QScrollArea,
    QSpinBox, QLineEdit, QInputDialog, QMessageBox, QGridLayout, QColorDialog, QMenu
)
from PySide6.QtGui import QAction, QKeySequence, QColor, QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.excel_loader import DataLoader
from src.plotting import PlottingEngine
from src.filters import FilterEngine
from src.normalization import NormalizationEngine
from src.statistics_engine import StatisticsEngine
from src.utils import SummaryStatistics
from src.export_engine import ExportEngine
from src.report_generator import PDFReportGenerator
from src import plate_reader as pr

NO_SELECTION = "(none)"


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
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Plate Grid", "", "Excel/CSV Files (*.xlsx *.xls *.csv)")
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".csv"):
                raw = pd.read_csv(file_path, header=None)
            else:
                raw = pd.read_excel(file_path, header=None)
            loc = pr.detect_plate_grid(raw)
            if loc is None:
                QMessageBox.warning(self, "Plate not found",
                    "Could not auto-detect a plate layout (row letters beside column numbers) "
                    "in this file. Please check the file or enter values manually.")
                return
            row_start, col_start, n_rows, n_cols = loc
            self.set_shape(n_rows, n_cols)
            block = raw.iloc[row_start:row_start + n_rows, col_start:col_start + n_cols].values
            for r in range(n_rows):
                for c in range(n_cols):
                    v = block[r][c]
                    text = "" if pd.isna(v) else str(v)
                    self.table.item(r, c).setText(text)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Could not read this file: {e}")

    def get_grid_values(self):
        grid = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                item = self.table.item(r, c)
                row.append(item.text() if item is not None else "")
            grid.append(row)
        return grid


class DropZoneWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel("Drag & Drop Excel or CSV files here\n- or -\nUse File > Open")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("dropLabel")
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setAcceptDrops(True)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile().endswith(('.xlsx', '.xls', '.csv'))]
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

        self.init_menu()
        self.init_central_widget()
        self.init_dock_panels()
        self.apply_theme()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def init_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open File(s)", self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

    def init_central_widget(self):
        self.tabs = QTabWidget()

        # Tab 1: Data View
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.drop_zone = DropZoneWidget(self)
        self.table_viewer = QTableWidget()
        self.table_viewer.hide()
        self.data_layout.addWidget(self.drop_zone)
        self.data_layout.addWidget(self.table_viewer)

        # Tab 2: Graphing View
        self.graph_tab = QWidget()
        self.graph_layout = QHBoxLayout(self.graph_tab)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_host = QWidget()
        self.control_panel = QVBoxLayout(control_host)
        control_scroll.setWidget(control_host)

        # --- Styling ---
        style_group = QGroupBox("UI & Plot Styling")
        style_box = QVBoxLayout()
        self.dark_mode_cb = QCheckBox("Enable Dark Mode")
        self.dark_mode_cb.toggled.connect(self.toggle_dark_theme)
        style_box.addWidget(self.dark_mode_cb)

        style_box.addWidget(QLabel("Plot Color Palette:"))
        self.palette_selector = QComboBox()
        self.palette_selector.addItems(["muted", "deep", "colorblind", "rocket", "crest"])
        self.palette_selector.currentTextChanged.connect(self.update_plot)
        style_box.addWidget(self.palette_selector)

        self.bw_mode_cb = QCheckBox("Black & white mode (grayscale + hatch/linestyle)")
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

        stat_group = QGroupBox("Prism Inference Core")
        stat_box = QVBoxLayout()
        self.btn_run_stats = QPushButton("Run Statistical Test")
        self.btn_run_stats.setObjectName("successBtn")
        self.btn_run_stats.clicked.connect(self.run_statistical_pipeline)
        stat_box.addWidget(self.btn_run_stats)

        self.btn_export_stats = QPushButton("Export Summary (Excel)")
        self.btn_export_stats.clicked.connect(self.export_stats_data)
        stat_box.addWidget(self.btn_export_stats)

        self.btn_export_pdf = QPushButton("Compile PDF Report")
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

        right_analysis_layout.addWidget(QLabel("<b>Statistical Inference Matrix:</b>"))
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
        self.dock = QDockWidget("Data Slicing Engine", self)
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
        if self.is_dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #1E1E1E; color: #E0E0E0; font-family: 'Segoe UI', Arial; }
                QTableWidget { background-color: #2D2D2D; gridline-color: #3F3F3F; color: #E0E0E0; }
                QHeaderView::section { background-color: #3A3A3A; color: #E0E0E0; border: 1px solid #2D2D2D; }
                QComboBox, QDoubleSpinBox, QTextEdit, QListWidget { background-color: #2D2D2D; border: 1px solid #444; color: #E0E0E0; border-radius: 4px; padding: 4px; }
                QPushButton { background-color: #3D3D3D; border: 1px solid #555; color: #E0E0E0; border-radius: 4px; padding: 6px; }
                QPushButton:hover { background-color: #4D4D4D; }
                QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 12px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
                #dropLabel { border: 2px dashed #4A90E2; border-radius: 8px; background-color: #252830; color: #A0A5B0; font-size: 14px; }
                #primaryBtn { background-color: #2980B9; color: white; font-weight: bold; }
                #successBtn { background-color: #27AE60; color: white; font-weight: bold; }
                #dangerBtn { background-color: #C0392B; color: white; font-weight: bold; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #FAFAFA; color: #2C3E50; font-family: 'Segoe UI', Arial; }
                QTableWidget { background-color: white; gridline-color: #E2E8F0; color: #2C3E50; }
                QHeaderView::section { background-color: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
                QComboBox, QDoubleSpinBox, QTextEdit, QListWidget { background-color: white; border: 1px solid #CBD5E1; color: #2C3E50; border-radius: 4px; padding: 4px; }
                QPushButton { background-color: #F1F5F9; border: 1px solid #CBD5E1; color: #475569; border-radius: 4px; padding: 6px; }
                QPushButton:hover { background-color: #E2E8F0; }
                QGroupBox { border: 1px solid #E2E8F0; border-radius: 6px; margin-top: 12px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
                #dropLabel { border: 2px dashed #4A90E2; border-radius: 8px; background-color: #F0F4F8; color: #333333; font-size: 14px; }
                #primaryBtn { background-color: #4A90E2; color: white; font-weight: bold; }
                #successBtn { background-color: #2ECC71; color: white; font-weight: bold; }
                #dangerBtn { background-color: #E74C3C; color: white; font-weight: bold; }
            """)

    def toggle_dark_theme(self, checked):
        self.is_dark_mode = checked
        self.apply_theme()
        self.update_plot()

    def open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Data File(s)", "", "Excel/CSV Files (*.xlsx *.xls *.csv)")
        if files: self.process_files(files)

    def process_files(self, file_paths):
        try:
            df, recognized, metadata = self.data_loader.load_and_merge(file_paths)
            if df.empty: return
            self.recognized_cols = recognized
            self.ingest_dataframe(df, append=False)
        except Exception as e: self.status_bar.showMessage(f"Engine Error: {str(e)}", 5000)

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
        # everything else (including recognized-but-categorical columns like
        # "Organism"/"Medium") is filterable metadata. This matters because
        # column_detection's "recognized" list mixes numeric signal columns
        # (GFP, mKate) together with categorical labels (Organism, Medium) —
        # only the former should be excluded from the filter checklists.
        self.metric_cols = [c for c in combined.columns if pd.api.types.is_numeric_dtype(combined[c])]
        self.metadata_cols = [c for c in combined.columns if c not in self.metric_cols]
        if "Source File" not in self.metadata_cols and "Source File" in combined.columns:
            self.metadata_cols.append("Source File")
        self.display_data(self.filtered_df)
        self.update_selectors()
        self.metadata_col_list.clear()
        self.metadata_col_list.addItems(self.metadata_cols)
        self.rebuild_data_selection_panel()
        self.status_bar.showMessage(f"Loaded {combined.shape[0]} rows, {combined.shape[1]} columns.", 4000)

    # ---------------------------------------------------------------
    # Selector population
    # ---------------------------------------------------------------
    def update_selectors(self):
        cols = list(self.filtered_df.columns) if self.filtered_df is not None else []
        self.x_selector.blockSignals(True)
        self.y_selector.blockSignals(True)
        self.x_selector.clear()
        self.y_selector.clear()
        self.x_selector.addItems(cols)
        # Prefer numeric metric columns for the Metric dropdown, if any exist
        metric_candidates = [c for c in getattr(self, 'metric_cols', []) if c in cols] or cols
        self.y_selector.addItems(metric_candidates)
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
        self.table_viewer.setRowCount(df.shape[0])
        self.table_viewer.setColumnCount(df.shape[1])
        self.table_viewer.setHorizontalHeaderLabels(df.columns.astype(str))
        for row_idx in range(df.shape[0]):
            for col_idx in range(df.shape[1]):
                val = df.iat[row_idx, col_idx]
                item = QTableWidgetItem(str(val) if pd.notna(val) else "")
                self.table_viewer.setItem(row_idx, col_idx, item)
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
        self.current_summary_df = SummaryStatistics.generate_table(df, x_var, y_var)
        if self.current_summary_df.empty: return
        self.summary_table_view.setRowCount(self.current_summary_df.shape[0])
        self.summary_table_view.setColumnCount(self.current_summary_df.shape[1])
        self.summary_table_view.setHorizontalHeaderLabels(self.current_summary_df.columns)
        for r in range(self.current_summary_df.shape[0]):
            for c in range(self.current_summary_df.shape[1]):
                self.summary_table_view.setItem(r, c, QTableWidgetItem(str(self.current_summary_df.iat[r, c])))
        self.summary_table_view.resizeColumnsToContents()

    def run_bg_sub(self):
        if self.filtered_df is None: return
        metrics = self.get_selected_metrics()
        if not metrics: return
        self.filtered_df = NormalizationEngine.background_subtraction(self.filtered_df, metrics[0], self.bg_spin.value())
        self.display_data(self.filtered_df)
        self.update_selectors()

    def run_wt_fold(self):
        if self.filtered_df is None: return
        metrics = self.get_selected_metrics()
        reference = self.get_reference()
        if not metrics or not reference:
            self.status_bar.showMessage("Select a Metric and a reference group ('Compare To') first.", 5000)
            return
        self.filtered_df = NormalizationEngine.wild_type_fold_change(
            self.filtered_df, metrics[0], self.x_selector.currentText(), wt_label=reference)
        self.display_data(self.filtered_df)
        self.update_selectors()

    def run_statistical_pipeline(self):
        df = self.get_active_plot_df()
        x_var = self.x_selector.currentText()
        metrics = self.get_selected_metrics()
        y_var = metrics[0] if metrics else None
        if not x_var or not y_var or df is None: return
        self.update_summary_table_view()

        test_override = self.stat_mode_selector.currentText()
        if test_override == StatisticsEngine.NONE:
            test_override = StatisticsEngine.AUTO

        report = StatisticsEngine.analyze_groups(
            df, x_var, y_var,
            test_override=test_override,
            reference=self.get_reference(),
            pairing_col=self.get_pairing_col()
        )
        self.stats_output.setText(report)

    def export_current_plot(self, format_type):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure", f"publication_figure.{format_type}", f"Images (*.{format_type})")
        if file_path:
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path).split('.')[0]
            saved = ExportEngine.save_figure(self.plot_figure, directory, filename, format_type)
            self.status_bar.showMessage(f"Figure saved successfully to: {saved}", 5000)

    def export_stats_data(self):
        if self.current_summary_df is None or self.current_summary_df.empty: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Analysis Data", "statistical_report.xlsx", "Excel Sheets (*.xlsx)")
        if file_path:
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path).split('.')[0]
            saved = ExportEngine.save_excel_summary(self.current_summary_df, self.stats_output.toPlainText(), directory, filename)
            self.status_bar.showMessage(f"Excel report saved to: {saved}", 5000)

    def export_pdf_report(self):
        if self.current_summary_df is None or self.current_summary_df.empty: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Compile Lab Dossier", "analytical_dossier.pdf", "Documents (*.pdf)")
        if file_path:
            self.status_bar.showMessage("Generating PDF layout elements...")
            temp_img = os.path.join(os.path.dirname(file_path), "_temp_report_plot.png")
            ExportEngine.save_figure(self.plot_figure, os.path.dirname(file_path), "_temp_report_plot", "png", dpi=300)
            PDFReportGenerator.generate_pdf(file_path, self.current_summary_df, self.stats_output.toPlainText(), temp_img)
            if os.path.exists(temp_img): os.remove(temp_img)
            self.status_bar.showMessage(f"Dossier compiled successfully: {file_path}", 6000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BioStatExplorer()
    window.show()
    sys.exit(app.exec())
