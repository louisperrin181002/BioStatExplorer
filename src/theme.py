# src/theme.py
"""Light and dark styles for the whole window (one place for colours)."""

LIGHT = dict(bg="#F6F7F9", surface="#FFFFFF", surface2="#F1F5F9",
             border="#E2E8F0", border2="#CBD5E1", text="#0F172A",
             muted="#64748B", accent="#2563EB", accent_hi="#1D4ED8",
             accent_bg="#EFF6FF", good="#16A34A", warn="#B45309",
             bad="#DC2626", plot_text="#1E293B", plot_grid="#E2E8F0",
             plot_line="#334155")
DARK = dict(bg="#0F1115", surface="#171A21", surface2="#1F232C",
            border="#2A2F3A", border2="#3A4150", text="#E5E7EB",
            muted="#94A3B8", accent="#3B82F6", accent_hi="#60A5FA",
            accent_bg="#1E293B", good="#22C55E", warn="#F59E0B",
            bad="#F87171", plot_text="#E5E7EB", plot_grid="#2A2F3A",
            plot_line="#CBD5E1")


def stylesheet(dark=False):
    t = DARK if dark else LIGHT
    return f"""
QMainWindow, QWidget {{ background: {t['bg']}; color: {t['text']}; }}
QToolBar {{ background: {t['surface']}; border: none;
    border-bottom: 1px solid {t['border']}; padding: 6px 10px; spacing: 6px; }}
QToolBar QToolButton {{ padding: 6px 12px; border-radius: 6px; border: none;
    background: transparent; color: {t['text']}; font-weight: 500; }}
QToolBar QToolButton:hover {{ background: {t['surface2']}; }}
QToolBar QToolButton#Primary {{ background: {t['accent']}; color: white; }}
QToolBar QToolButton#Primary:hover {{ background: {t['accent_hi']}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QTabWidget::pane {{ border: none; background: {t['bg']}; }}
QTabBar::tab {{ background: transparent; color: {t['muted']};
    padding: 9px 16px; border: none; border-bottom: 2px solid transparent;
    font-weight: 500; }}
QTabBar::tab:selected {{ color: {t['accent']};
    border-bottom: 2px solid {t['accent']}; }}
QTabBar::tab:hover {{ color: {t['text']}; }}
QGroupBox {{ background: {t['surface']}; border: 1px solid {t['border']};
    border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px;
    font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: {t['muted']}; }}
QLabel {{ background: transparent; }}
QLabel#Hint {{ color: {t['muted']}; }}
QLabel#Title {{ font-size: 15px; font-weight: 600; }}
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit, QPlainTextEdit,
QListWidget {{ background: {t['surface']}; border: 1px solid {t['border2']};
    border-radius: 6px; padding: 4px 6px; color: {t['text']};
    selection-background-color: {t['accent']}; }}
QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {t['accent']}; }}
QComboBox QAbstractItemView {{ background: {t['surface']};
    border: 1px solid {t['border2']}; selection-background-color:
    {t['accent_bg']}; selection-color: {t['text']}; }}
QPushButton {{ background: {t['surface']}; border: 1px solid {t['border2']};
    border-radius: 6px; padding: 6px 12px; color: {t['text']}; }}
QPushButton:hover {{ background: {t['surface2']}; }}
QPushButton:disabled {{ color: {t['muted']}; }}
QPushButton#primaryBtn, QPushButton#successBtn {{ background: {t['accent']};
    border: 1px solid {t['accent']}; color: white; font-weight: 600; }}
QPushButton#primaryBtn:hover, QPushButton#successBtn:hover {{
    background: {t['accent_hi']}; }}
QPushButton#dangerBtn {{ background: {t['surface']};
    border: 1px solid {t['border2']}; color: {t['text']}; }}
QTableView, QTableWidget {{ background: {t['surface']};
    alternate-background-color: {t['surface2']};
    gridline-color: {t['border']}; border: 1px solid {t['border']};
    border-radius: 6px; selection-background-color: {t['accent_bg']};
    selection-color: {t['text']}; }}
QHeaderView::section {{ background: {t['surface2']}; color: {t['muted']};
    border: none; border-right: 1px solid {t['border']};
    border-bottom: 1px solid {t['border']}; padding: 5px 6px;
    font-weight: 600; }}
QScrollArea {{ border: none; }}
QDockWidget::title {{ background: {t['surface']}; padding: 6px;
    font-weight: 600; }}
QStatusBar {{ background: {t['surface']}; color: {t['muted']};
    border-top: 1px solid {t['border']}; }}
QCheckBox {{ background: transparent; spacing: 7px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px;
    border: 1px solid {t['border2']}; background: {t['surface']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']};
    border: 1px solid {t['accent']}; image: none; }}
QCheckBox::indicator:hover {{ border: 1px solid {t['accent']}; }}
QMenu {{ background: {t['surface']}; border: 1px solid {t['border2']};
    padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['accent_bg']}; color: {t['text']}; }}
QLabel#Section {{ color: {t['muted']}; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.6px; padding-top: 8px; }}
QWidget#Card {{ background: {t['surface']}; border: 1px solid {t['border']};
    border-radius: 10px; }}
QTabWidget#ControlTabs::pane {{ background: {t['surface']};
    border: 1px solid {t['border']}; border-radius: 10px; top: -1px; }}
QTabWidget#ControlTabs QScrollArea, QTabWidget#ControlTabs QScrollArea > QWidget > QWidget {{
    background: {t['surface']}; }}
QTabWidget#ControlTabs QTabBar::tab {{ padding: 7px 14px; }}
QListWidget {{ outline: none; }}
QListWidget::item {{ padding: 4px 4px; border-radius: 5px; }}
QListWidget::item:selected {{ background: {t['accent_bg']}; color: {t['text']}; }}
QListWidget::item:hover {{ background: {t['surface2']}; }}
QToolButton {{ background: {t['surface']}; border: 1px solid {t['border2']};
    border-radius: 5px; padding: 3px 8px; color: {t['text']}; }}
QToolButton:hover {{ background: {t['surface2']}; }}
#dropLabel {{ border: 2px dashed {t['border2']}; border-radius: 12px;
    background: {t['surface']}; color: {t['muted']}; font-size: 15px; }}
"""


def plot_colors(dark=False):
    t = DARK if dark else LIGHT
    return dict(bg=t["surface"], text=t["plot_text"], grid=t["plot_grid"],
                line=t["plot_line"])
