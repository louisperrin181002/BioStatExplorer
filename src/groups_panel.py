# src/groups_panel.py
"""The Groups panel beside the graph: every group with its colour.
Drag a row to move that group; click its colour square to change it."""

# ---------------------------------------------------------------------------
# HOW THIS PANEL WORKS
# ---------------------------------------------------------------------------
# Two lists, both of the same _GroupList class:
#
#   x_list  the groups along the X axis, in plotting order
#   s_list  the Color By levels, shown only when there are any
#
# Which list owns the colours depends on the plot. With no Color By
# column the X groups are the series and carry the swatches; as soon as
# one is chosen the colours belong to it instead, and x_list becomes
# order-only. set_groups() decides that and passes with_colors down.
#
# The panel never changes the plot itself. Every action ends in a signal
# — order_changed, color_changed and so on — and the main window
# redraws. That keeps the ordering logic in one place rather than split
# between the panel and the plotting code.
#
# One Qt detail worth knowing before reading on: the visible text of a
# row and its identity are kept apart. The name is stored in the _NAME
# role, because the display text can be elided by the widget and would
# then no longer match the group it stands for.
# ---------------------------------------------------------------------------

from PySide6.QtCore import Qt, Signal, QSize, QRect
from PySide6.QtGui import QColor, QPixmap, QPainter, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QAbstractItemView, QPushButton, QColorDialog, QMenu, QToolButton,
)

#: Side of the colour square, in pixels. Also the width of the click
#: target in mousePressEvent, so the two cannot drift apart.
SWATCH = 18

#: Custom item role holding the group's real name — see the note above.
#: UserRole + 1 rather than UserRole, leaving the first free for Qt's own
#: conventional uses.
_NAME = Qt.ItemDataRole.UserRole + 1


def swatch_icon(color, size=SWATCH):
    """A rounded colour square to put beside a group's name.

    Painted rather than loaded, since the colour is only known at
    runtime. Transparent background so the rounded corners show the
    list's own colour through them instead of white.
    """
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    # A faint outline, so a pale swatch is still visible against a white
    # row — without it, white and "no colour" look the same.
    p.setPen(QColor(0, 0, 0, 40))
    p.setBrush(QColor(color))
    p.drawRoundedRect(1, 1, size - 2, size - 2, 5, 5)
    p.end()
    return QIcon(pm)


class _GroupList(QListWidget):
    """Draggable list; a click on the colour square asks for a colour."""

    moved = Signal()
    swatch_clicked = Signal(str)

    def __init__(self, with_colors=True, parent=None):
        super().__init__(parent)
        self.with_colors = with_colors
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setIconSize(QSize(SWATCH, SWATCH))
        self.setSpacing(1)
        # Guard against the panel reacting to its own rebuilds. _fill()
        # clears and repopulates the list, which Qt reports as rows
        # moving; without the flag that would emit moved, redraw the
        # plot, and in the sort case recurse.
        self.filling = False
        self.model().rowsMoved.connect(
            lambda *_: None if self.filling else self.moved.emit())

    def names(self):
        return [self.item(i).data(_NAME) for i in range(self.count())]

    def mousePressEvent(self, event):
        """A click on the swatch opens the colour picker; anything else
        is a normal click, so drag-to-reorder keeps working."""
        it = self.itemAt(event.position().toPoint())
        if it is not None and self.with_colors:
            # The hit area is the swatch plus 10px of slack, over the full
            # row height. Qt gives no way to ask "was the icon clicked",
            # so the rectangle is reconstructed from the row's own.
            r = self.visualItemRect(it)
            hit = QRect(r.left(), r.top(), SWATCH + 10, r.height())
            if hit.contains(event.position().toPoint()) and \
                    event.button() == Qt.MouseButton.LeftButton:
                self.setCurrentItem(it)
                self.swatch_clicked.emit(it.data(_NAME))
                # Returning without calling super() swallows the event,
                # so the click does not also start a drag.
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        it = self.itemAt(event.position().toPoint())
        if it is not None and self.with_colors:
            self.swatch_clicked.emit(it.data(_NAME))
            return
        super().mouseDoubleClickEvent(event)


class GroupsPanel(QWidget):
    """Signals: order_changed(list) for the X axis, series_order_changed(list)
    for Color-By series, color_changed(name, hex), colors_reset(),
    order_reset()."""

    order_changed = Signal(list)
    series_order_changed = Signal(list)
    color_changed = Signal(str, str)
    colors_reset = Signal()
    order_reset = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(210)
        self.setMaximumWidth(280)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(6)
        title = QLabel("Groups")
        title.setObjectName("Title")
        lay.addWidget(title)
        hint = QLabel("Drag to reorder · click a colour to change it")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._x_label = QLabel("X axis")
        self._x_label.setObjectName("Section")
        lay.addWidget(self._x_label)
        self.x_list = _GroupList(with_colors=True)
        lay.addWidget(self.x_list, 3)
        self._x_buttons = self._move_buttons(self.x_list, self._emit_order)
        lay.addLayout(self._x_buttons)

        self._s_label = QLabel("Colours (Color By)")
        self._s_label.setObjectName("Section")
        lay.addWidget(self._s_label)
        self.s_list = _GroupList(with_colors=True)
        lay.addWidget(self.s_list, 2)
        self._s_buttons = self._move_buttons(self.s_list, self._emit_series)
        lay.addLayout(self._s_buttons)

        row = QHBoxLayout()
        reset_c = QPushButton("Reset colours")
        reset_c.clicked.connect(self.colors_reset.emit)
        reset_o = QPushButton("Reset order")
        reset_o.clicked.connect(self.order_reset.emit)
        row.addWidget(reset_c)
        row.addWidget(reset_o)
        lay.addLayout(row)

        self.x_list.moved.connect(self._emit_order)
        self.s_list.moved.connect(self._emit_series)
        for lst in (self.x_list, self.s_list):
            lst.swatch_clicked.connect(self._pick_color)
            lst.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            lst.customContextMenuRequested.connect(
                lambda pos, l=lst: self._menu(l, pos))
        self._colors = {}
        self._series_mode = False

    # ------------------------------------------------------------------

    def _move_buttons(self, lst, emit):
        row = QHBoxLayout()
        row.setSpacing(4)
        for text, tip, fn in (("↑", "Move up", lambda: self._move(lst, -1, emit)),
                              ("↓", "Move down", lambda: self._move(lst, 1, emit)),
                              ("A→Z", "Sort by name", lambda: self._sort(lst, emit))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        return row

    def _move(self, lst, step, emit):
        """Move the selected row one place up (-1) or down (+1)."""
        r = lst.currentRow()
        # Nothing selected, or already at the end.
        if r < 0 or not 0 <= r + step < lst.count():
            return
        # takeItem detaches the item and hands over ownership, so it can
        # be reinserted; removeItemWidget would destroy it instead.
        it = lst.takeItem(r)
        lst.insertItem(r + step, it)
        lst.setCurrentRow(r + step)
        emit()

    def _sort(self, lst, emit):
        """Sort the list the way a person reads names, not the way a
        computer does."""
        import re
        names = lst.names()
        # Natural sort. re.split with a captured group keeps the digits,
        # so "LP10" becomes ["LP", 10, ""] and the number is compared as
        # a number — otherwise "LP10" sorts before "LP2", which is the
        # single most common complaint about ordered group lists.
        key = lambda n: [int(t) if t.isdigit() else t.lower()
                         for t in re.split(r"(\d+)", str(n))]
        self._fill(lst, sorted(names, key=key), lst.with_colors)
        emit()

    def _menu(self, lst, pos):
        it = lst.itemAt(pos)
        if it is None:
            return
        name = it.data(_NAME)
        m = QMenu(self)
        m.addAction("Change colour…", lambda: self._pick_color(name))
        # Only offered when a custom colour was set: the empty string is
        # the signal's way of saying "go back to the palette".
        if name in self._colors:
            m.addAction("Default colour", lambda: self.color_changed.emit(name, ""))
        m.addSeparator()
        # Which signal to fire depends on which list was right-clicked.
        emit = self._emit_series if lst is self.s_list else self._emit_order
        m.addAction("Move to start", lambda: self._to_end(lst, it, 0, emit))
        m.addAction("Move to end", lambda: self._to_end(lst, it, lst.count() - 1, emit))
        m.exec(lst.mapToGlobal(pos))

    def _to_end(self, lst, it, row, emit):
        lst.insertItem(row, lst.takeItem(lst.row(it)))
        emit()

    def _pick_color(self, name):
        """Open the colour picker for one group."""
        # Start from the colour currently drawn, not from the custom one:
        # if the group has no custom colour, the dialog should open on
        # the palette colour it actually has on screen.
        current = QColor(self._shown.get(name, "#2563EB"))
        c = QColorDialog.getColor(current, self, f"Colour for '{name}'")
        if c.isValid():
            self.color_changed.emit(name, c.name())

    def _emit_order(self):
        self.order_changed.emit(self.x_list.names())

    def _emit_series(self):
        self.series_order_changed.emit(self.s_list.names())

    def _fill(self, lst, names, with_colors):
        """Rebuild one list from *names*, keeping the selection.

        Remembering the selected name across the rebuild is what stops
        the highlight jumping to the top of the list every time a colour
        is changed or the groups are re-read.
        """
        lst.filling = True          # see the rowsMoved guard above
        cur = lst.currentItem().data(_NAME) if lst.currentItem() else None
        lst.clear()
        lst.with_colors = with_colors
        for n in names:
            it = QListWidgetItem(str(n))
            it.setData(_NAME, str(n))
            if with_colors and str(n) in self._shown:
                it.setIcon(swatch_icon(self._shown[str(n)]))
            it.setToolTip("Drag to move" + (" · click the square to recolour"
                                            if with_colors else ""))
            lst.addItem(it)
            if str(n) == cur:
                lst.setCurrentItem(it)
        lst.filling = False

    # ------------------------------------------------------------------

    def set_groups(self, x_groups, series=None, shown_colors=None,
                   custom_colors=None):
        """*x_groups*: X order; *series*: Color-By levels (None = colours
        belong to the X groups); *shown_colors*: {name: colour drawn}."""
        # Keys forced to str: group names arrive from a DataFrame column
        # and may be numbers or numpy scalars, which would then not match
        # the strings stored in the _NAME role.
        self._shown = {str(k): v for k, v in (shown_colors or {}).items()}
        self._colors = dict(custom_colors or {})
        has_series = bool(series)
        # Colours follow the series when there is one — see the note at
        # the top of the file.
        self._fill(self.x_list, x_groups, with_colors=not has_series)
        self._fill(self.s_list, series or [], with_colors=True)
        for w in (self._s_label, self.s_list):
            w.setVisible(has_series)
        # A QLayout cannot be hidden, only the widgets inside it, so the
        # series buttons are hidden one at a time. itemAt().widget() is
        # None for the stretch, hence the check.
        for i in range(self._s_buttons.count()):
            w = self._s_buttons.itemAt(i).widget()
            if w:
                w.setVisible(has_series)
        self._x_label.setText("X axis" if has_series else "X axis — order & colour")
