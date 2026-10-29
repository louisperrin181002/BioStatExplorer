# src/theme.py
"""Light and dark styles for the whole window.

The look itself lives in ``src/labstyle.py``, which is shared with FlowPy
and the Expression Calculator so the three apps match. Edit the canonical
copy at ``_shared/labstyle.py`` and run ``_shared/sync_style.sh``; this
module only re-exports it under the names this app already used.
"""

from src.labstyle import (                                      # noqa: F401
    LIGHT, DARK, stylesheet, plot_colors, tokens,
    PALETTES, PALETTE_NAMES, palette_colors,
    DPI_CHOICES, DEFAULT_DPI, FONT, MONO,
)
