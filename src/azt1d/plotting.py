"""
Shared matplotlib styling, using the validated palette from the dataviz
design system (categorical hues pass CVD-safety checks in a fixed order;
status colors are reserved for glycemic state, never reused as series 4+).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Categorical (light mode), fixed order — do not re-sort per chart.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Status palette — reserved for glycemic range, never used as a generic series color.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
GLUCOSE_BAND_COLORS = {"hypo": STATUS["critical"], "in_range": STATUS["good"], "hyper": STATUS["warning"]}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )
