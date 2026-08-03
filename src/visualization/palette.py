"""
Shared Publication Color Palette
------------------------------------
Validated colorblind-safe categorical palette and single-hue sequential
ramp (see the project's dataviz skill: fixed hue order, never cycled;
sequential = one hue light->dark; diverging = two hues + neutral midpoint;
never a rainbow colormap). All `src/visualization/*.py` modules import
from here rather than each picking their own ad-hoc colors, so every
figure in the paper reads as one consistent visual system.
"""

from matplotlib.colors import LinearSegmentedColormap

# Fixed categorical order -- assign by entity identity (e.g. one color per
# experiment/model), NEVER re-cycled or reassigned when a filter changes
# which entities are shown.
CATEGORICAL_PALETTE = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

# Single-hue sequential ramp (blue, light -> dark) for magnitude/heatmap encoding.
SEQUENTIAL_BLUE_STEPS = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)
SEQUENTIAL_BLUE_CMAP = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_STEPS)

# Diverging pair (blue <-> red) with a neutral gray midpoint, for signed
# deltas (e.g. resource-contribution gains/losses relative to baseline).
DIVERGING_STEPS = ("#184f95", "#6da7ec", "#f0efec", "#eb99a0", "#e34948")
DIVERGING_CMAP = LinearSegmentedColormap.from_list("diverging_blue_red", DIVERGING_STEPS)

# Chart chrome (light-mode ink/gridlines), matplotlib rcParam-compatible hex.
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
CHART_SURFACE = "#fcfcfb"


def apply_publication_style() -> None:
    """Apply a consistent matplotlib style for all publication figures.

    Sets sans-serif font, recessive gridlines/spines, and a neutral chart
    surface -- call once at the top of any script/notebook producing
    figures for `paper/figures/`.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.edgecolor": BASELINE_AXIS,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "figure.facecolor": CHART_SURFACE,
            "axes.facecolor": CHART_SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def get_categorical_color(index: int) -> str:
    """Return a fixed-order categorical color by index, wrapping if needed.

    Args:
        index: Zero-based entity index (e.g. experiment number E0=0, E1=1, ...).

    Returns:
        str: Hex color. Beyond 8 entities, colors repeat -- per the
            palette's non-negotiable, a 9th+ series should really fold
            into "Other"/facets rather than reuse a hue, but repeating is
            a safer fallback than crashing.
    """
    return CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)]
