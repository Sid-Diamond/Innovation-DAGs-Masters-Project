import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# CONFIGURATION — EDIT THIS BLOCK ONLY
# ==================================================

FIG_CONFIG = {
    # Font
    "font_family": "Times New Roman",

    # Figure size
    "figsize": (10, 8),

    # Empirical marker settings
    "empirical_marker_size": 0.1,
    "empirical_marker_style": "s",   # "x", "o", "s", "^", "D",...
    "empirical_hollow": True,
    "empirical_color": "#1f77b4",
    "empirical_zorder": 3,

    # Diamond theory line (red)
    "diamond_linewidth": 0.5,
    "diamond_color": "#d62728",
    "diamond_zorder": 2,

    # Sterling theory line (black)
    "sterling_linewidth": 0.3,
    "sterling_color": "#000000",
    "sterling_zorder": 2,

    # Font sizes
    "title_size": 6,
    "axis_label_size": 6,
    "tick_label_size": 6,
    "legend_size": 6,

    # Tick styling
    "major_tick_length": 1,
    "minor_tick_length": 0.3,
    "tick_width": 0.1,

    # Grid
    "grid_alpha": 0.2,
    "grid_style": "--",

    # Legend
    "legend_frame": False,

    # Log scale
    "x_log": True,
    "y_log": True,
}

# ==================================================
# GLOBAL STYLE
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.family": FIG_CONFIG["font_family"],
    "axes.linewidth": 0.1,
})

LINE_MARKERS = {"x", "+", "1", "2", "3", "4", "|"}


# ==================================================
# CORE PLOT FUNCTION
# ==================================================

def _scatter_empirical(ax, k, p_emp):
    cfg = FIG_CONFIG
    marker = cfg["empirical_marker_style"]

    if marker in LINE_MARKERS:
        # Unfillable markers: simple color
        ax.scatter(
            k,
            p_emp,
            s=cfg["empirical_marker_size"],
            marker=marker,
            color=cfg["empirical_color"],
            zorder=cfg["empirical_zorder"],
            label="Empirical",
        )
    else:
        # Fill / hollow markers
        if cfg["empirical_hollow"]:
            facecolor, edgecolor = "none", cfg["empirical_color"]
        else:
            facecolor = edgecolor = cfg["empirical_color"]

        ax.scatter(
            k,
            p_emp,
            s=cfg["empirical_marker_size"],
            marker=marker,
            facecolors=facecolor,
            edgecolors=edgecolor,
            zorder=cfg["empirical_zorder"],
            label="Empirical",
        )


def plot_from_csv(csv_name: str, title: str = ""):
    cfg = FIG_CONFIG
    df = pd.read_csv(BASE_DIR / csv_name, comment="#", skip_blank_lines=True)

    k = df["k"].to_numpy()
    p_emp = df["empirical_pmf"].to_numpy()

    fig, ax = plt.subplots(figsize=cfg["figsize"])

    # --------------------------
    # Empirical data
    # --------------------------
    _scatter_empirical(ax, k, p_emp)

    # --------------------------
    # Red diamond theory line
    # --------------------------
    # Make sure your CSV has a column named "Diamond_pmf"
    # or rename this to whatever your red theory column is.
    if "Diamond_pmf" in df.columns:
        ax.plot(
            k,
            df["Diamond_pmf"].to_numpy(),
            linewidth=cfg["diamond_linewidth"],
            linestyle="-",
            color=cfg["diamond_color"],
            zorder=cfg["diamond_zorder"],
            label="Diamond theory",
        )

    # --------------------------
    # Sterling theory line
    # --------------------------
    if "Sterling_pmf" in df.columns:
        ax.plot(
            k,
            df["Sterling_pmf"].to_numpy(),
            linewidth=cfg["sterling_linewidth"],
            linestyle="--",
            color=cfg["sterling_color"],
            zorder=cfg["sterling_zorder"],
            label="Sterling's approximation",
        )

    # --------------------------
    # Axes scaling, labels, etc.
    # --------------------------
    if cfg["x_log"]:
        ax.set_xscale("log")
    if cfg["y_log"]:
        ax.set_yscale("log")

    ax.set_xlabel("Degree k", fontsize=cfg["axis_label_size"])
    ax.set_ylabel("P(k)", fontsize=cfg["axis_label_size"])
    ax.set_title(title, fontsize=cfg["title_size"], pad=20)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=cfg["tick_label_size"],
        length=cfg["major_tick_length"],
        width=cfg["tick_width"],
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=cfg["minor_tick_length"],
        width=cfg["tick_width"],
    )

    ax.grid(
        True,
        which="both",
        linestyle=cfg["grid_style"],
        alpha=cfg["grid_alpha"],
    )

    ax.legend(
        fontsize=cfg["legend_size"],
        frameon=cfg["legend_frame"],
    )

    fig.tight_layout()
    plt.show()


# ==================================================
# FIGURE DEFINITIONS
# ==================================================

def figure_type_a():
    plot_from_csv(
        csv_name="degree_dist_linear_type_a.csv",
        title="Type A Degree Distribution",
    )


if __name__ == "__main__":
    figure_type_a()