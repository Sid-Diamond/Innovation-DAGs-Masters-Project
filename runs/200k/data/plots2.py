import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _parse_log_header_n_and_type(csv_path):
    """
    Parse node_type and n from the header of degree_dist_log_binned_type_*.csv.

    Looks for a line like:
        # Node type: a, n=123,456
    """
    node_type = None
    n_val = None

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break
            if line.startswith("# Node type:"):
                m = re.search(r"Node type:\s*([ab]),\s*n=([\d,]+)", line)
                if m:
                    node_type = m.group(1)
                    n_val = int(m.group(2).replace(",", ""))
                break
    return node_type, n_val


def plot_degree_distributions_log_from_two_csvs(
    csv_a,
    csv_b,
    figsize=(15, 6),
    dpi=300
):
    """
    Recreate the log-binned degree distribution plots using two CSV files
    (for type a and type b), assuming they have the same layout as
    degree_dist_log_binned_type_*.csv produced by NetworkVis.plot_degree_distributions_log.
    """

    # Match font config you used in NetworkVis
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    y_max_fixed = 1.0  # top of y-axis = 10^0
    y_min_positive = None

    # ---------- FIRST PASS: get shared y_bottom from both CSVs ----------
    for path in [csv_a, csv_b]:
        path = Path(path)
        if not path.exists():
            continue
        df_tmp = pd.read_csv(path, comment="#")
        probs_tmp = df_tmp["empirical_prob"].values
        positive = probs_tmp[probs_tmp > 0]
        if positive.size > 0:
            local_min = positive.min()
            y_min_positive = local_min if y_min_positive is None else min(y_min_positive, local_min)

    if y_min_positive is not None:
        y_bottom = y_min_positive * 0.8
    else:
        y_bottom = 1e-6

    # ---------- SECOND PASS: plot each type ----------
    def _plot_one(csv_path, default_node_type, color, ax):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return

        node_type_header, n_from_header = _parse_log_header_n_and_type(csv_path)
        node_type = node_type_header or default_node_type

        # Load numeric part (skip header comments)
        df = pd.read_csv(csv_path, comment="#")

        bin_centers = df["k_bin_center"].values
        probs = df["empirical_prob"].values

        # Theory columns are *_prob except empirical_prob
        theory_cols = [
            c for c in df.columns
            if c.endswith("_prob") and c != "empirical_prob"
        ]

        # Empirical scatter (match your style)
        ax.scatter(
            bin_centers,
            probs,
            s=50,
            alpha=0.7,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            label="Simulation",
            zorder=3,
        )

        # Theory curves
        for col in theory_cols:
            theo_vals = df[col].values
            theory_name = col[:-5]  # strip "_prob"

            base_style = dict(linestyle="-.", linewidth=2.0, alpha=0.85)

            # Diamond -> p_{a,∞}(k) or p_{b,∞}(k)
            if theory_name == "Diamond":
                label = rf"$p_{{{node_type},\infty}}(k)$"
            else:
                label = theory_name

            mask = theo_vals > 0
            ax.plot(
                bin_centers[mask],
                theo_vals[mask],
                color="dark" + color,
                zorder=2,
                label=label,
                **base_style,
            )

        # Axes: same as your NetworkVis.plot_degree_distributions_log
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("log")
        ax.set_xlim(left=-0.05)
        ax.set_ylim(bottom=y_bottom, top=y_max_fixed)

        ax.set_xlabel(r"Citation count", fontsize=13)
        ax.set_ylabel(r"Citation Count Probability", fontsize=13)

        # Title with n (using header n for exact match)
        if n_from_header is not None:
            n_val = n_from_header
        else:
            n_val = len(bin_centers)
        n_str = f"{n_val:,}"

        ax.set_title(
            rf'Type "{node_type}" paper citation distribution '
            rf"(n={n_str})",
            fontsize=13,
            fontweight="normal",
        )

        legend = ax.legend(fontsize=11, loc="best", framealpha=0.9)
        for text in legend.get_texts():
            text.set_fontfamily("serif")

        ax.grid(True, alpha=0.4, which="both", linestyle="--", linewidth=0.6)

    # Left: type a, Right: type b (same pattern as your linear CSV plotter)
    _plot_one(csv_a, "a", "red", axes[0])
    _plot_one(csv_b, "b", "blue", axes[1])

    fig.suptitle(
        r"Citation Count Distributions in a Directed Acyclic Homophilic Network",
        fontsize=20,
        fontweight="normal",
        y=0.94,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ----------------------------------------
# Example call (matches your working style)
# ----------------------------------------
if __name__ == "__main__":
    fig = plot_degree_distributions_log_from_two_csvs(
        csv_a="runs/200k/data/degree_dist_log_binned_type_a.csv",
        csv_b="runs/200k/data/degree_dist_log_binned_type_b.csv",
        figsize=(15, 6),
        dpi=300,
    )
    plt.show()