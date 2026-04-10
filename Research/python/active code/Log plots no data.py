import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib.ticker import LogLocator

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

data_dir = Path("runs/run_20260410_100911/data")
save_as_pdf = True
save_as_png = True
dpi = 200
negative_power_scale = 0.5

csv_a = data_dir / "degree_dist_log_binned_type_a.csv"
csv_b = data_dir / "degree_dist_log_binned_type_b.csv"  

def read_csv_ignore_comments(path):
    with open(path, "r") as f:
        lines = [ln for ln in f.readlines() if not ln.lstrip().startswith("#")]
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)))

def make_plot(csv_path, out_base, color1, color2, title_prefix, xlim_left=0.01, xlim_right=400):
    df = read_csv_ignore_comments(csv_path)
    theory_cols = [c for c in df.columns if c not in ["k_bin_center", "empirical_prob"]]
    if len(theory_cols) < 2:
        raise ValueError("Need at least 2 theory columns")
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=dpi)
    x = df["k_bin_center"].values
    valid_k = x[x > 0]
    if valid_k.size == 0:
        raise ValueError("No positive k values found")
    k_min, k_max = 0.01, valid_k.max()
    k_continuous = np.logspace(np.log10(k_min), np.log10(k_max), 1000)
    th1 = theory_cols[0]
    y1 = df[th1].values
    mask1 = np.isfinite(y1) & (y1 > 0) & (x > 0)
    if mask1.sum() > 1:
        f1 = interp1d(x[mask1], y1[mask1], kind='cubic', fill_value='extrapolate', bounds_error=False)
        y1_continuous = f1(k_continuous)
        mask_pos1 = y1_continuous > 0
        label1 = r"$p_{x,\infty}(k)$" if th1.lower().startswith("diamond") else th1
        ax.plot(k_continuous[mask_pos1], y1_continuous[mask_pos1], linestyle="-", linewidth=1, color=color1, label=label1, zorder=2)
    th2 = theory_cols[1]
    y2 = df[th2].values
    mask2 = np.isfinite(y2) & (y2 > 0) & (x > 0)
    if mask2.sum() > 1:
        f2 = interp1d(x[mask2], y2[mask2], kind='cubic', fill_value='extrapolate', bounds_error=False)
        y2_continuous = f2(k_continuous)
        mask_pos2 = y2_continuous > 0
        label2 = r"Power Law Approximation" if th2.lower().startswith("sterling") else th2
        ax.plot(k_continuous[mask_pos2], y2_continuous[mask_pos2], linestyle="--", linewidth=1, color=color2, label=label2, zorder=2)
    ax.set_xscale("symlog", linthresh=0.1, base=10)
    ax.set_yscale("log")
    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.set_ylabel(r'Probability', fontsize=13)
    ax.set_xlabel(r'Citation count', fontsize=13, labelpad=0)
    ax.grid(True, alpha=0.4, which="both", linestyle="--", linewidth=0.6)
    ax.set_title(rf'{title_prefix}$p_{{x,\infty}}(k)$ vs Power Law for Citation Distribution in 28,005 Network', fontsize=15, fontweight="normal", pad=0)
    leg = ax.legend(fontsize=12, loc="best", framealpha=0.9)
    for txt in leg.get_texts():
        txt.set_fontfamily("serif")
    fig.tight_layout(pad=4)
    if save_as_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_as_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()

out_base_a = data_dir.parent / "degree_dist_log_binned_type_a_theory"
make_plot(csv_path=csv_a, out_base=out_base_a, color1="red", color2="black", title_prefix="Type A: ", xlim_left=0.01, xlim_right=1000)
out_base_b = data_dir.parent / "degree_dist_log_binned_type_b_theory"
make_plot(csv_path=csv_b, out_base=out_base_b, color1="blue", color2="blue", title_prefix="Type B: ", xlim_left=0.01, xlim_right=400)
# --------- TYPE A (original colors) ----------
out_base_a = data_dir.parent / "degree_dist_log_binned_type_a_theory"
make_plot(
    csv_path=csv_a,
    out_base=out_base_a,
    color1="red",      # first theory curve
    color2="black",    # second theory curve
    title_prefix="Type A: "
)

# --------- TYPE B (blue counterpart) ----------
out_base_b = data_dir.parent / "degree_dist_log_binned_type_b_theory"
make_plot(
    csv_path=csv_b,
    out_base=out_base_b,
    color1="blue",     # first theory curve in blue
    color2="black",     # second theory curve also in blue (dashed)
    title_prefix="Type B: "
)