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

data_dir = Path("runs/run_20260319_134959/data")
save_as_pdf = True
save_as_png = True
dpi = 200
combined_plot = True

csv_a = data_dir / "degree_dist_log_binned_type_a.csv"
csv_b = data_dir / "degree_dist_log_binned_type_b.csv"

def read_csv_ignore_comments(path):
    with open(path, "r") as f:
        lines = [ln for ln in f.readlines() if not ln.lstrip().startswith("#")]
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)))

def plot_theory(ax, csv_path, color1, color2, title_prefix, var_name):
    df = read_csv_ignore_comments(csv_path)
    theory_cols = [c for c in df.columns if c not in ["k_bin_center", "empirical_prob"]]
    x = df["k_bin_center"].values
    valid_k = x[x > 0]
    k_continuous = np.logspace(np.log10(0.01), np.log10(valid_k.max()), 1000)
    for i, (th, color, linestyle) in enumerate([(theory_cols[0], color1, "-"), (theory_cols[1], color2, "--")]):
        y = df[th].values
        mask = np.isfinite(y) & (y > 0) & (x > 0)
        if mask.sum() > 1:
            f = interp1d(x[mask], y[mask], kind='cubic', fill_value='extrapolate', bounds_error=False)
            y_continuous = f(k_continuous)
            mask_pos = y_continuous > 0
            label = f"Type {var_name}: $p_{{{var_name},\infty}}(k)$" if i == 0 else f"Type {var_name}: Power Law"
            ax.plot(k_continuous[mask_pos], y_continuous[mask_pos], linestyle=linestyle, linewidth=1, color=color, label=label, zorder=2)

def format_plot(ax, xlim_right):
    ax.set_xscale("symlog", linthresh=0.1, base=10)
    ax.set_yscale("log")
    ax.set_xlim(left=0.01, right=xlim_right)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.set_ylabel(r'Probability', fontsize=15)
    ax.set_xlabel(r'Citation count', fontsize=15, labelpad=0)
    ax.grid(True, alpha=0.4, which="both", linestyle="--", linewidth=0.6)
    leg = ax.legend(fontsize=11, loc="best", framealpha=0.9)
    for txt in leg.get_texts():
        txt.set_fontfamily("serif")

if combined_plot:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=dpi)
    plot_theory(ax, csv_a, "red", "darkred", "Type A", "x")
    plot_theory(ax, csv_b, "blue", "darkblue", "Type B", "y")
    #ax.set_title(r'$p_{x,\infty}(k)$ vs Power Law for Citation Distribution in 28,005 Network', fontsize=15, fontweight="normal", pad=0)
    format_plot(ax, 140)
    fig.tight_layout(pad=4)
    if save_as_png:
        fig.savefig(data_dir.parent / "combined_theory.png", dpi=dpi, bbox_inches="tight")
    if save_as_pdf:
        fig.savefig(data_dir.parent / "combined_theory.pdf", bbox_inches="tight")
    plt.show()
else:
    for csv_path, color1, color2, title_prefix, var_name, xlim_right in [
        (csv_a, "red", "darkred", "Type A: ", "x", 500),
        (csv_b, "blue", "darkblue", "Type B: ", "y", 100)
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=dpi)
        plot_theory(ax, csv_path, color1, color2, title_prefix, var_name)
        ax.set_title(rf'{title_prefix}$p_{{{var_name},\infty}}(k)$ vs Power Law for Citation Distribution in 28,005 Network', fontsize=15, fontweight="normal", pad=0)
        format_plot(ax, xlim_right)
        fig.tight_layout(pad=4)
        out_base = data_dir.parent / f"degree_dist_log_binned_type_{var_name}_theory"
        if save_as_png:
            fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        if save_as_pdf:
            fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.show()