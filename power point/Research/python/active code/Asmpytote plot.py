import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

data_dir = Path("runs/run_20260318_202157/data")
save_as_pdf = True
save_as_png = True
figsize = (10, 6)
dpi = 200
tail_fraction = 0.05  # last 5% of points

csv_path = data_dir / "asymptotes.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"asymptotes.csv not found in {data_dir}")

with open(csv_path, "r") as f:
    lines = [ln for ln in f.readlines() if not ln.lstrip().startswith("#")]
from io import StringIO
df = pd.read_csv(StringIO("".join(lines)))

for col in ["t","mean_deg_a","mean_deg_b","asymptote_a","asymptote_b"]:
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}' in asymptotes.csv")

t = df["t"].values
m_a = df["mean_deg_a"].values
m_b = df["mean_deg_b"].values

def tail_stats(values, frac, label):
    n = len(values)
    print(f"\n--- {label} ---")
    print(f"Total number of points n = {n}")
    if n == 0:
        print("No points; returning NaN.")
        return np.nan, np.nan, None
    n_tail = max(1, int(n * frac))
    print(f"tail_fraction = {frac}, n_tail = max(1, int({n} * {frac})) = {n_tail}")
    tail = values[-n_tail:]
    print(f"Tail values (last {n_tail} points):")
    for i, v in enumerate(tail, 1):
        print(f"  tail[{i}] = {v:.12f}")
    mean = tail.mean()
    if n_tail > 1:
        sigma = tail.std(ddof=1)
        se = sigma / np.sqrt(n_tail)
    else:
        sigma = 0.0
        se = 0.0
    print(f"Tail mean = {mean:.12f}")
    print(f"Tail sample std (ddof=1) = {sigma:.12f}")
    print(f"Standard error (sigma / sqrt(n_tail)) = {se:.12f}")
    start_idx = n - n_tail
    return mean, se, start_idx

g_a, se_a, start_idx_a = tail_stats(m_a, tail_fraction, "Type a")
g_b, se_b, start_idx_b = tail_stats(m_b, tail_fraction, "Type b")

start_idx = start_idx_a if start_idx_a is not None else start_idx_b
t_start_tail = t[start_idx] if start_idx is not None else None
t_end = t[-1]

fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

ax.plot(t, m_a, color="red", linewidth=1, label="Type 'x' mean in-degree")
ax.plot(t, m_b, color="blue", linewidth=1, label="Type 'y' mean in-degree")

ax.axhline(
    g_a,
    color="darkred",
    linestyle="--",
    linewidth=1,
    label=fr"$\langle g_x \rangle$ Numerically Esitmated = {g_a:.3f}"
)
ax.axhline(
    g_b,
    color="darkblue",
    linestyle="--",
    linewidth=1,
    label=fr"$\langle g_y \rangle$ Numerically Esitmated = {g_b:.3f}"
)

if t_start_tail is not None:
    ax.axvline(t_start_tail, color="grey", linestyle=":", linewidth=1)
    ax.axvspan(t_start_tail, t_end, color="grey", alpha=0.08)

ax.set_xlabel("t (number of nodes)", fontsize=15)
ax.set_ylabel("Mean in-degree", fontsize=15)
ax.grid(True, linestyle="--", alpha=0.3)

leg = ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
for txt in leg.get_texts():
    txt.set_fontfamily("serif")

# --- Tail region text (rotated 90°) ---------------------------------
if t_start_tail is not None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    vertical_offset = -0.4 * (ymax - ymin)   # up/down
    y_text = ymax + vertical_offset

    horizontal_offset = 0.02 * (xmax - xmin)  # left/right
    x_text = t_start_tail + horizontal_offset

    ax.text(
        x_text,
        y_text,
        "tail region",
        rotation=90,
        va="top",
        ha="center",
        fontsize=12,
        color="dimgrey"
    )
# --------------------------------------------------------------------

ax.set_title("Asymptotic in-edge density evolution", fontsize=14, fontweight="normal")
fig.tight_layout(pad=3)

out_base = data_dir.parent / "asymptotes_from_csv_tail_marked"
if save_as_png:
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
if save_as_pdf:
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")

plt.show()