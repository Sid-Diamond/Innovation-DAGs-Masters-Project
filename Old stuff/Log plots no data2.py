import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from scipy.special import gamma as gamma_func, betaln

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

plot_config = {'figsize': (9, 7), 'dpi': 300, 'k_max': 200, 'discretisations': 10**5, 'xlim_left': 0.01, 'xlim_right': 400, 'diamond_linewidth': 1, 'sterling_linewidth': 0.5, 'ylim_bottom': None, 'ylim_top': 10}

def plot_theoretical_from_metadata(run_dir, plot_cfg=plot_config):
    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json", 'r') as f:
        meta = json.load(f)
    g_a = meta['g_a_empirical']
    g_b = meta['g_b_empirical']
    h = meta['network']['h']
    f_a = meta['network']['f_a']
    m_edges = meta['network']['m_edges']
    mu_a = meta['network']['mu_a']
    mu_b = meta['network']['mu_b']
    lambda_a = h * f_a + (1 - f_a) * (1 - h)
    lambda_b = h * (1 - f_a) + f_a * (1 - h)
    for idx, (node_type, color, mu_x, lambda_x) in enumerate([('a', 'red', mu_a, lambda_a), ('b', 'blue', mu_b, lambda_b)]):
        fig, ax = plt.subplots(1, 1, figsize=plot_cfg['figsize'], dpi=plot_cfg['dpi'])
        Z_factor = g_a * lambda_a + g_b * lambda_b + f_a * mu_a + (1 - f_a) * mu_b
        Z_tilde = m_edges / Z_factor
        alpha = mu_x / lambda_x
        gamma = 1 + 1 / (Z_tilde * lambda_x)
        p0 = 1 / (1 + mu_x * Z_tilde)
        k_range = np.concatenate([[0], np.linspace(0.01, plot_cfg['k_max'], plot_cfg['discretisations'])])
        diamond_pmf = np.zeros_like(k_range)
        diamond_pmf[0] = p0
        k_pos = k_range[1:]
        diamond_pmf[1:] = p0 * np.exp(betaln(k_pos, alpha + gamma) - betaln(k_pos, alpha))
        A = p0 * gamma_func(alpha + gamma) / gamma_func(alpha)
        sterling_pmf = A * (k_range + alpha) ** (-gamma)
        mask_d = diamond_pmf > 0
        mask_s = sterling_pmf > 0
        ax.plot(k_range[mask_d], diamond_pmf[mask_d], linestyle='-', linewidth=plot_cfg['diamond_linewidth'], color=color, label=rf'$p_{{{node_type},\infty}}(k)$', zorder=2)
        ax.plot(k_range[mask_s], sterling_pmf[mask_s], linestyle='--', linewidth=plot_cfg['sterling_linewidth'], color='dark' + color, label="Power Law Approximation", zorder=2)
        ax.set_xscale('log', base=10)
        ax.set_yscale('log')
        ax.set_xlim(left=plot_cfg['xlim_left'], right=plot_cfg['xlim_right'])
        ax.set_ylim(bottom=plot_cfg['ylim_bottom'], top=plot_cfg['ylim_top'])
        ax.set_xlabel(r'Citation count', fontsize=10)
        ax.set_ylabel(r'Probability', fontsize=10)
        ax.set_title(rf'Type "{node_type}" Theoretical Distribution', fontsize=10, fontweight='normal')
        from matplotlib.ticker import LogLocator
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=50))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=50))
        ax.grid(True, alpha=0.4, which='both', linestyle='--', linewidth=0.6)
        leg = ax.legend(fontsize=12, loc='best', framealpha=0.9)
        for txt in leg.get_texts():
            txt.set_fontfamily('serif')
        fig.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.2)
        plt.show()

plot_theoretical_from_metadata(r"C:\Users\sidne\OneDrive - Imperial College London\Masters\Msci Project\runs\run_20260410_110326")