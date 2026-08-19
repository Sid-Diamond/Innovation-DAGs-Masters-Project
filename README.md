[README.md](https://github.com/user-attachments/files/31222405/README.md)
# Homophilic Network Growth Simulation

This directory contains the main simulation and analysis script used in the project:

`Homophilic CA Simulation and Analysis.py`

The script studies network growth driven by two mechanisms:

- **Cumulative advantage:** nodes that already have many incoming links are more likely to receive new links.
- **Homophily:** links are more likely to attach to nodes with similar characteristics.

The model is designed for networks such as scientific citation networks and social networks. It combines simulation, asymptotic theory, degree-distribution analysis, and a finite-size goodness-of-fit workflow for discrete network data.

## Project Context

The associated research develops a mathematical model for how links are distributed as a network grows. It derives an asymptotic expression for the expected number of links received by a network member, tests the theory on finite simulations, and examines how validity depends on the initial network structure.

The analysis is particularly concerned with whether sparse and dense initial networks remain well described as they grow, and whether finite-size deviations are better understood in terms of the fraction of nodes added or the fraction of links added.

## What The Script Does

The main script:

1. Generates a directed network with two node types, `a` and `b`.
2. Assigns node types according to `f_a` and `1 - f_a`.
3. Adds nodes sequentially using type-dependent preferential attachment.
4. Computes asymptotic quantities including `g_a`, `g_b`, `Z_factor`, and `Z_tilde`.
5. Compares simulated degree distributions with theoretical forms.
6. Runs optional finite-size goodness-of-fit sweeps.
7. Prints network statistics and theoretical consistency checks.
8. Optionally saves figures, CSV data, and run metadata.

## Requirements

Use Python 3.14 or a compatible recent Python 3 release. Install the required packages with:

```powershell
python -m pip install numpy networkx matplotlib scipy pandas
```

The script uses Matplotlib's non-interactive `Agg` backend, so it can run from a terminal without opening plot windows.

## Running The Script

From the repository root:

```powershell
python ".\active code\Homophilic CA Simulation and Analysis.py"
```

If the Python launcher is not on `PATH`, use the full path to the selected interpreter:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" ".\active code\Homophilic CA Simulation and Analysis.py"
```

The script currently uses the values in the `config` dictionary near the bottom of the file. It does not currently accept command-line arguments.

## Configuration

Edit the top-level `config` dictionary to change a run. The main sections are:

### `theory`

Controls the theoretical models used for fitting and goodness-of-fit analysis.

```python
theory=dict(
    network_basic=['Diamond', 'Sterling'],
    gof='Diamond',
    mle='Diamond',
)
```

- `network_basic`: theory curves included in the basic degree-distribution analysis.
- `gof`: theory used by the goodness-of-fit calculations.
- `mle`: theory used by the maximum-likelihood fitting step.

The script contains support for the `Diamond`, `Sterling`, and `power_law_2` theory paths, although the active default configuration uses `Diamond` and `Sterling` for the basic plots.

### `display`

Controls whether results are written to disk:

```python
display=dict(save_outputs=True),
```

Use `True` to save results. Use `False` to run the calculations without creating a `runs` directory, figures, CSV files, or metadata:

```python
display=dict(save_outputs=False),
```

When saving is disabled, the script prints:

```text
No data saved. To save, toggle save_outputs to True.
```

### `plots`

Enables or disables groups of analyses:

```python
plots=dict(
    network_basic=True,
    sweep_m_edges_csn=False,
    sweep_n0_csn=False,
    csn_p_vs_b_m=5,
    csn_p_vs_b_n0=5,
    grid_2d_sweep=False,
)
```

- `network_basic`: degree distributions, network visualisation, asymptotes, and normalisation plots.
- `sweep_m_edges_csn`: one-dimensional sweeps over the number of links added per node.
- `sweep_n0_csn`: one-dimensional sweeps over the initial network size.
- `grid_2d_sweep`: two-dimensional sweeps over initial network size and links per node.
- `csn_p_vs_b_m` and `csn_p_vs_b_n0`: number of diagnostic `p`-value plots selected for the corresponding one-dimensional sweep.

### `network`

Defines the simulated network:

```python
network=dict(
    n0=5,
    n_nodes=28000,
    m_edges=2,
    h=0.7,
    f_a=0.4,
    mu_a=2,
    mu_b=1,
    seed=5,
    power_law_params=None,
)
```

- `n0`: number of nodes in the initial core.
- `n_nodes`: number of nodes added during growth.
- `m_edges`: number of incoming links selected by each new node.
- `h`: homophily parameter.
- `f_a`: expected fraction of type `a` nodes.
- `mu_a`, `mu_b`: type-specific baseline attachment terms.
- `seed`: NumPy random seed for reproducible simulations.
- `power_law_params`: optional manually supplied power-law parameters.

The initial core must be large enough to support the requested number of distinct targets: in particular, choose `n0 > m_edges`.

### Sweep sections

`sweep_m`, `sweep_n0`, and `grid_2d` contain the parameter ranges and statistical settings for the optional finite-size analyses. Their most important controls are:

- minimum, maximum, and step values for the swept parameter;
- `node_type` and lower cutoff `a`;
- critical values in `p_c_list`;
- number of simulations in `N_sims`;
- the grid used for the upper cutoff `b`.

These analyses can be computationally expensive, particularly with large `N_sims` or a two-dimensional grid.

## Output Structure

With `save_outputs=True`, each run creates a timestamped directory:

```text
runs/
└── run_YYYYMMDD_HHMMSS/
    ├── metadata.json
    ├── *.png
    └── data/
        └── *.csv
```

`metadata.json` stores the configuration and is updated with the total runtime when the script finishes.

Typical figures include:

- log-binned degree distributions;
- discrete linear-scale degree distributions;
- network graph layouts;
- asymptotic in-degree plots;
- normalisation plots;
- one-dimensional goodness-of-fit sweep plots;
- two-dimensional contour plots.

Typical CSV files include degree-distribution data, node and edge data, network summary statistics, asymptotic evolution data, normalisation values, and sweep grids.

The exact set of files depends on the switches in `config["plots"]`.

## Reproducibility

Set a fixed integer `seed` in `config["network"]` to reproduce the random network construction. Results can still vary if the configuration, Python version, or dependency versions change. For publication-quality comparisons, record the generated `metadata.json` alongside the figures and CSV files.

## Interpreting The Terminal Output

The script reports:

- network generation time;
- empirical/theoretical `Z` agreement;
- comparison between theoretical and empirical `g_b`;
- node counts and type-specific in-degree statistics;
- `g_a + g_b` compared with `m_edges`;
- the theory used for goodness-of-fit and maximum-likelihood analysis;
- total runtime when outputs are saved.

Agreement close to one in the ratio checks indicates consistency between the simulated network and the corresponding asymptotic quantities. These checks do not, by themselves, establish that the full finite network follows the asymptotic degree distribution; use the goodness-of-fit analyses for that purpose.

## Code Structure

The main abstractions in the script are:

- `FileManager`: creates run directories and manages figures, CSV files, and metadata.
- `DirectedHomophilicNetwork`: constructs the network and computes theoretical quantities.
- `DirectedHomophilicNetwork.NetworkVis`: creates network and degree-distribution visualisations.
- `GoFDiagnostics`: implements discrete goodness-of-fit, maximum-likelihood, and sweep calculations.
- `NetworkStatistics`: prints consistency checks and summary statistics.

The script is currently organised as a single executable research file. The `config` dictionary is the intended user interface for changing experiments.

## Troubleshooting

**`ModuleNotFoundError` for NumPy, SciPy, or another package**

Install packages through the same interpreter used to run the script:

```powershell
python -m pip install numpy networkx matplotlib scipy pandas
```

Check the interpreter and package location with:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip show numpy
```

**The script uses an old Visual Studio Python installation**

Do not run it with a hard-coded path such as `Python37_64\python.exe`. Select the current Python interpreter in VS Code or run it with the normal `python` command after opening a new terminal.

**No files appear after a run**

Check that the configuration contains:

```python
display=dict(save_outputs=True),
```

Also check that the relevant entries in `config["plots"]` are enabled.

## Research Scope

This README documents the current main simulation and analysis script. The wider repository contains earlier experiments, plotting scripts, NetLogo models, literature notes, and archived run data. Those materials provide research history and context, while this directory is the most direct starting point for reproducing the current computational results.
