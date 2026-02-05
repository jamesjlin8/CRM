#!/usr/bin/env python3
"""
Plot FlowCalc-generated scattering pattern data files.
Usage: python plotting.py <data_file> [--smear] [--both] [--save]

Parameters are extracted from the filename (St, cyl, r, l) or set in CONFIG below.
"""

import argparse
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Physics parameters (not extractable from filename)
    'background': 0.0,
    'phi0': 0.0,
    'beta': 3,              # Structure factor parameter
    'scalvolfrac': 8.577,   # Scaling volume fraction factor
    
    # Plot appearance
    'graphingparameter': 4, # Marker size scaling factor
    'fontsize': 12,         # Font size for labels
    'caxis_min': None,      # Color axis minimum (log scale), None = auto
    'caxis_max': None,      # Color axis maximum (log scale), None = auto
    'fig_width': 6,         # Figure width in inches
    'fig_height': 5,        # Figure height in inches
    
    # Smearing parameters (only used if smear=True)
    'smear_sigma_rel': 0.15,
    'smear_n_sigma': 3.0,
    'smear_min_sigma': 1e-12,
    
    # Save parameters
    'save_dpi': 300,        # DPI for saved figures
}
# =============================================================================


def parse_filename(filename):
    """Extract parameters from filename.
    
    Expected patterns in filename:
        St{value} - stretch value
        {value}cyl - number of cylinders  
        {value}r - radius in Angstroms
        {value}l - length in Angstroms
    
    Returns dict with extracted values (None if not found).
    """
    stem = Path(filename).stem
    
    params = {
        'stretch': None,
        'n_cyl': None,
        'radius': None,
        'length': None,
    }
    
    # Extract stretch: St followed by number (int or float)
    match = re.search(r'St([\d.]+)', stem)
    if match:
        params['stretch'] = float(match.group(1))
    
    # Extract n_cyl: number followed by 'cyl'
    match = re.search(r'(\d+)cyl', stem)
    if match:
        params['n_cyl'] = int(match.group(1))
    
    # Extract radius: number followed by 'r' (but not preceded by letters)
    match = re.search(r'(?<![a-zA-Z])([\d.]+)r(?![a-zA-Z])', stem)
    if match:
        params['radius'] = float(match.group(1))
    
    # Extract length: number followed by 'l' (at end or before underscore/non-letter)
    match = re.search(r'([\d.]+)l(?:_|$|[^a-zA-Z])', stem)
    if match:
        params['length'] = float(match.group(1))
    
    return params


def _build_grid_from_scatter(qx, qy, values):
    x_vals = np.unique(qx)
    y_vals = np.unique(qy)
    x_index = {value: idx for idx, value in enumerate(x_vals)}
    y_index = {value: idx for idx, value in enumerate(y_vals)}

    grid = np.full((y_vals.size, x_vals.size), np.nan, dtype=float)
    counts = np.zeros_like(grid)
    for x_val, y_val, val in zip(qx, qy, values):
        iy = y_index[y_val]
        ix = x_index[x_val]
        if np.isnan(grid[iy, ix]):
            grid[iy, ix] = val
        else:
            grid[iy, ix] += val
        counts[iy, ix] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    return x_vals, y_vals, grid


def _build_binned_grid(qx, qy, values, bin_size):
    x_min, x_max = float(qx.min()), float(qx.max())
    y_min, y_max = float(qy.min()), float(qy.max())

    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    if x_edges.size < 2:
        x_edges = np.array([x_min - bin_size, x_max + bin_size])
    if y_edges.size < 2:
        y_edges = np.array([y_min - bin_size, y_max + bin_size])

    x_vals = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_vals = (y_edges[:-1] + y_edges[1:]) / 2.0

    x_idx = np.digitize(qx, x_edges) - 1
    y_idx = np.digitize(qy, y_edges) - 1
    x_idx = np.clip(x_idx, 0, x_vals.size - 1)
    y_idx = np.clip(y_idx, 0, y_vals.size - 1)

    grid = np.full((y_vals.size, x_vals.size), np.nan, dtype=float)
    counts = np.zeros_like(grid)
    for ix, iy, val in zip(x_idx, y_idx, values):
        if np.isnan(grid[iy, ix]):
            grid[iy, ix] = val
        else:
            grid[iy, ix] += val
        counts[iy, ix] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    return x_vals, y_vals, grid, x_edges, y_edges


def _scatter_from_grid(qx, qy, x_vals, y_vals, grid):
    x_index = {value: idx for idx, value in enumerate(x_vals)}
    y_index = {value: idx for idx, value in enumerate(y_vals)}

    out = np.empty_like(qx, dtype=float)
    for idx, (x_val, y_val) in enumerate(zip(qx, qy)):
        out[idx] = grid[y_index[y_val], x_index[x_val]]

    return out


def _scatter_from_binned_grid(qx, qy, x_edges, y_edges, grid):
    x_idx = np.digitize(qx, x_edges) - 1
    y_idx = np.digitize(qy, y_edges) - 1
    x_idx = np.clip(x_idx, 0, grid.shape[1] - 1)
    y_idx = np.clip(y_idx, 0, grid.shape[0] - 1)

    out = np.empty_like(qx, dtype=float)
    for idx, (ix, iy) in enumerate(zip(x_idx, y_idx)):
        out[idx] = grid[iy, ix]

    return out


def smear_intensity(qx, qy, intensity, sigma_rel=0.15, n_sigma=3.0, min_sigma=1e-12,
                    grid_max_factor=5.0, grid_max_points=1_000_000,
                    bin_max_bins=256, target_sigma_bins=12):
    """Apply q-dependent 2D Gaussian resolution smearing."""
    x_unique = np.unique(qx)
    y_unique = np.unique(qy)
    grid_points = x_unique.size * y_unique.size
    max_grid_points = max(grid_max_points, int(grid_max_factor * len(qx)))

    use_direct_grid = grid_points <= max_grid_points
    if use_direct_grid:
        x_vals, y_vals, grid = _build_grid_from_scatter(qx, qy, intensity)
        x_edges = y_edges = None
    else:
        q_max = float(np.max(np.hypot(qx, qy)))
        x_range = float(qx.max() - qx.min())
        y_range = float(qy.max() - qy.min())
        range_max = max(x_range, y_range)

        bin_size = max(min_sigma, (sigma_rel * q_max) / target_sigma_bins)
        bin_size = max(bin_size, range_max / bin_max_bins)
        x_vals, y_vals, grid, x_edges, y_edges = _build_binned_grid(
            qx, qy, intensity, bin_size
        )

    if x_vals.size < 2 or y_vals.size < 2:
        return intensity.copy()

    dx = float(np.median(np.diff(x_vals)))
    dy = float(np.median(np.diff(y_vals)))
    dx = abs(dx) if dx != 0 else 1.0
    dy = abs(dy) if dy != 0 else 1.0

    out_grid = np.full_like(grid, np.nan, dtype=float)
    for iy, y0 in enumerate(y_vals):
        for ix, x0 in enumerate(x_vals):
            q_val = float(np.hypot(x0, y0))
            sigma = max(min_sigma, sigma_rel * q_val)

            if sigma == min_sigma:
                out_grid[iy, ix] = grid[iy, ix]
                continue

            half_x = max(1, int(np.ceil(n_sigma * sigma / dx)))
            half_y = max(1, int(np.ceil(n_sigma * sigma / dy)))

            x_start = max(0, ix - half_x)
            x_stop = min(x_vals.size, ix + half_x + 1)
            y_start = max(0, iy - half_y)
            y_stop = min(y_vals.size, iy + half_y + 1)

            x_slice = x_vals[x_start:x_stop]
            y_slice = y_vals[y_start:y_stop]
            dx2 = (x_slice - x0) ** 2
            dy2 = (y_slice - y0) ** 2

            dist2 = dy2[:, None] + dx2[None, :]
            weights = np.exp(-0.5 * dist2 / (sigma ** 2))
            subgrid = grid[y_start:y_stop, x_start:x_stop]

            valid = np.isfinite(subgrid)
            if not np.any(valid):
                out_grid[iy, ix] = np.nan
                continue

            weights *= valid
            weight_sum = weights.sum()
            if weight_sum <= 0:
                out_grid[iy, ix] = np.nan
                continue

            out_grid[iy, ix] = np.nansum(subgrid * weights) / weight_sum

    if use_direct_grid:
        return _scatter_from_grid(qx, qy, x_vals, y_vals, out_grid)
    return _scatter_from_binned_grid(qx, qy, x_edges, y_edges, out_grid)


def calculate_intensity(qx, qy, p, params):
    """Calculate intensity with and without structure factor."""
    pref = params['scalvolfrac'] * np.pi * params['radius']**2 * \
           params['length'] * params['n_cyl'] * 1e-5
    
    q = np.sqrt(qx**2 + qy**2)
    inosq = pref * p + params['background']
    s = 1.0 / (1.0 + params['beta'] * p)
    iwithsq = pref * s * p + params['background']
    
    return inosq, iwithsq, s, q


def plot_scattering_pattern(qx, qy, intensity, params, title="", use_log=True, 
                            output_path=None):
    """Create a 2D scatter plot of scattering pattern and display it.
    
    Parameters
    ----------
    qx, qy : array-like
        q-vector components
    intensity : array-like
        Intensity values to plot
    params : dict
        Plotting parameters
    title : str, optional
        Plot title
    use_log : bool
        Use log10 scale for intensity
    output_path : str or Path, optional
        If provided, save figure to this path at save_dpi
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    if use_log:
        plot_values = np.log10(intensity)
        plot_values = np.where(np.isfinite(plot_values), plot_values, np.nan)
    else:
        plot_values = intensity
    
    fig, ax = plt.subplots(figsize=(params['fig_width'], params['fig_height']))
    
    n_points = len(qx)
    marker_size = params['graphingparameter'] * 150000 / n_points
    
    scatter = ax.scatter(qy, qx, s=marker_size, c=plot_values, 
                        cmap='jet', marker='s', edgecolors='none')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('I(q)', fontsize=params['fontsize'])
    
    if use_log:
        # Auto-scale color axis if limits are None
        if params.get('caxis_min') is None or params.get('caxis_max') is None:
            valid_values = plot_values[np.isfinite(plot_values)]
            if len(valid_values) > 0:
                vmin = params['caxis_min'] if params.get('caxis_min') is not None else valid_values.min()
                vmax = params['caxis_max'] if params.get('caxis_max') is not None else valid_values.max()
                scatter.set_clim(vmin, vmax)
            else:
                scatter.set_clim(-1.0, 2.0)
        else:
            scatter.set_clim(params['caxis_min'], params['caxis_max'])
    
    ax.set_xlabel(r'$q_x [\AA^{-1}]$', fontsize=params['fontsize'])
    ax.set_ylabel(r'$q_y [\AA^{-1}]$', fontsize=params['fontsize'])
    if title:
        ax.set_title(title, fontsize=params['fontsize'] + 2)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=params['fontsize'])
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=params.get('save_dpi', 300), 
                    format='pdf', bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_from_data_file(data_file, smear=False, plot_both=False, save=False):
    """Read data file and create scattering pattern plot.
    
    Parameters are extracted from the filename and CONFIG.
    Defaults: no save, no smear, inosq only, always show.
    
    Parameters
    ----------
    data_file : str or Path
        Path to the data file
    smear : bool
        Apply resolution smearing (default: False)
    plot_both : bool
        Plot both inosq and iwithsq (default: False, inosq only)
    save : bool
        Save figures to PDF at 300 DPI (default: False)
    """
    data_file = Path(data_file)
    
    if not data_file.exists():
        print(f"Error: File not found: {data_file}")
        sys.exit(1)
    
    # Parse parameters from filename
    file_params = parse_filename(data_file)
    
    # Check required parameters were found
    missing = []
    if file_params['n_cyl'] is None:
        missing.append('n_cyl (e.g., 4cyl)')
    if file_params['radius'] is None:
        missing.append('radius (e.g., 25r)')
    if file_params['length'] is None:
        missing.append('length (e.g., 500l)')
    
    if missing:
        print(f"Warning: Could not parse from filename: {', '.join(missing)}")
        print(f"Filename: {data_file.name}")
        print("Using defaults: n_cyl=1, radius=1, length=100")
        file_params['n_cyl'] = file_params['n_cyl'] or 1
        file_params['radius'] = file_params['radius'] or 1
        file_params['length'] = file_params['length'] or 100
    
    stretch_val = file_params['stretch'] if file_params['stretch'] is not None else 0.0
    
    # Build full params dict
    params = {
        'n_cyl': file_params['n_cyl'],
        'radius': file_params['radius'],
        'length': file_params['length'],
        **CONFIG
    }
    
    print(f"Loading data from: {data_file}")
    print(f"  Stretch: {stretch_val}")
    print(f"  n_cyl: {params['n_cyl']}, radius: {params['radius']}, length: {params['length']}")
    
    # Load data
    data = np.loadtxt(data_file)
    qx = data[:, 0]
    qy = data[:, 1]
    p = data[:, 2]
    
    print(f"  Loaded {len(qx)} data points")
    
    # Calculate intensities
    inosq, iwithsq, s, q = calculate_intensity(qx, qy, p, params)

    if smear:
        inosq = smear_intensity(
            qx, qy, inosq,
            sigma_rel=CONFIG['smear_sigma_rel'],
            n_sigma=CONFIG['smear_n_sigma'],
            min_sigma=CONFIG['smear_min_sigma']
        )
        if plot_both:
            iwithsq = smear_intensity(
                qx, qy, iwithsq,
                sigma_rel=CONFIG['smear_sigma_rel'],
                n_sigma=CONFIG['smear_n_sigma'],
                min_sigma=CONFIG['smear_min_sigma']
            )
    
    # Build output filenames if saving
    output_dir = data_file.parent
    n_cyl_str = f"{int(round(params['n_cyl']))}"
    radius_str = f"{int(round(params['radius']))}"
    length_str = f"{int(round(params['length']))}"
    stretch_str = f"{stretch_val:.2f}".rstrip('0').rstrip('.')
    if not stretch_str or stretch_str == '.':
        stretch_str = '0'
    beta_str = f"{params['beta']:.2f}".rstrip('0').rstrip('.')
    if not beta_str or beta_str == '.':
        beta_str = '0'
    
    # Plot inosq
    inosq_path = output_dir / f'Phi0_St{stretch_str}_{n_cyl_str}cyl_{radius_str}r_{length_str}l_InoSq.pdf' if save else None
    plot_scattering_pattern(
        qx, qy, inosq, params,
        title=f"Scattering Pattern (No Structure Factor)\nStretch = {stretch_val}",
        output_path=inosq_path
    )
    
    # Plot iwithsq if requested
    if plot_both:
        iwithsq_path = output_dir / f'Phi0_St{stretch_str}_{n_cyl_str}cyl_{radius_str}r_{length_str}l_{beta_str}B_IwithSq.pdf' if save else None
        plot_scattering_pattern(
            qx, qy, iwithsq, params,
            title=f"Scattering Pattern (With Structure Factor)\nStretch = {stretch_val}",
            output_path=iwithsq_path
        )
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot FlowCalc scattering pattern data file."
    )
    parser.add_argument("data_file", type=str, help="Path to .dat file")
    parser.add_argument("--smear", action="store_true",
                        help="Apply q-dependent 2D Gaussian resolution smearing")
    parser.add_argument("--both", action="store_true",
                        help="Plot both inosq and iwithsq (default: inosq only)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to PDF at 300 DPI")
    
    if len(sys.argv) < 2:
        parser.print_help()
        print("\nParameters (St, cyl, r, l) are extracted from the filename.")
        print("Other parameters can be edited in the CONFIG section at the top of the file.")
        sys.exit(1)
    
    args = parser.parse_args()
    plot_from_data_file(args.data_file, smear=args.smear, plot_both=args.both, save=args.save)


if __name__ == "__main__":
    main()
