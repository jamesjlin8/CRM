#!/usr/bin/env python3
"""
Plot FlowCalc-generated scattering pattern data files.
Usage: python plotting.py <data_file> [data_file ...] [--sigma SIGMA] [--beta BETA]
       [--background BG] [--q-min QMIN] [--q-max QMAX] [--subtract FILE2] [--save] [--ivq]

Multiple files with --ivq overlay all I(q) curves (averaged and raw panels).
python plotting.py file --subtract file2  →  plot (file minus file2).

Model parameters (St, cyl, r, l) are extracted from each filename.
Use --q-min/--q-max to restrict I(q) overlays to a reliable experimental window.
"""

import argparse
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_SCALVOLFRAC = 8.577
DEFAULT_BACKGROUND = 0.0
SMEAR_N_SIGMA = 3.0
SMEAR_MIN_SIGMA = 1e-12
DEFAULT_SAVE_FORMAT = 'png'
DEFAULT_SAVE_DPI = 300


def _save_figure(fig, output_path, dpi=None):
    """Save a figure as PNG (format inferred from suffix, default png)."""
    output_path = Path(output_path)
    if output_path.suffix.lower() == '.pdf':
        output_path = output_path.with_suffix('.png')
    fmt = output_path.suffix.lstrip('.').lower() or DEFAULT_SAVE_FORMAT
    fig.savefig(
        output_path,
        dpi=dpi or DEFAULT_SAVE_DPI,
        format=fmt,
        bbox_inches='tight',
    )
    print(f"Saved: {output_path}")

RHEO_STITCHED_RE = re.compile(
    r"^(?P<conc>\d+mM)_(?P<shear>\d+)s1_all\.dat$", re.IGNORECASE
)
RHEO_DISTANCE_RE = re.compile(
    r"^(?P<conc>\d+mM)_(?P<shear>\d+)s1_(?P<dist>13mLen|\d+m)\.dat$",
    re.IGNORECASE,
)


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


def parse_rheo_stitched_filename(filename):
    """Parse Rheo-SANS names like 60mM_90s1_all.dat or 60mM_1200s1_12m.dat."""
    name = Path(filename).name
    match = RHEO_STITCHED_RE.match(name)
    if match:
        return {
            "concentration": match.group("conc"),
            "shear_rate": int(match.group("shear")),
            "detector_distance": None,
        }
    match = RHEO_DISTANCE_RE.match(name)
    if match:
        return {
            "concentration": match.group("conc"),
            "shear_rate": int(match.group("shear")),
            "detector_distance": match.group("dist"),
        }
    return None


def _rheo_legend_label(rheo_meta):
    label = f"{rheo_meta['concentration']}, {rheo_meta['shear_rate']} s$^{{-1}}$"
    if rheo_meta.get("detector_distance"):
        label += f", {rheo_meta['detector_distance']}"
    return label


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
                        cmap='turbo', marker='s', edgecolors='none')
    
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
        _save_figure(fig, output_path, dpi=params.get('save_dpi', DEFAULT_SAVE_DPI))
    
    return fig


def _radial_average(q, intensity, n_bins=200):
    """Compute 1D radial average I(q) from 2D intensity."""
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    mask = np.isfinite(q) & np.isfinite(intensity)
    if not np.any(mask):
        return np.array([]), np.array([])

    q = q[mask]
    intensity = intensity[mask]

    q_min, q_max = float(q.min()), float(q.max())
    if q_min == q_max:
        return np.array([q_min]), np.array([np.nanmean(intensity)])

    bin_edges = np.linspace(q_min, q_max, n_bins + 1)
    bin_indices = np.digitize(q, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    sums = np.bincount(bin_indices, weights=intensity, minlength=n_bins)
    counts = np.bincount(bin_indices, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        I_q = np.where(counts > 0, sums / counts, np.nan)

    q_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return q_centers, I_q


def _build_params(file_params, beta):
    """Assemble calculation and plot parameters for one file."""
    return {
        'n_cyl': file_params['n_cyl'],
        'radius': file_params['radius'],
        'length': file_params['length'],
        'beta': beta,
        'background': DEFAULT_BACKGROUND,
        'scalvolfrac': DEFAULT_SCALVOLFRAC,
        'graphingparameter': 4,
        'fontsize': 12,
        'caxis_min': None,
        'caxis_max': None,
        'fig_width': 6,
        'fig_height': 5,
        'save_dpi': 300,
    }


def _subtract_from(minuend, subtrahend):
    """Replace minuend intensities with minuend minus subtrahend."""
    if len(minuend['qx']) != len(subtrahend['qx']):
        print("Error: point count mismatch between subtract files.")
        sys.exit(1)
    if not (
        np.allclose(minuend['qx'], subtrahend['qx'])
        and np.allclose(minuend['qy'], subtrahend['qy'])
    ):
        print("Warning: q grids differ; subtracting by row order.")

    minuend['inosq'] = minuend['inosq'] - subtrahend['inosq']
    minuend['iwithsq'] = minuend['iwithsq'] - subtrahend['iwithsq']
    minuend['label'] = f"{minuend['label']} − {subtrahend['label']}"


def _legend_label(stretch_val, params):
    """Legend label with parsed model parameters."""
    stretch_str = f"{stretch_val:.2f}".rstrip('0').rstrip('.')
    if not stretch_str or stretch_str == '.':
        stretch_str = '0'

    n_cyl = int(round(params['n_cyl']))
    radius = params['radius']
    length = params['length']
    radius_str = f"{int(round(radius))}" if radius == int(round(radius)) else f"{radius:g}"
    length_str = f"{int(round(length))}" if length == int(round(length)) else f"{length:g}"

    return f"St{stretch_str}, {n_cyl}cyl, {radius_str}r, {length_str}l"


def _format_ivq_legend(ax, params=None):
    """Legend inside the plot axes."""
    p = params or {}
    fontsize = p.get('legend_fontsize', max(8, p.get('fontsize', 12) - 2))
    ax.legend(loc='best', fontsize=fontsize, frameon=True)


def _mask_positive_log(q, intensity):
    """Keep finite, positive values for log-log I(q) plots."""
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    mask = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
    return q[mask], intensity[mask]


def _create_ivq_axes(params=None, title_suffix=""):
    """Create upper (azimuthal average) and lower (raw points) I(q) subplots."""
    p = params or {}
    fontsize = p.get('fontsize', 12)
    fig_height = p.get('fig_height', 5) * 1.6
    fig, (ax_avg, ax_raw) = plt.subplots(
        2, 1, sharex=True,
        figsize=(p.get('fig_width', 6), fig_height),
    )
    ax_avg.set_ylabel(r'$I(q)$', fontsize=fontsize)
    ax_raw.set_ylabel(r'$I(q)$', fontsize=fontsize)
    ax_raw.set_xlabel(r'$q$', fontsize=fontsize)
    ax_avg.set_title('Azimuthal average', fontsize=fontsize, loc='left')
    ax_raw.set_title('All points (not azimuthally averaged)', fontsize=fontsize, loc='left')
    if title_suffix:
        fig.suptitle(f'I(q) {title_suffix}', fontsize=fontsize + 2, y=1.02)
    return fig, ax_avg, ax_raw


def _apply_q_range(q, intensity, q_min=None, q_max=None):
    """Keep points inside an optional |q| window."""
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    mask = np.ones(q.shape, dtype=bool)
    if q_min is not None:
        mask &= q >= float(q_min)
    if q_max is not None:
        mask &= q <= float(q_max)
    return q[mask], intensity[mask]


def _plot_ivq_series(ax_avg, ax_raw, q, intensity, label, params=None,
                     q_min=None, q_max=None):
    """Plot azimuthally averaged curve and raw scatter for one intensity series."""
    q_win, intensity_win = _apply_q_range(q, intensity, q_min=q_min, q_max=q_max)
    q_pos, intensity_pos = _mask_positive_log(q_win, intensity_win)
    if q_pos.size == 0:
        print(f"Warning: No positive intensities to plot for {label or 'series'}.")
        return

    q_1d, Iq = _radial_average(q_pos, intensity_pos)
    if q_1d.size == 0:
        print(f"Warning: Could not compute radial average for {label or 'series'}.")
        return

    series_label = label or "I(q)"
    (line_avg,) = ax_avg.plot(q_1d, Iq, label=series_label)
    color = line_avg.get_color()
    ax_raw.scatter(
        q_pos, intensity_pos,
        s=1, alpha=0.2, c=color, label=series_label,
        edgecolors='none', rasterized=True,
    )


def _finalize_ivq_axes(ax_avg, ax_raw, params=None, q_min=None, q_max=None):
    """Apply log scales, legend, and layout to I(q) subplots."""
    p = params or {}
    fontsize = p.get('fontsize', 12)
    for ax in (ax_avg, ax_raw):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=fontsize)
        if q_min is not None or q_max is not None:
            left = float(q_min) if q_min is not None else None
            right = float(q_max) if q_max is not None else None
            ax.set_xlim(left, right)
    _format_ivq_legend(ax_avg, params)


def plot_I_vs_q(q, inosq, iwithsq=None, params=None, title_suffix="", output_path=None,
                label=None, ax_avg=None, ax_raw=None, q_min=None, q_max=None):
    """Plot azimuthally averaged I(q) and raw (non-averaged) intensities vs q."""
    own_fig = ax_avg is None
    if own_fig:
        fig, ax_avg, ax_raw = _create_ivq_axes(params, title_suffix=title_suffix)
    else:
        fig = ax_avg.figure

    _plot_ivq_series(ax_avg, ax_raw, q, inosq, label, params, q_min=q_min, q_max=q_max)
    if iwithsq is not None:
        withsq_label = f"{label} with S(q)" if label else "I(q) with S(q)"
        _plot_ivq_series(
            ax_avg, ax_raw, q, iwithsq, withsq_label, params,
            q_min=q_min, q_max=q_max,
        )

    if own_fig:
        _finalize_ivq_axes(ax_avg, ax_raw, params, q_min=q_min, q_max=q_max)
        plt.tight_layout()

        if output_path is not None:
            dpi = params.get('save_dpi', DEFAULT_SAVE_DPI) if params else DEFAULT_SAVE_DPI
            _save_figure(fig, output_path, dpi=dpi)

    return fig


def plot_I_vs_q_overlay(datasets, params=None, use_with_sq=False, title="", output_path=None,
                        q_min=None, q_max=None):
    """Plot multiple I(q) curves on shared average and raw subplots.

    Parameters
    ----------
    datasets : list of dict
        Each dict has keys: q, inosq, iwithsq (optional), label, data_file
    """
    if not datasets:
        return None

    fig, ax_avg, ax_raw = _create_ivq_axes(params)
    if title:
        fig.suptitle(title, fontsize=(params.get('fontsize', 12) + 2 if params else 14), y=1.02)

    for entry in datasets:
        _plot_ivq_series(
            ax_avg, ax_raw,
            entry['q'], entry['inosq'],
            entry['label'], params,
            q_min=q_min, q_max=q_max,
        )
        if use_with_sq:
            withsq_label = f"{entry['label']} with S(q)"
            _plot_ivq_series(
                ax_avg, ax_raw,
                entry['q'], entry['iwithsq'],
                withsq_label, params,
                q_min=q_min, q_max=q_max,
            )

    _finalize_ivq_axes(ax_avg, ax_raw, params, q_min=q_min, q_max=q_max)
    plt.tight_layout()

    if output_path is not None:
        dpi = params.get('save_dpi', DEFAULT_SAVE_DPI) if params else DEFAULT_SAVE_DPI
        _save_figure(fig, output_path, dpi=dpi)

    return fig


def _load_processed_data(data_file, sigma=0.0, beta=0.0, background=None):
    """Load a data file and return processed scattering arrays."""
    data_file = Path(data_file)

    if not data_file.exists():
        print(f"Error: File not found: {data_file}")
        sys.exit(1)

    rheo_meta = parse_rheo_stitched_filename(data_file)
    file_params = parse_filename(data_file)

    if rheo_meta is None:
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
    params = _build_params(file_params, beta)
    if background is not None:
        params['background'] = float(background)

    print(f"Loading data from: {data_file}")
    if rheo_meta is not None:
        msg = f"  Rheo-SANS: {rheo_meta['concentration']}, shear = {rheo_meta['shear_rate']} s^-1"
        if rheo_meta.get("detector_distance"):
            msg += f", detector = {rheo_meta['detector_distance']}"
        print(msg)
    else:
        print(f"  Stretch: {stretch_val}")
        print(f"  n_cyl: {params['n_cyl']}, radius: {params['radius']}, length: {params['length']}")
        print(f"  background: {params['background']}")
    if sigma > 0:
        print(f"  Smearing sigma: {sigma}")
    if beta > 0:
        print(f"  Structure factor beta: {beta}")

    data = np.loadtxt(data_file)
    if rheo_meta is not None:
        qx, qy, p = data[:, 0], data[:, 1], data[:, 2]
    else:
        qx = data[:, 1]
        qy = data[:, 0]
        p = data[:, 2]

    print(f"  Loaded {len(qx)} data points")

    if rheo_meta is not None:
        q = np.sqrt(qx ** 2 + qy ** 2)
        inosq = p.astype(float, copy=False)
        iwithsq = inosq
        label = _rheo_legend_label(rheo_meta)
    else:
        inosq, iwithsq, s, q = calculate_intensity(qx, qy, p, params)
        label = _legend_label(stretch_val, params)

    if sigma > 0:
        inosq = smear_intensity(
            qx, qy, inosq,
            sigma_rel=sigma,
            n_sigma=SMEAR_N_SIGMA,
            min_sigma=SMEAR_MIN_SIGMA,
        )
        if beta > 0:
            iwithsq = smear_intensity(
                qx, qy, iwithsq,
                sigma_rel=sigma,
                n_sigma=SMEAR_N_SIGMA,
                min_sigma=SMEAR_MIN_SIGMA,
            )

    return {
        'data_file': data_file,
        'qx': qx,
        'qy': qy,
        'inosq': inosq,
        'iwithsq': iwithsq,
        'q': q,
        'params': params,
        'stretch_val': stretch_val,
        'label': label,
    }


def _load_subtrahend(subtract, sigma, beta, background=None):
    """Load the file passed to --subtract."""
    subtrahend = _load_processed_data(
        subtract, sigma=sigma, beta=beta, background=background,
    )
    return subtrahend


def _apply_subtract(minuend, subtract, sigma, beta, background=None):
    """Subtract --subtract file from the primary (first) file."""
    minuend_path = minuend['data_file'].resolve()
    subtrahend_path = Path(subtract).resolve()

    if minuend_path == subtrahend_path:
        print("Error: --subtract file must differ from the data file.")
        sys.exit(1)

    subtrahend = _load_subtrahend(subtract, sigma, beta, background=background)
    _subtract_from(minuend, subtrahend)
    print(
        f"Subtracted {subtrahend['data_file'].name} "
        f"from {minuend['data_file'].name}"
    )


def plot_from_loaded(loaded, save=False, plot_ivq=False, use_with_sq=False, show=True,
                     q_min=None, q_max=None):
    """Plot pre-loaded scattering data."""
    data_file = loaded['data_file']
    qx, qy = loaded['qx'], loaded['qy']
    inosq, iwithsq, q = loaded['inosq'], loaded['iwithsq'], loaded['q']
    params = loaded['params']
    stretch_val = loaded['stretch_val']

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
    inosq_path = output_dir / f'Phi0_St{stretch_str}_{n_cyl_str}cyl_{radius_str}r_{length_str}l_InoSq.png' if save else None
    plot_scattering_pattern(
        qx, qy, inosq, params,
        title=f"Scattering Pattern (No Structure Factor)\nStretch = {stretch_val}",
        output_path=inosq_path
    )
    
    # Plot iwithsq if structure factor is enabled
    if use_with_sq:
        iwithsq_path = output_dir / f'Phi0_St{stretch_str}_{n_cyl_str}cyl_{radius_str}r_{length_str}l_{beta_str}B_IwithSq.png' if save else None
        plot_scattering_pattern(
            qx, qy, iwithsq, params,
            title=f"Scattering Pattern (With Structure Factor)\nStretch = {stretch_val}",
            output_path=iwithsq_path
        )

    # Plot I vs q (1D) if requested
    if plot_ivq:
        ivq_suffix = f"(Stretch = {stretch_val})"
        ivq_path = None
        if save:
            ivq_path = output_dir / f'Phi0_St{stretch_str}_{n_cyl_str}cyl_{radius_str}r_{length_str}l_Iq.png'
        plot_I_vs_q(
            q, inosq, iwithsq if use_with_sq else None,
            params=params,
            title_suffix=ivq_suffix,
            output_path=ivq_path,
            label=loaded['label'],
            q_min=q_min,
            q_max=q_max,
        )
    
    if show:
        plt.show()


def plot_multiple_data_files(data_files, sigma=0.0, beta=0.0, save=False, plot_ivq=False,
                             subtract=None, output_path=None, title=None,
                             q_min=None, q_max=None, background=None):
    """Plot multiple data files; overlay I(q) when --ivq is set."""
    data_files = [Path(f) for f in data_files]
    use_with_sq = beta > 0

    loadeds = [
        _load_processed_data(data_file, sigma=sigma, beta=beta, background=background)
        for data_file in data_files
    ]

    if subtract is not None:
        _apply_subtract(loadeds[0], subtract, sigma, beta, background=background)

    if plot_ivq:
        datasets = [{
            'data_file': loaded['data_file'],
            'q': loaded['q'],
            'inosq': loaded['inosq'],
            'iwithsq': loaded['iwithsq'],
            'label': loaded['label'],
        } for loaded in loadeds]

        ivq_path = None
        if save:
            ivq_path = Path(output_path) if output_path else datasets[0]['data_file'].parent / 'Iq_overlay.png'

        plot_I_vs_q_overlay(
            datasets,
            params=loadeds[0]['params'],
            use_with_sq=use_with_sq,
            title=title or 'I(q) overlay',
            output_path=ivq_path,
            q_min=q_min,
            q_max=q_max,
        )
        if save:
            plt.close('all')
        else:
            plt.show()
        return

    for i, loaded in enumerate(loadeds):
        plot_from_loaded(
            loaded,
            save=save,
            plot_ivq=False,
            use_with_sq=use_with_sq,
            show=(i == len(loadeds) - 1),
            q_min=q_min,
            q_max=q_max,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot FlowCalc scattering pattern data file(s)."
    )
    parser.add_argument(
        "data_files",
        nargs="+",
        type=str,
        help="One or more paths to .dat files (use --ivq to overlay I(q) on one axis)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Resolution smearing strength (0 = off; typical values ~0.1–0.2)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Structure factor parameter (0 = form factor only; >0 also plots I with S(q))",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=None,
        help=(
            "Constant background added to simulated intensities "
            f"(default: {DEFAULT_BACKGROUND}; ignored for Rheo-SANS files)"
        ),
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=None,
        dest="q_min",
        help="Lower |q| cutoff for I(q) plots (default: none)",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=None,
        dest="q_max",
        help="Upper |q| cutoff for I(q) plots (default: none)",
    )
    parser.add_argument(
        "--subtract",
        type=str,
        default=None,
        metavar="FILE",
        help="Subtract FILE from the data file (e.g. plotting.py file --subtract FILE)",
    )
    parser.add_argument("--save", action="store_true",
                        help="Save figures to PNG at 300 DPI")
    parser.add_argument("--ivq", action="store_true",
                        help="Plot I(q): azimuthal average (top) and all raw points (bottom)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for --ivq overlay PNG (default: Iq_overlay.png next to first file)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title for --ivq overlay",
    )
    
    args = parser.parse_args()
    use_with_sq = args.beta > 0

    if len(args.data_files) == 1:
        loaded = _load_processed_data(
            args.data_files[0],
            sigma=args.sigma,
            beta=args.beta,
            background=args.background,
        )
        if args.subtract is not None:
            _apply_subtract(
                loaded, args.subtract, args.sigma, args.beta,
                background=args.background,
            )
        plot_from_loaded(
            loaded,
            save=args.save,
            plot_ivq=args.ivq,
            use_with_sq=use_with_sq,
            q_min=args.q_min,
            q_max=args.q_max,
        )
    else:
        if not args.ivq:
            print("Note: multiple files without --ivq produce separate 2D plots.")
            print("      Use --ivq to overlay all I(q) curves on one axis.")
        plot_multiple_data_files(
            args.data_files,
            sigma=args.sigma,
            beta=args.beta,
            save=args.save,
            plot_ivq=args.ivq,
            subtract=args.subtract,
            output_path=args.output,
            title=args.title,
            q_min=args.q_min,
            q_max=args.q_max,
            background=args.background,
        )


if __name__ == "__main__":
    main()
