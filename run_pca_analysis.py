#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) for Scattering Patterns

This script performs PCA on scattering intensity patterns from the CRM output directory.
It reads .dat files containing 2D scattering patterns (qx, qy, p) and converts them
to 1D intensity arrays for dimensionality reduction analysis.

Default:
    python run_pca_analysis.py --input-dir output --output-dir pca_results --n-components 50 --n-modes 20 --q-min 0

Usage:
    python run_pca_analysis.py [--input-dir DIR] [--output-dir DIR] [--n-components N] [--n-modes M] [--load] [--q-min QMIN]
    [--length-min L] [--length-max L] [--stretch-min S] [--stretch-max S] [--n-cyl-min N] [--n-cyl-max N] [--radius-min R] [--radius-max R]
    Parameter filters (parsed from filenames) restrict which patterns are included in PCA, e.g. --length-min 300 --length-max 500.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.utils.extmath import randomized_svd
import seaborn as sns
import pickle


def parse_metadata_from_filename(filename: str) -> dict:
    """
    Parse physical parameters from filename.
    Expected format: PHIO_0_St{stretch}_{n_cyl}cyl_{radius}r_{length}l.dat
    Example: PHIO_0_St0.05_10cyl_10r_100l.dat
    
    Returns:
        Dictionary with keys: 'stretch', 'n_cyl', 'radius', 'length'
    """
    # Pattern to match: St{value}_{n}cyl_{r}r_{l}l
    pattern = r'St([\d.]+)_(\d+)cyl_(\d+)r_(\d+)l'
    match = re.search(pattern, filename)
    
    if match:
        stretch = float(match.group(1))
        n_cyl = int(match.group(2))
        radius = int(match.group(3))
        length = int(match.group(4))
        return {
            'stretch': stretch,
            'n_cyl': n_cyl,
            'radius': radius,
            'length': length
        }
    else:
        # Return None values if pattern doesn't match
        return {
            'stretch': None,
            'n_cyl': None,
            'radius': None,
            'length': None
        }


def _passes_param_filters(meta: dict, filters: dict) -> bool:
    """
    Return True if metadata passes all specified parameter range filters.
    Each filter is (min, max) inclusive; None means no bound.
    If a filter is set for a key but meta[key] is None (unparsed), return False.
    """
    for key, (lo, hi) in filters.items():
        if lo is None and hi is None:
            continue
        val = meta.get(key)
        if val is None:
            return False
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
    return True


def load_scattering_patterns(
    data_dir: Path,
    max_files: Optional[int] = None,
    beamstop_qmin: float = 0,
    *,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
    stretch_min: Optional[float] = None,
    stretch_max: Optional[float] = None,
    n_cyl_min: Optional[int] = None,
    n_cyl_max: Optional[int] = None,
    radius_min: Optional[int] = None,
    radius_max: Optional[int] = None,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load scattering patterns from .dat files using the native simulation spatial grid
    from the first valid file as a fixed template.

    Args:
        data_dir: Directory containing .dat files
        max_files: Maximum number of files to load (None = all)
        beamstop_qmin: Minimum |q|; intensity at q_mag < beamstop_qmin is set to 0 (default: 0)
        length_min, length_max: Include only files with length in [length_min, length_max] (parsed from filename)
        stretch_min, stretch_max: Include only files with stretch in [stretch_min, stretch_max]
        n_cyl_min, n_cyl_max: Include only files with n_cyl in [n_cyl_min, n_cyl_max]
        radius_min, radius_max: Include only files with radius in [radius_min, radius_max]

    Returns:
        Tuple of (intensity_matrix, file_names, q_values, qx_ref, qy_ref, phi_ref, metadata)
        - intensity_matrix: Array of shape (n_patterns, n_q_points)
        - file_names: List of source file names
        - q_values: |q| at each grid point (from reference grid)
        - qx_ref: qx coordinates from reference (first valid) file
        - qy_ref: qy coordinates from reference file
        - phi_ref: Polar angle phi = arctan2(qy, qx) from reference file
        - metadata: DataFrame with columns: stretch, n_cyl, radius, length
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    dat_files = sorted(data_dir.rglob("*.dat"))
    dat_files = [f for f in dat_files if not f.name.endswith("_fit.dat")]
    dat_files = [f for f in dat_files if "_nmax" not in f.stem]

    param_filters = {
        'length': (length_min, length_max),
        'stretch': (stretch_min, stretch_max),
        'n_cyl': (n_cyl_min, n_cyl_max),
        'radius': (radius_min, radius_max),
    }
    if any(lo is not None or hi is not None for lo, hi in param_filters.values()):
        n_before = len(dat_files)
        dat_files = [
            f for f in dat_files
            if _passes_param_filters(parse_metadata_from_filename(f.name), param_filters)
        ]
        print(f"Parameter filter: {n_before} -> {len(dat_files)} files")

    if max_files is not None:
        dat_files = dat_files[:max_files]

    if not dat_files:
        raise ValueError(f"No .dat files found in {data_dir} (or none passed parameter filters)")

    print(f"Found {len(dat_files)} .dat files to process")

    def load_one(dat_file: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        try:
            data = np.loadtxt(dat_file)
            if data.size == 0:
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 3:
                return None
            qx = data[:, 0]
            qy = data[:, 1]
            p = data[:, 2]
            return qx, qy, p
        except Exception:
            return None

    # Use first valid file to define master (qx, qy) reference grid
    master_qx = None
    master_qy = None
    n_ref = None
    for dat_file in dat_files:
        out = load_one(dat_file)
        if out is None:
            continue
        qx, qy, p = out
        master_qx = qx.copy()
        master_qy = qy.copy()
        n_ref = len(master_qx)
        print(f"Reference grid from first valid file: {dat_file.name} ({n_ref} points)")
        break

    if master_qx is None or n_ref is None:
        raise ValueError("No valid .dat file found to define reference grid")

    q_mag_ref = np.sqrt(master_qx**2 + master_qy**2)
    beamstop_mask = q_mag_ref < beamstop_qmin
    if beamstop_mask.any():
        print(f"Beamstop: masking {beamstop_mask.sum()} points where |q| < {beamstop_qmin}")

    patterns = []
    file_names = []
    metadata_list = []

    for dat_file in dat_files:
        out = load_one(dat_file)
        if out is None:
            print(f"Warning: Skipping invalid/empty file: {dat_file}")
            continue

        qx, qy, p = out
        if len(p) == n_ref:
            # Same grid size: assume same (qx, qy) order; use raw intensity
            row = p.astype(float, copy=True)
        else:
            # Different number of points: interpolate onto master grid
            row = griddata(
                (qx, qy), p, (master_qx, master_qy), method="linear", fill_value=0.0
            )
            row = np.asarray(row, dtype=float)
            if np.any(np.isnan(row)):
                row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply beamstop mask
        row[beamstop_mask] = 0.0
        patterns.append(row)
        file_names.append(dat_file.name)
        metadata_list.append(parse_metadata_from_filename(dat_file.name))

    if not patterns:
        raise ValueError("No valid patterns loaded")

    intensity_matrix = np.array(patterns)
    q_values_ref = np.sqrt(master_qx**2 + master_qy**2)
    phi_ref = np.arctan2(master_qy, master_qx)
    metadata_df = pd.DataFrame(metadata_list)

    print(
        f"Loaded {intensity_matrix.shape[0]} patterns with {intensity_matrix.shape[1]} q-points each"
    )
    return (
        intensity_matrix,
        file_names,
        q_values_ref,
        master_qx,
        master_qy,
        phi_ref,
        metadata_df,
    )


def perform_pca(intensity_matrix: np.ndarray, 
                n_components: int = 50,
                use_randomized: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Perform PCA on scattering intensity patterns.
    
    Args:
        intensity_matrix: Array of shape (n_patterns, n_q_points)
        n_components: Number of PCA components to compute
        use_randomized: Use randomized SVD for efficiency
        
    Returns:
        Tuple of (U, S, VT, pca_info)
        - U: Principal components (eigenvectors), shape (n_q_points, n_components)
        - S: Singular values, shape (n_components,)
        - VT: Right singular vectors, shape (n_components, n_patterns).
              PCA scores can be recovered as alpha = VT[:k].T * S[:k].
        - pca_info: Dictionary with PCA metadata (explained_variance_ratio, etc.)
    """
    n_samples, n_features = intensity_matrix.shape
    max_components = min(n_samples, n_features)
    
    # Limit n_components to maximum possible
    if n_components > max_components:
        print(f"Warning: Requested {n_components} components, but only {max_components} available. Using {max_components}.")
        n_components = max_components
    
    print(f"Performing PCA with {n_components} components on {n_samples} patterns...")
    
    # Center the data (subtract mean)
    mean_intensity = np.mean(intensity_matrix, axis=0)
    centered_data = intensity_matrix - mean_intensity
    
    # Transpose for SVD: we want modes in q-space (features)
    # centered_data.T is (n_q_points, n_patterns)
    # After SVD: U is (n_q_points, n_components), VT is (n_components, n_patterns)
    U, S, VT = randomized_svd(centered_data.T, n_components=n_components, 
                              random_state=42, n_iter=7)
    
    # Calculate explained variance
    explained_variance = S**2 / (n_samples - 1)
    total_variance = np.einsum('ij,ij->', centered_data, centered_data) / (n_samples - 1)
    explained_variance_ratio = explained_variance / total_variance
    
    # Create info dictionary (similar to sklearn PCA model)
    pca_info = {
        'explained_variance_': explained_variance,
        'explained_variance_ratio_': explained_variance_ratio,
        'n_components': n_components,
        'mean_': mean_intensity
    }
    
    print(f"PCA completed. Explained variance: {explained_variance_ratio.sum():.4f}")
    
    return U, S, VT, pca_info


def calculate_reconstruction_error(intensity_matrix: np.ndarray,
                                  U: np.ndarray,
                                  n_modes: int,
                                  mean_intensity: np.ndarray) -> float:
    """
    Calculate reconstruction error using specified number of PCA modes.

    Uses the orthogonal-projector identity to avoid allocating a full
    reconstructed matrix: ||X - X U_k U_k^T||^2 = ||X||^2 - ||X U_k||^2.
    
    Args:
        intensity_matrix: Original intensity patterns
        U: Principal components
        n_modes: Number of modes to use for reconstruction
        mean_intensity: Mean intensity (per feature) used for centering
        
    Returns:
        Normalized reconstruction error
    """
    centered_data = intensity_matrix - mean_intensity
    alpha = centered_data @ U[:, :n_modes]

    total_sq = np.einsum('ij,ij->', centered_data, centered_data)
    projected_sq = np.einsum('ij,ij->', alpha, alpha)
    residual_sq = max(total_sq - projected_sq, 0.0)

    rmse = np.sqrt(residual_sq / intensity_matrix.size)
    mean_val = np.mean(intensity_matrix)
    return rmse / mean_val if mean_val > 0 else rmse


def plot_singular_values(S: np.ndarray, output_path: Path):
    """Plot singular values vs mode number."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.loglog(np.arange(1, len(S) + 1), S, 'o-', markersize=4)
    ax.set_xlabel('Mode Number', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('PCA Singular Values', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _compute_raster_indices(qx: np.ndarray, qy: np.ndarray):
    """Validate that (qx, qy) form a regular raster and return index arrays.

    Returns (ix, iy, ux, uy) where ix[k], iy[k] are the column/row indices
    for the k-th data point, or None if the grid is not a regular raster.
    Computed once and reused for every mode via _fill_grid.
    """
    ux = np.unique(qx)
    uy = np.unique(qy)
    if len(ux) * len(uy) != len(qx):
        return None
    ix = np.searchsorted(ux, qx)
    iy = np.searchsorted(uy, qy)
    if not (np.allclose(ux[ix], qx) and np.allclose(uy[iy], qy)):
        return None
    return ix, iy, ux, uy


def _fill_grid(values: np.ndarray, ix: np.ndarray, iy: np.ndarray,
               ny: int, nx: int) -> np.ndarray:
    """Map 1-D values onto a 2-D grid using precomputed index arrays."""
    Z = np.empty((ny, nx))
    Z[:] = np.nan
    Z[iy, ix] = values
    return Z


def plot_pca_modes_2d(U: np.ndarray,
                      qx: np.ndarray,
                      qy: np.ndarray,
                      n_modes: int = 10,
                      output_path: Path = None):
    """
    Plot the first n_modes PCA modes as 2D scattering patterns (I vs qx, qy).
    Uses the native simulation grid: imshow for regular raster, tricontourf for scattered.
    """
    n_modes = min(n_modes, U.shape[1])

    n_cols = 3
    n_rows = (n_modes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), dpi=300)
    if n_modes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    vmin = np.percentile(U[:, :n_modes], 1)
    vmax = np.percentile(U[:, :n_modes], 99)

    # Precompute raster index map once (reused for every mode)
    raster = _compute_raster_indices(qx, qy)
    if raster is not None:
        r_ix, r_iy, r_ux, r_uy = raster
        r_extent = [r_ux[0], r_ux[-1], r_uy[0], r_uy[-1]]

    for i in range(n_modes):
        ax = axes[i]
        mode = U[:, i]

        if raster is not None:
            Z = _fill_grid(mode, r_ix, r_iy, len(r_uy), len(r_ux))
            im = ax.imshow(
                Z, extent=r_extent, origin='lower', aspect='equal',
                cmap='RdBu_r', vmin=vmin, vmax=vmax,
                interpolation='bilinear'
            )
            ax.set_xlabel(r'$q_x$ [Å⁻¹]', fontsize=10)
            ax.set_ylabel(r'$q_y$ [Å⁻¹]', fontsize=10)
            plt.colorbar(im, ax=ax, label='Amplitude')
        else:
            try:
                ax.tricontourf(qx, qy, mode, levels=32, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            except Exception:
                ax.scatter(qx, qy, c=mode, cmap='RdBu_r', s=1, vmin=vmin, vmax=vmax, edgecolors='none')
            ax.set_xlabel(r'$q_x$ [Å⁻¹]', fontsize=10)
            ax.set_ylabel(r'$q_y$ [Å⁻¹]', fontsize=10)
            ax.set_aspect('equal')
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Amplitude')

        ax.set_title(f'Mode {i+1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    for i in range(n_modes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'First {n_modes} PCA Modes (2D Scattering Patterns)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close()

    return fig


def plot_pca_modes_1d(U: np.ndarray,
                      q_values: Optional[np.ndarray],
                      n_modes: int,
                      output_path: Path):
    """
    Plot the first n_modes PCA modes as 1D intensity vs q (I(q)) in a single
    multi-panel figure, similar to the 2D mode plots.
    
    Args:
        U: Principal components, shape (n_q_points, n_components)
        q_values: Q-values corresponding to the mode values (or None to use index)
        n_modes: Number of modes to plot
        output_path: Path to save the combined plot
    """
    n_modes = min(n_modes, U.shape[1])
    
    # Subplot layout similar to 2D modes
    n_cols = 3
    n_rows = (n_modes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), dpi=300)
    if n_modes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x_axis = q_values if q_values is not None else np.arange(U.shape[0])
    x_label = r'$q$ [Å⁻¹]' if q_values is not None else 'Q-point Index'
    
    # Common y-limits across modes for easier comparison
    all_vals = U[:, :n_modes].ravel()
    y_min = np.percentile(all_vals, 1)
    y_max = np.percentile(all_vals, 99)
    
    for i in range(n_modes):
        ax = axes[i]
        mode = U[:, i]
        
        ax.plot(x_axis, mode, 'b-', linewidth=1.0)
        ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
        ax.set_title(f'Mode {i+1}', fontsize=11, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Mode Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)
    
    # Hide unused axes
    for i in range(n_modes, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'First {n_modes} PCA Modes (I(q))', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reconstructed_patterns(intensity_matrix: np.ndarray,
                                U: np.ndarray,
                                mean_intensity: np.ndarray,
                                n_patterns: int = 3,
                                n_modes_list: list = [5, 10, 20, 40],
                                q_values: Optional[np.ndarray] = None,
                                pattern_indices: Optional[list] = None,
                                output_path: Path = None):
    """
    Plot original vs reconstructed patterns using different numbers of modes.
    
    Args:
        intensity_matrix: Original intensity patterns
        U: Principal components
        mean_intensity: Mean intensity used for centering
        n_patterns: Number of example patterns to plot (used when pattern_indices is None)
        n_modes_list: List of number of modes to use for reconstruction
        q_values: Optional q-values for x-axis
        pattern_indices: Optional list of row indices to plot; when provided, overrides n_patterns
        output_path: Path to save the plot
    """
    n_max = intensity_matrix.shape[0]
    if pattern_indices is not None:
        pattern_indices = [int(i) for i in pattern_indices]
        pattern_indices = [max(0, min(i, n_max - 1)) for i in pattern_indices]
        n_patterns = len(pattern_indices)
    else:
        n_patterns = min(n_patterns, n_max)
        pattern_indices = np.linspace(0, n_max - 1, n_patterns, dtype=int)
    
    centered_data = intensity_matrix - mean_intensity
    
    n_cols = len(n_modes_list) + 1  # +1 for original
    fig, axes = plt.subplots(n_patterns, n_cols, figsize=(3*n_cols, 3*n_patterns), dpi=300)
    if n_patterns == 1:
        axes = axes.reshape(1, -1)
    
    x_axis = q_values if q_values is not None else np.arange(intensity_matrix.shape[1])
    x_label = r'$q$ [Å⁻¹]' if q_values is not None else 'Q-point Index'
    use_loglog = q_values is not None and np.all(np.asarray(q_values) > 0)
    
    for row, pattern_idx in enumerate(pattern_indices):
        original = intensity_matrix[pattern_idx, :]
        
        # Plot original
        ax = axes[row, 0]
        if use_loglog:
            orig_mask = np.isfinite(x_axis) & np.isfinite(original) & (x_axis > 0) & (original > 0)
            ax.loglog(x_axis[orig_mask], original[orig_mask], 'k-', linewidth=2, label='Original')
        else:
            ax.plot(x_axis, original, 'k-', linewidth=2, label='Original')
        ax.set_title('Original' if row == 0 else '', fontsize=10, fontweight='bold')
        if row == n_patterns - 1:
            ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot reconstructions with different numbers of modes
        for col, n_modes in enumerate(n_modes_list, start=1):
            n_modes = min(n_modes, U.shape[1])
            alpha = centered_data[pattern_idx:pattern_idx+1, :] @ U[:, :n_modes]
            reconstructed = (alpha @ U[:, :n_modes].T + mean_intensity).flatten()
            
            ax = axes[row, col]
            if use_loglog:
                orig_mask = np.isfinite(x_axis) & np.isfinite(original) & (x_axis > 0) & (original > 0)
                recon_mask = np.isfinite(x_axis) & np.isfinite(reconstructed) & (x_axis > 0) & (reconstructed > 0)
                ax.loglog(x_axis[orig_mask], original[orig_mask], 'k--', linewidth=1, alpha=0.5, label='Original')
                ax.loglog(x_axis[recon_mask], reconstructed[recon_mask], 'r-', linewidth=1.5, label=f'{n_modes} modes')
            else:
                ax.plot(x_axis, original, 'k--', linewidth=1, alpha=0.5, label='Original')
                ax.plot(x_axis, reconstructed, 'r-', linewidth=1.5, label=f'{n_modes} modes')
            ax.set_title(f'{n_modes} modes' if row == 0 else '', fontsize=10, fontweight='bold')
            if row == n_patterns - 1:
                ax.set_xlabel(x_label, fontsize=9)
            if col == 0:
                ax.set_ylabel('Intensity', fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Original vs Reconstructed Patterns', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return fig


def plot_mse_vs_modes(intensity_matrix: np.ndarray,
                      U: np.ndarray,
                      max_modes: int,
                      mean_intensity: np.ndarray,
                      S: np.ndarray,
                      output_path: Path):
    """Plot reconstruction MSE vs number of PCA modes.

    Uses the SVD identity: ||residual(k)||^2 = ||centered||^2 - sum(S[:k]^2),
    so no per-mode matrix reconstruction is needed.
    """
    n_samples, n_features = intensity_matrix.shape
    max_modes = min(max_modes, len(S), U.shape[1])

    total_sq = (np.einsum('ij,ij->', intensity_matrix, intensity_matrix)
                - n_samples * np.dot(mean_intensity, mean_intensity))
    mean_val = np.mean(intensity_matrix)

    cumulative_s2 = np.cumsum(S[:max_modes] ** 2)
    residual_sq = np.maximum(total_sq - cumulative_s2, 0.0)
    mse_values = np.sqrt(residual_sq / (n_samples * n_features))
    if mean_val > 0:
        mse_values = mse_values / mean_val

    modes_range = np.arange(1, max_modes + 1)

    for k in modes_range:
        if k % 10 == 0:
            print(f"  {k} modes: normalized error = {mse_values[k - 1]:.6e}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.semilogy(modes_range, mse_values, 'o-', markersize=4)
    ax.set_xlabel('Number of PCA Modes', fontsize=12)
    ax.set_ylabel('Normalized Reconstruction Error', fontsize=12)
    ax.set_title('Reconstruction Error vs PCA Modes', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return mse_values.tolist()


def plot_correlation_matrix(alpha: np.ndarray, metadata: pd.DataFrame, output_path: Path, n_components: int = 10):
    """
    Calculate and plot correlation matrix between PCA scores and physical parameters.
    
    Args:
        alpha: PCA scores (n_samples, n_components)
        metadata: DataFrame with physical parameters
        output_path: Path to save the correlation heatmap
        n_components: Number of PCA components to include
    """
    # Limit to first n_components
    n_components = min(n_components, alpha.shape[1])
    alpha_df = pd.DataFrame(alpha[:, :n_components], 
                           columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Combine with metadata
    combined_df = pd.concat([alpha_df, metadata], axis=1)
    
    # Calculate both Pearson and Spearman correlations
    param_cols = ['stretch', 'n_cyl', 'radius', 'length']
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    
    pearson_corr = combined_df[pc_cols + param_cols].corr(method='pearson').loc[pc_cols, param_cols]
    spearman_corr = combined_df[pc_cols + param_cols].corr(method='spearman').loc[pc_cols, param_cols]
    
    # Create subplots for both correlation types
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Pearson correlation
    sns.heatmap(pearson_corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Pearson r'})
    axes[0].set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Physical Parameters', fontsize=12)
    axes[0].set_ylabel('PCA Components', fontsize=12)
    
    # Spearman correlation
    sns.heatmap(spearman_corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Spearman ρ'})
    axes[1].set_title('Spearman Correlation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Physical Parameters', fontsize=12)
    axes[1].set_ylabel('PCA Components', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pearson_corr, spearman_corr


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Perform PCA on scattering patterns from CRM output",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output',
        help='Directory containing .dat files (default: output)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='pca_results',
        help='Output directory for results (default: pca_results)'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=50,
        help='Number of PCA components to compute (default: 50)'
    )
    parser.add_argument(
        '--n-modes',
        type=int,
        default=20,
        help='Number of PCA modes to use for analysis and plotting (default: 20)'
    )
    parser.add_argument(
        '--q-min',
        type=float,
        default=0,
        dest='q_min',
        help='Beamstop radius in q: intensity at |q| < this is set to 0 (default: 0)'
    )
    parser.add_argument('--length-min', type=int, default=None, help='Min length (from filename); only include patterns in range')
    parser.add_argument('--length-max', type=int, default=None, help='Max length (from filename)')
    parser.add_argument('--stretch-min', type=float, default=None, help='Min stretch (from filename)')
    parser.add_argument('--stretch-max', type=float, default=None, help='Max stretch (from filename)')
    parser.add_argument('--n-cyl-min', type=int, default=None, help='Min n_cyl (from filename)')
    parser.add_argument('--n-cyl-max', type=int, default=None, help='Max n_cyl (from filename)')
    parser.add_argument('--radius-min', type=int, default=None, help='Min radius (from filename)')
    parser.add_argument('--radius-max', type=int, default=None, help='Max radius (from filename)')

    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scattering patterns (reference grid from first valid .dat; beamstop applied)
    intensity_matrix, file_names, q_values, qx_ref, qy_ref, phi_ref, metadata = load_scattering_patterns(
        input_dir,
        beamstop_qmin=args.q_min,
        length_min=args.length_min,
        length_max=args.length_max,
        stretch_min=args.stretch_min,
        stretch_max=args.stretch_max,
        n_cyl_min=args.n_cyl_min,
        n_cyl_max=args.n_cyl_max,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
    )
    
    # Perform PCA
    U, S, VT, pca_model = perform_pca(intensity_matrix, n_components=args.n_components)
    if q_values is not None:
        pca_model["q_values"] = q_values
    if phi_ref is not None:
        pca_model["phi_values"] = phi_ref
    
    # Save PCA components
    pca_file = output_dir / 'pca_components.pkl'
    with open(pca_file, 'wb') as f:
        pickle.dump((U, S, pca_model), f)
    if q_values is not None:
        np.save(output_dir / 'q_values.npy', q_values)
    if qx_ref is not None and qy_ref is not None:
        np.save(output_dir / 'qx_ref.npy', qx_ref)
        np.save(output_dir / 'qy_ref.npy', qy_ref)
    if phi_ref is not None:
        np.save(output_dir / 'phi_ref.npy', phi_ref)
    
    # Compute PCA scores directly from SVD: alpha = VT[:k].T * S[:k]
    # This avoids re-allocating centered_data (= intensity_matrix - mean).
    mean_intensity = pca_model['mean_']
    n_modes = pca_model['n_components']
    alpha = VT[:n_modes, :].T * S[:n_modes]
    
    # Save reduced-order model
    rom_file = output_dir / f'reduced_order_model_{n_modes}modes.pkl'
    with open(rom_file, 'wb') as f:
        pickle.dump({
            'alpha': alpha,
            'U': U[:, :n_modes],
            'mean': mean_intensity,
            'file_names': file_names,
            'n_modes': n_modes
        }, f)
    
    # Calculate reconstruction error
    error = calculate_reconstruction_error(intensity_matrix, U, n_modes, mean_intensity)
    
    # Plot PCA modes as 2D scattering patterns
    n_modes_to_plot = min(args.n_modes, U.shape[1])
    if qx_ref is not None and qy_ref is not None:
        plot_pca_modes_2d(U, qx_ref, qy_ref, n_modes=n_modes_to_plot,
                          output_path=output_dir / 'pca_modes_2d.png')
    else:
        print("Warning: qx, qy coordinates not available, skipping 2D mode plots")
    
    # Plot PCA modes as 1D I(q) curves (all modes in one figure)
    if q_values is not None:
        plot_pca_modes_1d(U, q_values, n_modes=n_modes_to_plot,
                          output_path=output_dir / 'pca_modes_Ivq.png')
    else:
        print("Warning: q-values not available, skipping 1D mode plots")
    
    n_modes_list = [min(5, n_modes), min(10, n_modes), min(20, n_modes), n_modes]
    n_modes_list = sorted(set(n_modes_list))  # Remove duplicates and sort
    # Select patterns by stretch: lowest, middle, highest
    recon_pattern_indices = None
    if metadata is not None and 'stretch' in metadata.columns:
        valid = np.isfinite(metadata['stretch'])
        if valid.sum() >= 3:
            sorted_idx = metadata.loc[valid, 'stretch'].sort_values().index
            idx_list = sorted_idx.tolist()
            recon_pattern_indices = [
                idx_list[0],
                idx_list[len(idx_list) // 2],
                idx_list[-1],
            ]
    plot_reconstructed_patterns(intensity_matrix, U, mean_intensity,
                                n_patterns=3, n_modes_list=n_modes_list,
                                q_values=q_values,
                                pattern_indices=recon_pattern_indices,
                                output_path=output_dir / 'reconstructed_patterns.png')
    
    mse_values = plot_mse_vs_modes(intensity_matrix, U, 
                                  max_modes=min(args.n_components, U.shape[1]),
                                  mean_intensity=mean_intensity,
                                  S=S,
                                  output_path=output_dir / 'mse_vs_modes.png')
    
    # Save MSE values
    with open(output_dir / 'mse_values.pkl', 'wb') as f:
        pickle.dump(mse_values, f)
    
    # Correlation analysis with physical parameters
    if metadata is not None and not metadata.empty:
        
        # Filter out rows with missing metadata
        # Reset index to ensure alignment
        metadata = metadata.reset_index(drop=True)
        valid_mask = ~(metadata.isnull().any(axis=1))
        if valid_mask.sum() > 0:
            alpha_clean = alpha[valid_mask, :]
            metadata_clean = metadata[valid_mask].reset_index(drop=True)
            
            # Ensure we have enough components
            n_comp_for_corr = min(args.n_modes, alpha_clean.shape[1])
            
            # Correlation Matrix
            pearson_corr, spearman_corr = plot_correlation_matrix(
                alpha_clean, metadata_clean, 
                output_dir / 'correlation_matrix.png',
                n_components=n_comp_for_corr
            )
            
        else:
            print("\nWarning: No valid metadata found for correlation analysis")
    else:
        print("\nSkipping correlation analysis (no metadata available)")
    
    # Print summary
    print("\n" + "="*60)
    print("PCA Analysis Summary")
    print("="*60)
    print(f"Input patterns: {intensity_matrix.shape[0]}")
    print(f"Q-points per pattern: {intensity_matrix.shape[1]}")
    print(f"PCA components computed: {U.shape[1]}")
    print(f"Reduced-order modes used: {n_modes}")
    print(f"Explained variance: {pca_model['explained_variance_ratio_'].sum():.4f}")
    print(f"Reconstruction error: {error:.6e}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

