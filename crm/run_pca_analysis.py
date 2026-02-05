#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) for Scattering Patterns

This script performs PCA on scattering intensity patterns from the CRM output directory.
It reads .dat files containing 2D scattering patterns (qx, qy, p) and converts them
to 1D intensity arrays for dimensionality reduction analysis.

Usage:
    python pca_analysis.py [--input-dir DIR] [--output-dir DIR] [--n-components N] [--n-modes M] [--load]
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def load_scattering_patterns(data_dir: Path, max_files: Optional[int] = None) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load scattering patterns from .dat files in the specified directory.
    
    Args:
        data_dir: Directory containing .dat files
        max_files: Maximum number of files to load (None = all)
        
    Returns:
        Tuple of (intensity_matrix, file_names, q_values, qx_ref, qy_ref, metadata)
        - intensity_matrix: Array of shape (n_patterns, n_q_points)
        - file_names: List of source file names
        - q_values: Q-values corresponding to the columns (sorted by q magnitude)
        - qx_ref: qx coordinates from reference file (sorted by q magnitude)
        - qy_ref: qy coordinates from reference file (sorted by q magnitude)
        - metadata: DataFrame with columns: stretch, n_cyl, radius, length
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Find all .dat files recursively
    dat_files = sorted(data_dir.rglob("*.dat"))
    
    # Filter out fit result files (they have different format)
    dat_files = [f for f in dat_files if not f.name.endswith('_fit.dat')]
    
    if max_files is not None:
        dat_files = dat_files[:max_files]
    
    if not dat_files:
        raise ValueError(f"No .dat files found in {data_dir}")
    
    print(f"Found {len(dat_files)} .dat files to process")
    
    patterns = []
    file_names = []
    metadata_list = []
    q_values_ref = None  # Store q-values from first file
    qx_ref = None
    qy_ref = None
    
    for idx, dat_file in enumerate(dat_files):
        try:
            # Load data: columns are qx, qy, p
            data = np.loadtxt(dat_file)
            
            if data.size == 0:
                print(f"Warning: Empty file: {dat_file}")
                continue
            
            if data.ndim == 1:
                # Single row case
                data = data.reshape(1, -1)
            
            if data.shape[1] < 3:
                print(f"Warning: Invalid format in {dat_file}: expected 3 columns, got {data.shape[1]}")
                continue
            
            qx = data[:, 0]
            qy = data[:, 1]
            p = data[:, 2]
            
            # Convert 2D pattern to 1D intensity array
            # Calculate q magnitude and sort by q
            q = np.sqrt(qx**2 + qy**2)
            q_sorted_idx = np.argsort(q)
            q_sorted = q[q_sorted_idx]
            p_sorted = p[q_sorted_idx]
            
            # Store q-values and coordinates from first file for reference
            if idx == 0:
                q_values_ref = q_sorted
                qx_ref = qx[q_sorted_idx]
                qy_ref = qy[q_sorted_idx]
            
            # Store as intensity pattern
            patterns.append(p_sorted)
            file_names.append(dat_file.name)
            
            # Extract metadata from filename
            metadata_dict = parse_metadata_from_filename(dat_file.name)
            metadata_list.append(metadata_dict)
            
        except Exception as e:
            print(f"Warning: Error loading {dat_file}: {e}")
            continue
    
    if not patterns:
        raise ValueError("No valid patterns loaded")
    
    # Find common q-grid length (use minimum to ensure all patterns have same length)
    min_length = min(len(p) for p in patterns)
    
    # Truncate all patterns to same length
    patterns = [p[:min_length] for p in patterns]
    
    # Truncate q_values and coordinates to match
    if q_values_ref is not None:
        q_values_ref = q_values_ref[:min_length]
        qx_ref = qx_ref[:min_length]
        qy_ref = qy_ref[:min_length]
    
    # Stack into matrix: rows = patterns, columns = q-points
    intensity_matrix = np.array(patterns)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    
    print(f"Loaded {intensity_matrix.shape[0]} patterns with {intensity_matrix.shape[1]} q-points each")
    
    return intensity_matrix, file_names, q_values_ref, qx_ref, qy_ref, metadata_df


def perform_pca(intensity_matrix: np.ndarray, 
                n_components: int = 50,
                use_randomized: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Perform PCA on scattering intensity patterns.
    
    Args:
        intensity_matrix: Array of shape (n_patterns, n_q_points)
        n_components: Number of PCA components to compute
        use_randomized: Use randomized SVD for efficiency
        
    Returns:
        Tuple of (U, S, pca_info)
        - U: Principal components (eigenvectors), shape (n_q_points, n_components)
        - S: Singular values, shape (n_components,)
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
    total_variance = np.var(centered_data, axis=0, ddof=1).sum()
    explained_variance_ratio = explained_variance / total_variance
    
    # Create info dictionary (similar to sklearn PCA model)
    pca_info = {
        'explained_variance_': explained_variance,
        'explained_variance_ratio_': explained_variance_ratio,
        'n_components': n_components,
        'mean_': mean_intensity
    }
    
    print(f"PCA completed. Explained variance: {explained_variance_ratio.sum():.4f}")
    
    return U, S, pca_info


def calculate_reconstruction_error(intensity_matrix: np.ndarray,
                                  U: np.ndarray,
                                  n_modes: int,
                                  mean_intensity: np.ndarray) -> float:
    """
    Calculate reconstruction error using specified number of PCA modes.
    
    Args:
        intensity_matrix: Original intensity patterns
        U: Principal components
        n_modes: Number of modes to use for reconstruction
        mean_intensity: Mean intensity (per feature) used for centering
        
    Returns:
        Normalized reconstruction error
    """
    centered_data = intensity_matrix - mean_intensity
    
    # Project onto PCA space and reconstruct
    alpha = centered_data @ U[:, :n_modes]
    reconstructed = alpha @ U[:, :n_modes].T + mean_intensity
    
    # Calculate normalized RMSE
    rmse = np.sqrt(np.mean((intensity_matrix - reconstructed)**2))
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


def plot_pca_modes_2d(U: np.ndarray,
                      qx: np.ndarray,
                      qy: np.ndarray,
                      n_modes: int = 10,
                      output_path: Path = None):
    """
    Plot the first n_modes PCA modes as 2D scattering patterns (I vs qx, qy).
    
    Args:
        U: Principal components, shape (n_q_points, n_components)
        qx: qx coordinates corresponding to the mode values
        qy: qy coordinates corresponding to the mode values
        n_modes: Number of modes to plot
        output_path: Path to save the plot
    """
    n_modes = min(n_modes, U.shape[1])
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_modes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), dpi=300)
    if n_modes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Find common color scale for all modes
    all_mode_values = []
    for i in range(n_modes):
        mode = U[:, i]
        all_mode_values.extend(mode)
    vmin = np.percentile(all_mode_values, 1)
    vmax = np.percentile(all_mode_values, 99)
    
    for i in range(n_modes):
        ax = axes[i]
        mode = U[:, i]
        
        # Create scatter plot
        scatter = ax.scatter(qy, qx, c=mode, cmap='RdBu_r', s=1, 
                            vmin=vmin, vmax=vmax, edgecolors='none')
        ax.set_title(f'Mode {i+1}', fontsize=11, fontweight='bold')
        ax.set_xlabel(r'$q_x$ [Å⁻¹]', fontsize=10)
        ax.set_ylabel(r'$q_y$ [Å⁻¹]', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Amplitude', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_modes, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'First {n_modes} PCA Modes (2D Scattering Patterns)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return fig


def plot_reconstructed_patterns(intensity_matrix: np.ndarray,
                                U: np.ndarray,
                                mean_intensity: np.ndarray,
                                n_patterns: int = 3,
                                n_modes_list: list = [5, 10, 20, 40],
                                q_values: Optional[np.ndarray] = None,
                                output_path: Path = None):
    """
    Plot original vs reconstructed patterns using different numbers of modes.
    
    Args:
        intensity_matrix: Original intensity patterns
        U: Principal components
        mean_intensity: Mean intensity used for centering
        n_patterns: Number of example patterns to plot
        n_modes_list: List of number of modes to use for reconstruction
        q_values: Optional q-values for x-axis
        output_path: Path to save the plot
    """
    n_patterns = min(n_patterns, intensity_matrix.shape[0])
    pattern_indices = np.linspace(0, intensity_matrix.shape[0] - 1, n_patterns, dtype=int)
    
    centered_data = intensity_matrix - mean_intensity
    
    n_cols = len(n_modes_list) + 1  # +1 for original
    fig, axes = plt.subplots(n_patterns, n_cols, figsize=(3*n_cols, 3*n_patterns), dpi=300)
    if n_patterns == 1:
        axes = axes.reshape(1, -1)
    
    x_axis = q_values if q_values is not None else np.arange(intensity_matrix.shape[1])
    x_label = r'$q$ [Å⁻¹]' if q_values is not None else 'Q-point Index'
    use_log = True  # Use log scale for intensity
    
    for row, pattern_idx in enumerate(pattern_indices):
        original = intensity_matrix[pattern_idx, :]
        
        # Plot original
        ax = axes[row, 0]
        if use_log:
            ax.semilogy(x_axis, original, 'k-', linewidth=2, label='Original')
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
            if use_log:
                ax.semilogy(x_axis, original, 'k--', linewidth=1, alpha=0.5, label='Original')
                ax.semilogy(x_axis, reconstructed, 'r-', linewidth=1.5, label=f'{n_modes} modes')
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
                      output_path: Path):
    """Plot reconstruction MSE vs number of PCA modes."""
    mse_values = []
    modes_range = range(1, min(max_modes + 1, U.shape[1] + 1))
    
    print(f"Calculating MSE for {len(modes_range)} mode numbers...")
    for n_modes in modes_range:
        error = calculate_reconstruction_error(intensity_matrix, U, n_modes, mean_intensity)
        mse_values.append(error)
        if n_modes % 10 == 0:
            print(f"  {n_modes} modes: normalized error = {error:.6e}")
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.semilogy(modes_range, mse_values, 'o-', markersize=4)
    ax.set_xlabel('Number of PCA Modes', fontsize=12)
    ax.set_ylabel('Normalized Reconstruction Error', fontsize=12)
    ax.set_title('Reconstruction Error vs PCA Modes', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mse_values


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
    
    print(f"Saved correlation matrix to {output_path}")
    
    return pearson_corr, spearman_corr


def plot_parameter_trajectory(alpha: np.ndarray, metadata: pd.DataFrame, 
                              parameter: str, correlation_matrix: pd.DataFrame,
                              output_path: Path):
    """
    Plot the trajectory using the two PC modes with highest correlation to the parameter.
    
    Args:
        alpha: PCA scores (n_samples, n_components)
        metadata: DataFrame with physical parameters
        parameter: Name of parameter to plot trajectory for
        correlation_matrix: DataFrame with correlations (PCs x parameters)
        output_path: Path to save the plot
    """
    if parameter not in metadata.columns:
        raise ValueError(f"Parameter '{parameter}' not found in metadata")
    
    # Find the two PCs with highest absolute correlation to this parameter
    if parameter not in correlation_matrix.columns:
        raise ValueError(f"Parameter '{parameter}' not found in correlation matrix")
    
    abs_corr = correlation_matrix[parameter].abs()
    top2_idx = abs_corr.nlargest(2).index.tolist()
    
    # Extract PC numbers (e.g., 'PC1' -> 0, 'PC2' -> 1)
    pc_nums = [int(pc_idx[2:]) - 1 for pc_idx in top2_idx]
    pc_labels = [f'PC{num+1}' for num in pc_nums]
    
    # Sort by parameter value
    sort_idx = metadata[parameter].sort_values().index
    alpha_sorted = alpha[sort_idx, :]
    param_sorted = metadata[parameter].iloc[sort_idx].values
    
    # Use the top 2 correlated PCs
    pc1_idx, pc2_idx = pc_nums[0], pc_nums[1]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    scatter = ax.scatter(alpha_sorted[:, pc1_idx], alpha_sorted[:, pc2_idx],
                        c=param_sorted, cmap='viridis', s=50, alpha=0.7)
    ax.plot(alpha_sorted[:, pc1_idx], alpha_sorted[:, pc2_idx], 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel(f'{pc_labels[0]} (r={correlation_matrix.loc[top2_idx[0], parameter]:.3f})', fontsize=11)
    ax.set_ylabel(f'{pc_labels[1]} (r={correlation_matrix.loc[top2_idx[1], parameter]:.3f})', fontsize=11)
    ax.set_title(f'Trajectory: {parameter.capitalize()} (Top 2 Correlated PCs)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=parameter.capitalize())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parameter trajectory plot for {parameter} to {output_path}")


def plot_pairs_plot(alpha: np.ndarray, metadata: pd.DataFrame, 
                    output_path: Path, n_components: int = 5):
    """
    Generate a pairs plot for the first n PC scores, colored by 'Stretch'.
    
    Args:
        alpha: PCA scores (n_samples, n_components)
        metadata: DataFrame with physical parameters
        output_path: Path to save the plot
        n_components: Number of PC components to include in pairs plot
    """
    if 'stretch' not in metadata.columns:
        print("Warning: 'stretch' parameter not found, skipping pairs plot")
        return
    
    n_components = min(n_components, alpha.shape[1])
    
    # Create DataFrame with PC scores and stretch
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df = pd.DataFrame(alpha[:, :n_components], columns=pc_cols)
    df['stretch'] = metadata['stretch'].values
    
    # Filter out rows with missing stretch values
    df_clean = df.dropna(subset=['stretch'])
    
    if len(df_clean) == 0:
        print("Warning: No valid stretch values, skipping pairs plot")
        return
    
    # Create pairplot
    g = sns.pairplot(df_clean, vars=pc_cols, hue='stretch', 
                    palette='viridis', plot_kws={'alpha': 0.6, 's': 30})
    g.fig.suptitle('PC Scores Pairs Plot (Colored by Stretch)', 
                   fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved pairs plot to {output_path}")


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
        default=10,
        help='Number of PCA modes to use for analysis and plotting (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scattering patterns
    intensity_matrix, file_names, q_values, qx_ref, qy_ref, metadata = load_scattering_patterns(input_dir)
    
    # Perform PCA
    U, S, pca_model = perform_pca(intensity_matrix, n_components=args.n_components)
    if q_values is not None:
        pca_model["q_values"] = q_values
    
    # Save PCA components
    pca_file = output_dir / 'pca_components.pkl'
    with open(pca_file, 'wb') as f:
        pickle.dump((U, S, pca_model), f)
    print(f"Saved PCA components to {pca_file}")
    if q_values is not None:
        np.save(output_dir / 'q_values.npy', q_values)
    
    # Calculate reduced-order model coefficients
    mean_intensity = pca_model['mean_']
    centered_data = intensity_matrix - mean_intensity
    n_modes = pca_model['n_components']  # Use actual number of components computed
    alpha = centered_data @ U[:, :n_modes]
    
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
    print(f"Saved reduced-order model (shape: {alpha.shape}) to {rom_file}")
    
    # Calculate reconstruction error
    error = calculate_reconstruction_error(intensity_matrix, U, n_modes, mean_intensity)
    
    # Generate plots
    print("Generating plots...")
    # Plot PCA modes as 2D scattering patterns
    n_modes_to_plot = min(args.n_modes, U.shape[1])
    if qx_ref is not None and qy_ref is not None:
        plot_pca_modes_2d(U, qx_ref, qy_ref, n_modes=n_modes_to_plot,
                          output_path=output_dir / 'pca_modes_2d.png')
    else:
        print("Warning: qx, qy coordinates not available, skipping 2D mode plots")
    n_modes_list = [min(5, n_modes), min(10, n_modes), min(20, n_modes), n_modes]
    n_modes_list = sorted(set(n_modes_list))  # Remove duplicates and sort
    plot_reconstructed_patterns(intensity_matrix, U, mean_intensity,
                                n_patterns=3, n_modes_list=n_modes_list,
                                q_values=q_values,
                                output_path=output_dir / 'reconstructed_patterns.png')
    
    mse_values = plot_mse_vs_modes(intensity_matrix, U, 
                                  max_modes=min(args.n_components, U.shape[1]),
                                  mean_intensity=mean_intensity,
                                  output_path=output_dir / 'mse_vs_modes.png')
    
    # Save MSE values
    with open(output_dir / 'mse_values.pkl', 'wb') as f:
        pickle.dump(mse_values, f)
    
    # Correlation analysis with physical parameters
    if metadata is not None and not metadata.empty:
        print("\n" + "="*60)
        print("Correlation Analysis with Physical Parameters")
        print("="*60)
        
        # Filter out rows with missing metadata
        # Reset index to ensure alignment
        metadata = metadata.reset_index(drop=True)
        valid_mask = ~(metadata.isnull().any(axis=1))
        if valid_mask.sum() > 0:
            alpha_clean = alpha[valid_mask, :]
            metadata_clean = metadata[valid_mask].reset_index(drop=True)
            
            # Ensure we have enough components
            n_comp_for_corr = min(args.n_modes, alpha_clean.shape[1])
            
            # 1. Correlation Matrix
            print("\n1. Calculating correlation matrix...")
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

