#!/usr/bin/env python3
"""
Simple script to plot a FlowCalc-generated scattering pattern data file.
Usage: python plot_scattering.py [data_file] [options]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Default parameters (can be modified)
DEFAULT_PARAMS = {
    'n_cyl': 1,
    'radius': 1,  # in A
    'length': 100,  # in A
    'background': 0.0,
    'phi0': 0.0,
    'beta': 3,              # Structure factor parameter
    'scalvolfrac': 8.577,   # Scaling volume fraction factor
    'graphingparameter': 4, # Marker size scaling factor
    'fontsize': 12,         # Font size for labels
    'caxis_min': None,      # Color axis minimum (log scale), None = auto
    'caxis_max': None,      # Color axis maximum (log scale), None = auto
}

def calculate_intensity(qx, qy, p, params):
    """Calculate intensity with and without structure factor."""
    pref = params['scalvolfrac'] * np.pi * params['radius']**2 * \
           params['length'] * params['n_cyl'] * 1e-5
    
    q = np.sqrt(qx**2 + qy**2)
    inosq = pref * p + params['background']
    s = 1.0 / (1.0 + params['beta'] * p)
    iwithsq = pref * s * p + params['background']
    
    return inosq, iwithsq, s, q

def plot_scattering_pattern(qx, qy, intensity, params, title="", use_log=True):
    """Create a 2D scatter plot of scattering pattern and display it."""
    if use_log:
        plot_values = np.log10(intensity)
        plot_values = np.where(np.isfinite(plot_values), plot_values, np.nan)
    else:
        plot_values = intensity
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_points = len(qx)
    marker_size = params['graphingparameter'] * 150000 / n_points
    
    scatter = ax.scatter(qx, qy, s=marker_size, c=plot_values, 
                        cmap='jet', marker='s', edgecolors='none')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀ I(q)', fontsize=params['fontsize'])
    
    if use_log:
        # Auto-scale color axis if limits are None
        if params['caxis_min'] is None or params['caxis_max'] is None:
            valid_values = plot_values[np.isfinite(plot_values)]
            if len(valid_values) > 0:
                vmin = params['caxis_min'] if params['caxis_min'] is not None else valid_values.min()
                vmax = params['caxis_max'] if params['caxis_max'] is not None else valid_values.max()
                scatter.set_clim(vmin, vmax)
            else:
                # Fallback if no valid values
                scatter.set_clim(-1.0, 2.0)
        else:
            scatter.set_clim(params['caxis_min'], params['caxis_max'])
    
    ax.set_xlabel(r'$q_x [\AA^{-1}]$', fontsize=params['fontsize'])
    ax.set_ylabel(r'$q_y [\AA^{-1}]$', fontsize=params['fontsize'])
    ax.set_title(title, fontsize=params['fontsize'] + 2)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=params['fontsize'])
    
    # Set square, symmetric axis limits centered at (0,0)
    max_q = max(np.abs(qx).max(), np.abs(qy).max())
    ax.set_xlim(-max_q, max_q)
    ax.set_ylim(-max_q, max_q)
    
    plt.tight_layout()
    return fig

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_scattering.py <data_file> [stretch_value]")
        print("\nExample:")
        print("  python plot_scattering.py output/PHIO_0_St0.5_1cyl_1r_100l.dat 0.5")
        print("\nIf no stretch value is provided, it will be extracted from filename or default to 0.0")
        sys.exit(1)
    
    data_file = Path(sys.argv[1])
    
    if not data_file.exists():
        print(f"Error: File not found: {data_file}")
        sys.exit(1)
    
    # Parse stretch value if provided
    stretch_val = None
    if len(sys.argv) >= 3:
        try:
            stretch_val = float(sys.argv[2])
        except ValueError:
            print(f"Warning: Could not parse stretch value '{sys.argv[2]}', will extract from filename")
    
    # Extract stretch from filename if not provided
    if stretch_val is None:
        filename_stem = data_file.stem
        stretch_val = 0.0
        if 'St' in filename_stem:
            try:
                parts = filename_stem.split('St')
                if len(parts) > 1:
                    stretch_part = parts[1].split('_')[0]
                    stretch_val = float(stretch_part)
            except (ValueError, IndexError):
                pass
    
    print(f"Loading data from: {data_file}")
    print(f"Stretch value: {stretch_val}")
    
    # Load data
    try:
        data = np.loadtxt(data_file)
        qx = data[:, 0]
        qy = data[:, 1]
        p = data[:, 2]
        
        print(f"Loaded {len(qx)} data points")
        
        # Calculate intensities
        inosq, iwithsq, s, q = calculate_intensity(qx, qy, p, DEFAULT_PARAMS)
        
        # Create plots
        fig1 = plot_scattering_pattern(qx, qy, inosq, DEFAULT_PARAMS, 
                                      title=f"Scattering Pattern (No Structure Factor)\nStretch = {stretch_val}")
        
        fig2 = plot_scattering_pattern(qx, qy, iwithsq, DEFAULT_PARAMS,
                                      title=f"Scattering Pattern (With Structure Factor)\nStretch = {stretch_val}")
        
        # Display plots
        plt.show()
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

