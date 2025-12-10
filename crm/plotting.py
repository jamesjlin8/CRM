
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_intensity(qx, qy, p, params):
    """Calculate intensity with and without structure factor."""
    pref = params['scalvolfrac'] * np.pi * params['radius']**2 * \
           params['length'] * params['n_cyl'] * 1e-5
    
    q = np.sqrt(qx**2 + qy**2)
    inosq = pref * p + params['background']
    s = 1.0 / (1.0 + params['beta'] * p)
    iwithsq = pref * s * p + params['background']
    
    return inosq, iwithsq, s, q


def plot_scattering_pattern(qx, qy, intensity, output_path, params, use_log=True):
    """Create a 2D scatter plot of scattering pattern."""
    if use_log:
        plot_values = np.log10(intensity)
        plot_values = np.where(np.isfinite(plot_values), plot_values, np.nan)
    else:
        plot_values = intensity
    
    fig, ax = plt.subplots(figsize=(params['fig_width']/100, 
                                    params['fig_height']/100), 
                          dpi=params['dpi'])
    
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
                # Fallback if no valid values
                scatter.set_clim(-1.0, 2.0)
        else:
            scatter.set_clim(params['caxis_min'], params['caxis_max'])
    
    ax.set_xlabel(r'$q_x [\AA^{-1}]$', fontsize=params['fontsize'])
    ax.set_ylabel(r'$q_y [\AA^{-1}]$', fontsize=params['fontsize'])
    ax.set_aspect('equal')
    ax.tick_params(labelsize=params['fontsize'])
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=params['dpi'], format='pdf', 
                bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


def plot_from_data_file(data_file, output_dir=None, params=None, 
                        plot_both=True, stretch_val=None, phi0=None):
    """Read data file and create scattering pattern plots."""
    if params is None:
        raise ValueError("params must be provided")
    
    data_file = Path(data_file)
    data = np.loadtxt(data_file)
    qx = data[:, 0]
    qy = data[:, 1]
    p = data[:, 2]
    
    inosq, iwithsq, s, q = calculate_intensity(qx, qy, p, params)
    
    if output_dir is None:
        output_dir = data_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
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
    
    n_cyl = params['n_cyl']
    radius = params['radius']
    length = params['length']
    beta = params['beta']
    phi0_val = phi0 if phi0 is not None else 0.0
    
    n_cyl_str = f"{int(round(n_cyl))}"
    radius_str = f"{int(round(radius))}"
    length_str = f"{int(round(length))}"
    beta_str = f"{beta:.2f}".rstrip('0').rstrip('.')
    if not beta_str or beta_str == '.':
        beta_str = '0'
    stretch_str = f"{stretch_val:.2f}".rstrip('0').rstrip('.')
    if not stretch_str or stretch_str == '.':
        stretch_str = '0'
    
    output_files = []
    
    if plot_both:
        output_path_inosq = output_dir / f'Phi0_St{stretch_str}_{n_cyl_str}cyl_{radius_str}r_{length_str}l_InoSq.pdf'
        plot_scattering_pattern(qx, qy, inosq, output_path_inosq, params)
        output_files.append(output_path_inosq)
    
    output_path_iwithsq = output_dir / f'Phi0_St{stretch_str}_{n_cyl_str}cyl_{radius_str}r_{length_str}l_{beta_str}B_IwithSq.pdf'
    plot_scattering_pattern(qx, qy, iwithsq, output_path_iwithsq, params)
    output_files.append(output_path_iwithsq)
    
    return output_files

