
import numpy as np
import vegas
from scipy.optimize import least_squares
from pathlib import Path
import time
from collections import OrderedDict
from calculations import (
    equilibrium_integrand_kernel, 
    integrate_adaptive,
    calculate_normalization,
    apply_resolution_smearing
)

# --- File Paths ---
DATA_FILE = Path('./input/data_800k.dat')
OUTPUT_FILE = Path('./output/800k_ncyl22_fit2.dat')

# --- Parameter Definitions: (initial_guess, fit_this_parameter) ---
PARAMETERS = OrderedDict([
    ('amplitude',        (1.0,     False)),
    ('radius',           (15.0,    True)),
    ('length',           (54.0,    True)),
    ('background',       (0.07956,  False)),
    ('vol_fraction_sld', (3.0,     True)),
    ('n_cyl',            (22,      False)),
    ('sigma_r',          (2.0,     True)),
])

# --- Vegas Integration Parameters ---
VEGAS_PARAMS = {
    'nitn_min': 5,
    'nitn_max': 50,
    'neval': 100000,
    'alpha': 0.5,
    'rel_error_target': 0.01
}

# --- Parameter Bounds for Fitting ---
BOUNDS = {
    'amplitude': (0.01, 100.0),
    'radius': (1.0, 100.0),
    'length': (10.0, 500.0),
    'background': (-1.0, 10.0),
    'vol_fraction_sld': (0.01, 50.0),
    'n_cyl': (1, 100),
    'sigma_r': (0.1, 30.0),
}


class Integrand:
    """
    Integrand class for Vegas Monte Carlo integration.
    Wraps the JIT-compiled kernel with problem-specific parameters.
    """
    def __init__(self, q_array, radius, length, n_cyl, sigma_r):
        self.q_array = np.asarray(q_array, dtype=np.float64)
        self.radius = float(radius)
        self.length = float(length)
        self.n_cyl = int(n_cyl)
        self.sigma_r = float(sigma_r)
    
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return equilibrium_integrand_kernel(
            x, self.radius, self.sigma_r, self.length, 
            self.n_cyl, self.q_array
        )


def load_data(filepath):
    """
    Load experimental SANS data.
    File format: Q | Intensity | Intensity_error^2 | Q_error^2 | flag
    """
    print(f"Loading data from: {filepath}")
    data = np.loadtxt(filepath)
    q = data[:, 0]
    intensity = data[:, 1]
    sigma_intensity = np.sqrt(data[:, 2])
    sigma_q = np.sqrt(data[:, 3])
    print(f"  Loaded {len(q)} data points, Q range: [{q.min():.4f}, {q.max():.4f}]")
    return q, intensity, sigma_intensity, sigma_q


def calculate_intensity(q_data, sigma_q, params, verbose=True):
    """
    Calculate resolution-smeared scattering intensity I(q).
    
    Args:
        q_data: Experimental Q values
        sigma_q: Q resolution (sigma of Gaussian)
        params: List of parameter values in PARAMETERS order
        verbose: Print progress information
        
    Returns:
        Smeared intensity at experimental Q points
    """
    amplitude, radius, length, background, vol_fraction_sld, n_cyl, sigma_r = params
    n_cyl = int(n_cyl)
    
    if verbose:
        print(f"  Calculating: R={radius:.2f}, L={length:.1f}, N={n_cyl}, σ_r={sigma_r:.2f}")
    
    # Extended Q-grid for resolution smearing
    q_step = q_data[1] - q_data[0] if len(q_data) > 1 else q_data[0] * 0.1
    q_min = max(1e-4, q_data[0] - 8.0 * q_step)
    q_max = q_data[-1] + 4.0 * sigma_q[-1]
    q_ideal = np.unique(np.concatenate([
        np.linspace(q_min, q_max, 200),
        q_data
    ]))
    
    # Normalization
    r_sq_avg, int2 = calculate_normalization(radius, sigma_r)
    
    # Vegas integration
    n_dim = 1 + 2 * n_cyl
    integrand = Integrand(q_ideal, radius, length, n_cyl, sigma_r)
    integ = vegas.Integrator([[0, 1]] * n_dim)
    
    result, iterations_used = integrate_adaptive(
        integrand, integ,
        nitn_min=VEGAS_PARAMS['nitn_min'],
        neval=VEGAS_PARAMS['neval'],
        alpha=VEGAS_PARAMS['alpha'],
        nproc=1,
        nitn_max=VEGAS_PARAMS['nitn_max'],
        rel_error_target=VEGAS_PARAMS['rel_error_target']
    )
    
    if verbose:
        print(f"    Vegas: {iterations_used} iterations")
    
    # Extract results and apply scaling
    pq_unnormalized = np.array([r.mean for r in result])
    pq_unnormalized /= (1000.0 * n_cyl**2)
    
    pq_ideal = pq_unnormalized / int2 if int2 > 0 else pq_unnormalized
    
    intensity = vol_fraction_sld * pq_ideal
    volume_factor = np.pi * length * r_sq_avg * n_cyl * 1e-5
    intensity *= volume_factor
    intensity = amplitude * intensity + background
    
    # Resolution smearing
    intensity_smeared = apply_resolution_smearing(q_ideal, intensity, q_data, sigma_q)
    
    if verbose:
        print(f"    I(Q_min)={intensity_smeared[0]:.4f}, I(Q_max)={intensity_smeared[-1]:.4f}")
    
    return intensity_smeared


def run_fitting(q_exp, i_exp, sigma_i, sigma_q):
    """Run least-squares fitting."""
    param_names = list(PARAMETERS.keys())
    
    # Extract initial guesses and fit flags
    p0_all = [v[0] for v in PARAMETERS.values()]
    fit_flags = [v[1] for v in PARAMETERS.values()]
    
    # Get indices and values for fitted parameters
    fit_indices = [i for i, fit in enumerate(fit_flags) if fit]
    fit_names = [param_names[i] for i in fit_indices]
    p0_fit = [p0_all[i] for i in fit_indices]
    
    # Get bounds for fitted parameters
    lower_bounds = [BOUNDS[name][0] for name in fit_names]
    upper_bounds = [BOUNDS[name][1] for name in fit_names]
    
    def residual(p_fit):
        # Build full parameter list
        current_params = list(p0_all)
        for i, idx in enumerate(fit_indices):
            current_params[idx] = p_fit[i]
        
        try:
            i_model = calculate_intensity(q_exp, sigma_q, current_params, verbose=True)
            return (i_model - i_exp) / sigma_i
        except Exception as e:
            print(f"Error: {e}")
            return np.ones_like(i_exp) * 1e10
    
    print(f"\nFitting: {fit_names}")
    print(f"Initial: {p0_fit}")
    print(f"Bounds: lower={lower_bounds}, upper={upper_bounds}")
    
    # Note: Monte Carlo noise requires larger diff_step and looser tolerances
    result = least_squares(
        residual, p0_fit,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        max_nfev=10000,     # More function evaluations
        ftol=1e-4,          # Slightly tight tolerance (MC noise ~1-3%)
        xtol=1e-4,          # Slightly tight tolerance for parameter changes
        diff_step=0.05,     # 5% step for finite differences (overcomes MC noise)
        verbose=2
    )
    
    # Reconstruct full parameter list
    final_params = list(p0_all)
    for i, idx in enumerate(fit_indices):
        final_params[idx] = result.x[i]
    
    # Estimate uncertainties
    try:
        J = result.jac
        if J is not None and J.shape[0] >= J.shape[1]:
            cov = np.linalg.inv(J.T @ J)
            residuals = residual(result.x)
            chi_sq = np.sum(residuals**2)
            dof = len(residuals) - len(result.x)
            if dof > 0:
                cov *= chi_sq / dof
            perr = np.sqrt(np.diag(cov))
        else:
            perr = np.zeros(len(result.x))
            chi_sq = np.sum(residual(result.x)**2)
            dof = len(i_exp) - len(result.x)
    except Exception:
        perr = np.zeros(len(result.x))
        chi_sq = np.sum(residual(result.x)**2)
        dof = len(i_exp) - len(result.x)
    
    return final_params, fit_names, perr, chi_sq, dof, result.success


if __name__ == "__main__":
    start_time = time.time()
    
    print("=" * 60)
    print("Connected Rod Model - Equilibrium Scattering Fit")
    print("=" * 60)
    
    q_exp, i_exp, sigma_i, sigma_q = load_data(DATA_FILE)
    
    print("\nParameters:")
    for name, (val, fit) in PARAMETERS.items():
        status = "FIT" if fit else "FIXED"
        print(f"  {name:20s}: {val:10.4f}  [{status}]")
    
    print("\nVegas Settings:")
    for name, val in VEGAS_PARAMS.items():
        print(f"  {name:20s}: {val}")
    
    print("\n" + "-" * 60)
    print("Starting Optimization")
    print("-" * 60)
    
    final_params, fit_names, perr, chi_sq, dof, success = run_fitting(
        q_exp, i_exp, sigma_i, sigma_q
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not success:
        print("WARNING: Optimization may not have fully converged.")
    
    print("\nFinal Parameters:")
    param_names = list(PARAMETERS.keys())
    err_idx = 0
    for i, (name, (_, fit)) in enumerate(PARAMETERS.items()):
        if fit:
            print(f"  {name:20s}: {final_params[i]:10.4f} ± {perr[err_idx]:.4f}")
            err_idx += 1
        else:
            print(f"  {name:20s}: {final_params[i]:10.4f} (fixed)")
    
    chi_sq_red = chi_sq / dof if dof > 0 else chi_sq
    print(f"\nχ² = {chi_sq:.2f}, χ²/dof = {chi_sq_red:.2f} (dof = {dof})")
    
    print("\nGenerating final curve...")
    i_fit = calculate_intensity(q_exp, sigma_q, final_params)
    
    # Build header with fit results
    header_lines = ["Connected Rod Model - Equilibrium Fit Results"]
    header_lines.append(f"Data file: {DATA_FILE}")
    header_lines.append(f"Chi-squared: {chi_sq:.4f}, Chi-squared/dof: {chi_sq_red:.4f}, dof: {dof}")
    header_lines.append("")
    header_lines.append("Fit Parameters:")
    err_idx = 0
    for i, (name, (_, fit)) in enumerate(PARAMETERS.items()):
        if fit:
            header_lines.append(f"  {name}: {final_params[i]:.6f} +/- {perr[err_idx]:.6f}")
            err_idx += 1
        else:
            header_lines.append(f"  {name}: {final_params[i]:.6f} (fixed)")
    header_lines.append("")
    header_lines.append("Q\tI_exp\tI_fit\tsigma_I\tsigma_Q")
    header = "\n".join(header_lines)
    
    output_data = np.column_stack([q_exp, i_exp, i_fit, sigma_i, sigma_q])
    np.savetxt(OUTPUT_FILE, output_data, header=header, fmt='%.6e', delimiter='\t')
    print(f"Saved to: {OUTPUT_FILE}")
    
    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")
