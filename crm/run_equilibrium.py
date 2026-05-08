"""
Batch equilibrium scattering fit for Rheo-SANS 2D patterns.

Reads every equilibrium (0 s^-1) sample from ``Rheo-SANS/extracted/*_0s1_*m.dat``
(each file is a 2D detector pattern with columns ``Qx, Qy, I`` and 128x128 pixels),
combines all detector distances for a sample (1m / 3m / 12m or 13m) via
azimuthal binning into a 1D I(|Q|) with pixel-scatter uncertainties, then fits
the existing isotropic connected-rod model using Vegas Monte Carlo.
"""

import re
import numpy as np
import vegas
from scipy.optimize import least_squares
from pathlib import Path
import time
from collections import OrderedDict, defaultdict
from calculations import (
    equilibrium_integrand_kernel,
    integrate_adaptive,
    calculate_normalization,
    apply_resolution_smearing,
)

# --- File Paths ---
DATA_DIR = Path('./Rheo-SANS/extracted')
SAMPLE_GLOB = '*_0s1_*m.dat'
OUTPUT_DIR = Path('./output')
SUMMARY_FILE = OUTPUT_DIR / 'equilibrium_fit_summary.tsv'

# --- 2D -> 1D radial binning parameters ---
REDUCTION = {
    'q_min': 3e-3,        # beamstop / low-Q cutoff [A^-1]
    'q_max': 0.5,         # detector-edge / high-Q cutoff [A^-1]
    'n_bins': 80,         # log-spaced |Q| bins
    'min_pix_per_bin': 3, # drop bins with fewer pixels
}

# --- Parameter Definitions: (initial_guess, fit_this_parameter) ---
PARAMETERS = OrderedDict([
    ('amplitude',        (1.0,     False)),
    ('radius',           (20.0,    True)),
    ('length',           (100.0,    True)),
    ('background',       (0.13,  False)),
    ('vol_fraction_sld', (3.0,     True)),
    ('n_cyl',            (10,      True)),
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


# =============================================================================
# 2D Rheo-SANS data loading and reduction
# =============================================================================

# Matches the trailing "_<dist>m.dat" on Rheo-SANS filenames (e.g. "_1m.dat",
# "_3m.dat", "_12m.dat", "_13m.dat", or decimal forms like "_1p33m.dat").
_DIST_SUFFIX_RE = re.compile(r'_[0-9]+(?:p[0-9]+)?m\.dat$', re.IGNORECASE)


def discover_samples(data_dir, pattern):
    """
    Group Rheo-SANS extracted files by sample tag.

    A sample tag is the filename stem with the trailing ``_<dist>m`` removed,
    e.g. ``100mM_0s1_1m.dat`` -> ``100mM_0s1``. The three detector-distance
    files for a given sample are grouped together.

    Returns an OrderedDict mapping ``tag -> list[Path]`` sorted by tag.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(data_dir.glob(pattern)):
        name = path.name
        m = _DIST_SUFFIX_RE.search(name)
        if m:
            tag = name[:m.start()]
        else:
            tag = path.stem
        groups[tag].append(path)

    return OrderedDict((tag, sorted(paths)) for tag, paths in sorted(groups.items()))


def load_pattern_2d(filepath):
    """Load a single 2D Rheo-SANS pattern file (columns: Qx, Qy, I)."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 3:
        raise ValueError(
            f"Expected 3 columns (Qx, Qy, I) in {filepath}, got {data.shape[1]}"
        )
    return data[:, 0], data[:, 1], data[:, 2]


def reduce_2d_to_1d(qx, qy, intensity, q_min, q_max, n_bins, min_pix_per_bin):
    """
    Azimuthally bin scattered 2D pixels into a 1D I(|Q|) curve.

    For each log-spaced |Q| bin, computes:
        Q      = mean(|Q|_pix)
        I      = mean(I_pix)
        sigma_I= std(I_pix) / sqrt(N_pix)   (standard error of the mean)
        sigma_Q= bin half-width             (Gaussian-equivalent resolution)

    Pixels with non-finite or non-positive intensity and those outside
    ``[q_min, q_max]`` are dropped. Bins with fewer than
    ``min_pix_per_bin`` surviving pixels are dropped.

    NOTE: Dropping I<=0 slightly biases high-Q where Poisson statistics leave
    many near-zero pixels; acceptable here because the fitted model's
    background captures the floor.
    """
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    q_mag = np.sqrt(qx * qx + qy * qy)
    mask = (
        np.isfinite(intensity)
        & (intensity > 0)
        & (q_mag >= q_min)
        & (q_mag <= q_max)
    )
    q_mag = q_mag[mask]
    intensity = intensity[mask]

    if q_mag.size == 0:
        raise ValueError("No valid pixels remain after masking.")

    edges = np.geomspace(q_min, q_max, n_bins + 1)

    n_per_bin,     _ = np.histogram(q_mag, bins=edges)
    sum_q,         _ = np.histogram(q_mag, bins=edges, weights=q_mag)
    sum_i,         _ = np.histogram(q_mag, bins=edges, weights=intensity)
    sum_i2,        _ = np.histogram(q_mag, bins=edges, weights=intensity * intensity)

    keep = n_per_bin >= min_pix_per_bin
    if not np.any(keep):
        raise ValueError(
            f"No bins have >= {min_pix_per_bin} pixels; try increasing n_bins "
            "or broadening [q_min, q_max]."
        )

    n = n_per_bin[keep].astype(np.float64)
    q_bin = sum_q[keep] / n
    i_bin = sum_i[keep] / n
    var_bin = np.clip(sum_i2[keep] / n - i_bin * i_bin, 0.0, None)
    # Sample variance (unbiased: factor n/(n-1)); for n==1 falls back to std=0.
    with np.errstate(divide='ignore', invalid='ignore'):
        std_bin = np.where(n > 1, np.sqrt(var_bin * n / (n - 1.0)), 0.0)
    sigma_i_bin = std_bin / np.sqrt(n)

    lo = edges[:-1][keep]
    hi = edges[1:][keep]
    sigma_q_bin = 0.5 * (hi - lo)

    # Ensure a strictly positive sigma_I (avoid /0 in residuals). Use the median
    # of positive SEMs as a floor for degenerate bins.
    pos = sigma_i_bin > 0
    if np.any(pos):
        floor = np.median(sigma_i_bin[pos]) * 1e-3
        sigma_i_bin = np.where(sigma_i_bin > 0, sigma_i_bin, floor)
    else:
        sigma_i_bin = np.full_like(sigma_i_bin, 1e-6)

    order = np.argsort(q_bin)
    return q_bin[order], i_bin[order], sigma_i_bin[order], sigma_q_bin[order]


def load_and_reduce_sample(paths, *, q_min, q_max, n_bins, min_pix_per_bin):
    """Load all detector-distance files for one sample and azimuthally bin them."""
    qx_all, qy_all, i_all = [], [], []
    for p in paths:
        qx, qy, intensity = load_pattern_2d(p)
        qx_all.append(qx)
        qy_all.append(qy)
        i_all.append(intensity)
        print(f"  {p.name}: {len(qx)} pixels")

    qx = np.concatenate(qx_all)
    qy = np.concatenate(qy_all)
    intensity = np.concatenate(i_all)

    q, i, sigma_i, sigma_q = reduce_2d_to_1d(
        qx, qy, intensity,
        q_min=q_min, q_max=q_max,
        n_bins=n_bins, min_pix_per_bin=min_pix_per_bin,
    )
    print(f"  -> {len(q)} bins, Q range: [{q.min():.4f}, {q.max():.4f}]")
    return q, i, sigma_i, sigma_q


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
    
    # Resolution smearing (skip if all sigma_q are zero)
    if np.any(sigma_q > 0):
        intensity_smeared = apply_resolution_smearing(q_ideal, intensity, q_data, sigma_q)
    else:
        intensity_smeared = np.interp(q_data, q_ideal, intensity)
    
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


# =============================================================================
# Per-sample driver and batch I/O
# =============================================================================

def _fit_output_path(tag):
    return OUTPUT_DIR / f"{tag}_equilibrium_fit.dat"


def fit_one_sample(tag, paths):
    """
    Reduce the 3 detector distances for one sample, run the fit, and save
    the per-sample output file.

    Returns a dict of result fields (one row for the summary TSV).
    """
    t0 = time.time()
    print("\n" + "=" * 60)
    print(f"Sample: {tag}  ({len(paths)} files)")
    print("=" * 60)

    q_exp, i_exp, sigma_i, sigma_q = load_and_reduce_sample(paths, **REDUCTION)

    print("\nParameters:")
    for name, (val, fit) in PARAMETERS.items():
        status = "FIT" if fit else "FIXED"
        print(f"  {name:20s}: {val:10.4f}  [{status}]")

    print("\n" + "-" * 60)
    print("Starting Optimization")
    print("-" * 60)

    final_params, fit_names, perr, chi_sq, dof, success = run_fitting(
        q_exp, i_exp, sigma_i, sigma_q
    )

    print("\n" + "=" * 60)
    print(f"RESULTS [{tag}]")
    print("=" * 60)

    if not success:
        print("WARNING: Optimization may not have fully converged.")

    err_idx = 0
    print("\nFinal Parameters:")
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

    # Per-sample output: same header style as before, plus the list of input files.
    out_path = _fit_output_path(tag)
    header_lines = ["Connected Rod Model - Equilibrium Fit Results"]
    header_lines.append(f"Sample: {tag}")
    header_lines.append("Data files:")
    for p in paths:
        header_lines.append(f"  {p}")
    header_lines.append(
        f"Chi-squared: {chi_sq:.4f}, Chi-squared/dof: {chi_sq_red:.4f}, dof: {dof}"
    )
    header_lines.append("")
    header_lines.append("Fit Parameters:")
    err_idx = 0
    for i, (name, (_, fit)) in enumerate(PARAMETERS.items()):
        if fit:
            header_lines.append(
                f"  {name}: {final_params[i]:.6f} +/- {perr[err_idx]:.6f}"
            )
            err_idx += 1
        else:
            header_lines.append(f"  {name}: {final_params[i]:.6f} (fixed)")
    header_lines.append("")
    header_lines.append("Q\tI_exp\tI_fit\tsigma_I\tsigma_Q")
    header = "\n".join(header_lines)

    output_data = np.column_stack([q_exp, i_exp, i_fit, sigma_i, sigma_q])
    np.savetxt(out_path, output_data, header=header, fmt='%.6e', delimiter='\t')
    print(f"Saved to: {out_path}")

    elapsed = time.time() - t0
    print(f"Sample time: {elapsed:.1f} s")

    # Build one summary row, keyed by parameter name, with _err for fitted ones.
    row: dict[str, object] = {'sample': tag}
    err_idx = 0
    for i, (name, (_, fit)) in enumerate(PARAMETERS.items()):
        row[name] = final_params[i]
        row[f"{name}_err"] = perr[err_idx] if fit else 0.0
        if fit:
            err_idx += 1
    row['chi_sq'] = chi_sq
    row['chi_sq_red'] = chi_sq_red
    row['dof'] = dof
    row['n_points'] = len(q_exp)
    row['success'] = bool(success)
    row['elapsed_s'] = elapsed
    return row


def write_summary_tsv(path, rows):
    """Write one row per sample to a tab-separated summary file."""
    if not rows:
        print("No successful fits; skipping summary TSV.")
        return

    param_names = list(PARAMETERS.keys())
    columns = ['sample']
    for name in param_names:
        columns.append(name)
        columns.append(f"{name}_err")
    columns += ['chi_sq', 'chi_sq_red', 'dof', 'n_points', 'success', 'elapsed_s']

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\t'.join(columns) + '\n')
        for row in rows:
            vals = []
            for c in columns:
                v = row.get(c, '')
                if isinstance(v, float):
                    vals.append(f"{v:.6e}")
                else:
                    vals.append(str(v))
            fh.write('\t'.join(vals) + '\n')
    print(f"\nSummary written to: {path}")


if __name__ == "__main__":
    start_time = time.time()

    print("=" * 60)
    print("Connected Rod Model - Batch Equilibrium Scattering Fit")
    print("  (2D Rheo-SANS patterns -> radial average -> fit)")
    print("=" * 60)
    print(f"Data directory:  {DATA_DIR}")
    print(f"Sample glob:     {SAMPLE_GLOB}")
    print(f"Reduction:       {REDUCTION}")

    print("\nVegas Settings:")
    for name, val in VEGAS_PARAMS.items():
        print(f"  {name:20s}: {val}")

    samples = discover_samples(DATA_DIR, SAMPLE_GLOB)
    if not samples:
        raise SystemExit(
            f"No files matching {SAMPLE_GLOB!r} under {DATA_DIR}"
        )

    print(f"\nDiscovered {len(samples)} sample(s):")
    for tag, paths in samples.items():
        print(f"  {tag}: {[p.name for p in paths]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for tag, paths in samples.items():
        try:
            summary_rows.append(fit_one_sample(tag, paths))
        except Exception as e:
            print(f"\n[{tag}] FAILED: {e!r}")

    write_summary_tsv(SUMMARY_FILE, summary_rows)

    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")
    print(f"Fit {len(summary_rows)}/{len(samples)} samples successfully.")
