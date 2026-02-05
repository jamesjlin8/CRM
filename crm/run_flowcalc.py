
import argparse
import numpy as np
import vegas
import time
from pathlib import Path
from multiprocessing import Pool
from calculations import (
    flow_integrand_kernel,
    flow_integrand_kernel_polydisperse,
    integrate_adaptive,
    calculate_normalization,
    compute_cates_weights
)

# Vegas integration parameters
VEGAS_PARAMS = {
    'nitn_min': 1,          # Minimum iterations (starts here, adds more if needed)
    'nitn_max': 100,         # Maximum iterations if convergence is poor
    'neval': 200000,        # Evaluations per iteration (increase if needed)
    'alpha': 0.5,           # Adaptation rate (0.5 is default)
    'rel_error_target': 0.05,  # Target relative error (5%)
    'total_cores': 40       # Total cores to use (multiprocessing Pool workers)
}


class Integrand:
    """
    Integrand class for vegas Monte Carlo integration.
    """
    def __init__(self, q, cos_psi, sin_psi, config, params, n_cyl):
        self.q = q
        self.cos_psi = cos_psi
        self.sin_psi = sin_psi
        self.config = config
        self.n_cyl = int(n_cyl)
        self.radius = float(params['radius'])
        self.length = float(params['length'])
        self.phi0 = float(params['phi0'])
        self.sigma_r = float(params['sigma_r'])

    def __call__(self, x):
        """
        Compute integrand value for a single integration point.
        Calls numba JIT-compiled functions for fast computation.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.sigma_r > 0.0:
            return flow_integrand_kernel_polydisperse(
                x, self.q, self.radius, self.sigma_r, self.length,
                self.n_cyl, self.cos_psi, self.sin_psi,
                self.config['stretch'], self.phi0
            )
        return flow_integrand_kernel(
            x, self.q, self.radius, self.length,
            self.n_cyl, self.cos_psi, self.sin_psi,
            self.config['stretch'], self.phi0
        )


def _format_float(value):
    return f"{value:.6g}"


def _build_stretch_config(stretch_values, params, poly_config):
    configs = []
    if poly_config.get('use_cates'):
        lc_label = _format_float(poly_config['lc'])
        n_label = f"{lc_label}cyl"
        suffix = f"_nmax{poly_config['n_cyl_max']}"
    else:
        n_label = f"{params['n_cyl']}cyl"
        suffix = ""
    r_label = f"{_format_float(params['radius'])}r"
    l_label = f"{_format_float(params['length'])}l"
    sigma_label = ""
    if params['sigma_r'] > 0:
        sigma_label = f"_sigmar{_format_float(params['sigma_r'])}"
    for stretch in stretch_values:
        stretch_str = _format_float(stretch)
        phi0_str = _format_float(params['phi0'])
        output_file = f"PHIO_{phi0_str}_St{stretch_str}_{n_label}_{r_label}_{l_label}{sigma_label}{suffix}.dat"
        configs.append({'stretch': float(stretch), 'output_file': output_file})
    return configs


def _build_output_subdir(params, poly_config):
    if poly_config.get('use_cates'):
        lc_label = _format_float(poly_config['lc'])
        base = f"{lc_label}cyl"
    else:
        base = f"{params['n_cyl']}cyl"
    r_label = _format_float(params['radius'])
    l_label = _format_float(params['length'])
    base = f"{base}_{r_label}r_{l_label}l"
    if params['sigma_r'] > 0.0:
        base += f"_sigmar{_format_float(params['sigma_r'])}"
    if poly_config.get('use_cates'):
        base += f"_nmax{poly_config['n_cyl_max']}"
    return base


def _parse_stretch_values(values):
    parsed = []
    for value in values:
        if isinstance(value, (int, float)):
            parsed.append(float(value))
            continue
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(float(part))
    return parsed


def process_single_q_vector(args):
    """
    Process a single Q-vector integration with adaptive convergence.
    This function is designed to be called by multiprocessing workers.
    
    Uses vegas with nproc=1 to avoid nested multiprocessing.
    The Pool handles all parallelism at the Q-vector level.
    
    Args:
        args: Tuple of (q_vector, config, q_index, total_q_vectors, params, poly_config)
    
    Returns:
        Tuple of (q_index, qx, qy, pq, rel_error, iterations_used)
    """
    (qx, qy), config, q_idx, total_q, params, poly_config = args
    
    q = np.sqrt(qx**2 + qy**2)
    cos_psi = qx / q if q > 0 else 1.0
    sin_psi = qy / q if q > 0 else 0.0
    
    sigma_r = float(params['sigma_r'])
    int2 = None
    if sigma_r > 0.0:
        _, int2 = calculate_normalization(params['radius'], sigma_r)

    def integrate_for_n(n_cyl):
        integrand_func = Integrand(q, cos_psi, sin_psi, config, params, n_cyl)
        n_dim = 1 + 2 * n_cyl if sigma_r > 0.0 else 2 * n_cyl
        integ = vegas.Integrator([[0, 1]] * n_dim)
        result, iterations_used = integrate_adaptive(
            integrand_func, integ,
            nitn_min=VEGAS_PARAMS['nitn_min'],
            neval=VEGAS_PARAMS['neval'],
            alpha=VEGAS_PARAMS['alpha'],
            nproc=1,
            nitn_max=VEGAS_PARAMS['nitn_max'],
            rel_error_target=VEGAS_PARAMS['rel_error_target']
        )
        pq_local = result.mean / (n_cyl**2)
        if int2 is not None and int2 > 0.0:
            pq_local /= int2
        rel_error_local = abs(result.sdev / result.mean) if result.mean != 0 else 0.0
        return pq_local, rel_error_local, iterations_used

    if poly_config.get('use_cates'):
        pq_sum = 0.0
        rel_error_max = 0.0
        iterations_max = 0
        n_values = poly_config['n_values']
        weights = poly_config['weights']
        n_weighted_sum = poly_config['n_weighted_sum']
        for n_cyl, weight in zip(n_values, weights):
            pq_n, rel_error_n, iterations_n = integrate_for_n(int(n_cyl))
            pq_sum += pq_n * weight * n_cyl
            rel_error_max = max(rel_error_max, rel_error_n)
            iterations_max = max(iterations_max, iterations_n)
        pq = pq_sum / n_weighted_sum if n_weighted_sum > 0.0 else 0.0
        return (q_idx, qx, qy, pq, rel_error_max, iterations_max)

    pq, rel_error, iterations_used = integrate_for_n(int(params['n_cyl']))
    return (q_idx, qx, qy, pq, rel_error, iterations_used)


def process_stretch_config(config, q_vectors, output_dir, config_idx, total_configs, params, poly_config):
    """
    Process all Q-vectors for a single stretch configuration.
    Uses multiprocessing Pool to parallelize over Q-vectors.
    Each worker uses adaptive Vegas integration.
    """
    config_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Config {config_idx+1}/{total_configs}: stretch={config['stretch']}")
    print(f"  Output: {config['output_file']}")
    
    # Create subdirectory based on n_cyl for organized output
    output_subdir = output_dir / _build_output_subdir(params, poly_config)
    output_subdir.mkdir(exist_ok=True)
    
    total_cores = VEGAS_PARAMS['total_cores']
    n_workers = min(total_cores, len(q_vectors))
    
    print(f"  Using {n_workers} workers")
    
    # Prepare arguments for multiprocessing
    process_args = [
        ((qx, qy), config, idx, len(q_vectors), params, poly_config)
        for idx, (qx, qy) in enumerate(q_vectors)
    ]
    
    # Process Q-vectors in parallel
    start_q_time = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_single_q_vector, process_args)
    
    # Sort results by original Q-vector index to maintain order
    results.sort(key=lambda x: x[0])
    
    # Extract convergence statistics
    rel_errors = np.array([r[4] for r in results])
    iterations_used = np.array([r[5] for r in results])
    
    print(f"  Convergence: rel_error mean={np.mean(rel_errors):.2%}, max={np.max(rel_errors):.2%}")
    print(f"  Iterations: mean={np.mean(iterations_used):.1f}, range=[{iterations_used.min()}, {iterations_used.max()}]")
    
    # Extract just the physics results for saving
    results_data = [[qx, qy, pq] for _, qx, qy, pq, _, _ in results]
    
    q_time = time.time() - start_q_time
    print(f"  Processed {len(q_vectors)} Q-vectors in {q_time:.2f} seconds")
    
    # Save the results for this configuration
    output_path = output_subdir / config['output_file']
    np.savetxt(output_path, results_data, fmt='%.7e')
    print(f"  Saved: {output_path}")
    
    config_time = time.time() - config_start_time
    print(f"  Config {config_idx+1} completed in {config_time:.2f} seconds ({config_time/60:.2f} minutes)")
    print(f"{'='*60}")
    
    return output_path


# --- Main Program Execution ---
if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Connected Rod Model flow calculation")
    parser.add_argument("--n-cyl", type=int, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--phi0", type=float, default=0.0)
    parser.add_argument("--sigma-r", type=float, default=0.0)
    parser.add_argument("--n-cyl-max", type=int, default=0)
    parser.add_argument(
        "--stretch",
        type=str,
        nargs="+",
        default=[
            "0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45",
            "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"
        ]
    )
    parser.add_argument("--input-file", type=str, default="input/Reversed_LowQMidQHighQ_Combined.dat")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    params = {
        'n_cyl': int(args.n_cyl),
        'radius': float(args.radius),
        'length': float(args.length),
        'phi0': float(args.phi0),
        'sigma_r': float(args.sigma_r),
        'lc': float(args.n_cyl),
        'n_cyl_max': int(args.n_cyl_max)
    }

    n_values, weights, n_weighted_sum = compute_cates_weights(
        params['lc'], params['n_cyl_max']
    )
    poly_config = {
        'use_cates': n_values is not None,
        'lc': params['lc'],
        'n_cyl_max': params['n_cyl_max'],
        'n_values': n_values,
        'weights': weights,
        'n_weighted_sum': n_weighted_sum
    }

    stretch_values = _parse_stretch_values(args.stretch)
    stretch_config = _build_stretch_config(stretch_values, params, poly_config)

    # Define input/output paths
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    q_vectors = np.loadtxt(input_file)
    print(f"Loaded {len(q_vectors)} Q vectors from {input_file}")
    print("Fixed parameters:")
    print(f"  radius={params['radius']} A, length={params['length']} A, phi0={params['phi0']}")
    if poly_config['use_cates']:
        print(f"  Cates chain-length distribution: LC={params['lc']} (from n_cyl), Nmax={params['n_cyl_max']}")
        print("  n_cyl is used as the mean (LC) when n_cyl_max > 0")
    else:
        print(f"  n_cyl={params['n_cyl']}")
    if params['sigma_r'] > 0.0:
        print(f"  radius polydispersity: sigma_r={params['sigma_r']}")

    # Process stretch configurations sequentially
    # Q-vectors are parallelized within each config using multiprocessing Pool
    for i, config in enumerate(stretch_config):
        process_stretch_config(config, q_vectors, output_dir, i, len(stretch_config), params, poly_config)

    print(f"\n{'='*40}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"{'='*40}")

