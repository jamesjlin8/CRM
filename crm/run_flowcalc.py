
import numpy as np
import vegas
import time
import os
from pathlib import Path
from multiprocessing import Pool
from calculations import calculate_intensity_core, orientation_probability_vectorized, integrate_adaptive
from plotting import plot_from_data_file

# --- Simulation Parameters ---
PARAMS = {
    'n_cyl': 5,
    'radius': 10.0,  # in A
    'length': 100.0,  # in A
    'background': 0.0,
    'phi0': 0.0
}

# --- Vegas Integration Parameters ---
VEGAS_PARAMS = {
    'nitn_min': 5,      # Minimum iterations (starts here, adds more if needed)
    'nitn_max': 30,     # Maximum iterations if convergence is poor
    'neval': 100000,     # Evaluations per iteration (increase if needed)
    'alpha': 0.5,       # Adaptation rate (0.5 is default)
    'rel_error_target': 0.05,  # Target relative error (5%)
    'total_cores': 40   # Total cores to use (multiprocessing Pool workers)
}

# --- Plotting Parameters ---
PLOT_PARAMS = {
    'beta': 1,              # Structure factor parameter
    'scalvolfrac': 8.577,   # Scaling volume fraction factor
    'graphingparameter': 4, # Marker size scaling factor
    'fontsize': 12,         # Font size for labels
    'fig_width': 550,       # Figure width in pixels
    'fig_height': 400,      # Figure height in pixels
    'caxis_min': None,      # Color axis minimum (log scale), None = auto
    'caxis_max': None,      # Color axis maximum (log scale), None = auto
    'dpi': 300              # Dots per inch for output
}

STRETCH_CONFIG = [
    {'stretch': 0.6,  'output_file': 'PHIO_0_St0.6_5cyl_10r_100l.dat'}
]

class Integrand:
    """
    Integrand class for vegas Monte Carlo integration.
    """
    def __init__(self, q, cos_psi, sin_psi, config):
        self.q = q
        self.cos_psi = cos_psi
        self.sin_psi = sin_psi
        self.config = config
        # Cache frequently used constants
        self.n_cyl = PARAMS['n_cyl']
        self.jacobian = (np.pi**self.n_cyl) * ((2*np.pi)**self.n_cyl)

    def __call__(self, x):
        """
        Compute integrand value for a single integration point.
        Calls numba JIT-compiled functions for fast computation.
        """
        x = np.asarray(x, dtype=np.float64)
        n_cyl = self.n_cyl
        angles = np.empty(2 * n_cyl, dtype=np.float64)

        # Scale theta from [0, pi] and phi from [0, 2*pi]
        angles[:n_cyl] = x[:n_cyl] * np.pi             # Theta values
        angles[n_cyl:] = x[n_cyl:] * (2.0 * np.pi)     # Phi values

        # Calculate the core intensity
        iq_core = calculate_intensity_core(
            angles, self.q, PARAMS['radius'], PARAMS['length'],
            n_cyl, self.cos_psi, self.sin_psi
        )
        
        # Calculate the orientation probability for each cylinder
        theta = angles[:n_cyl]
        phi = angles[n_cyl:]
        prob = orientation_probability_vectorized(theta, phi, self.config['stretch'], PARAMS['phi0'])
        
        # The jacobian for the change of variables is (pi^n_cyl) * ((2*pi)^n_cyl)
        # Multiply probabilities for all cylinders (product of array elements)
        prob_product = np.prod(prob)
        
        return iq_core * prob_product * self.jacobian


def process_single_q_vector(args):
    """
    Process a single Q-vector integration with adaptive convergence.
    This function is designed to be called by multiprocessing workers.
    
    Uses vegas with nproc=1 to avoid nested multiprocessing.
    The Pool handles all parallelism at the Q-vector level.
    
    Args:
        args: Tuple of (q_vector, config, q_index, total_q_vectors)
    
    Returns:
        Tuple of (q_index, qx, qy, pq, rel_error, iterations_used)
    """
    (qx, qy), config, q_idx, total_q = args
    
    q = np.sqrt(qx**2 + qy**2)
    cos_psi = qx / q if q > 0 else 1.0
    sin_psi = qy / q if q > 0 else 0.0
    
    # Initialize the integrand class with current parameters
    integrand_func = Integrand(q, cos_psi, sin_psi, config)
    
    n_dim = 2 * PARAMS['n_cyl']
    integ = vegas.Integrator([[0, 1]] * n_dim)

    # Use adaptive integration - starts with nitn_min, adds more if needed
    result, iterations_used = integrate_adaptive(
        integrand_func, integ,
        nitn_min=VEGAS_PARAMS['nitn_min'],
        neval=VEGAS_PARAMS['neval'],
        alpha=VEGAS_PARAMS['alpha'],
        nproc=1,
        nitn_max=VEGAS_PARAMS['nitn_max'],
        rel_error_target=VEGAS_PARAMS['rel_error_target']
    )
    
    pq = result.mean / (PARAMS['n_cyl']**2)
    rel_error = abs(result.sdev / result.mean) if result.mean != 0 else 0.0
    
    return (q_idx, qx, qy, pq, rel_error, iterations_used)


def process_stretch_config(config, q_vectors, output_dir, config_idx, total_configs):
    """
    Process all Q-vectors for a single stretch configuration.
    Uses multiprocessing Pool to parallelize over Q-vectors.
    Each worker uses adaptive Vegas integration.
    """
    config_start_time = time.time()
    print(f"\n--- Running config {config_idx+1}/{total_configs}: stretch={config['stretch']} ---")
    
    # Create subdirectory based on n_cyl for organized output
    n_cyl = PARAMS['n_cyl']
    output_subdir = output_dir / f"{n_cyl}cyl"
    output_subdir.mkdir(exist_ok=True)
    
    total_cores = VEGAS_PARAMS['total_cores']
    n_workers = min(total_cores, len(q_vectors))
    
    print(f"  Using multiprocessing Pool with {n_workers} workers")
    
    # Prepare arguments for multiprocessing
    process_args = [
        ((qx, qy), config, idx, len(q_vectors))
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
    
    # Calculate efficiency metrics
    avg_iterations = np.mean(iterations_used)
    max_iterations = VEGAS_PARAMS['nitn_max']
    
    print(f"  Convergence statistics:")
    print(f"    Relative error: mean={np.mean(rel_errors):.2%}, max={np.max(rel_errors):.2%}")
    print(f"    Iterations: mean={avg_iterations:.1f}, range=[{iterations_used.min()}, {iterations_used.max()}]")
    print(f"    Points needing extra iterations: {np.sum(iterations_used > VEGAS_PARAMS['nitn_min'])}/{len(iterations_used)}")
    print(f"    Points with rel_error > {VEGAS_PARAMS['rel_error_target']*100}%: {np.sum(rel_errors > VEGAS_PARAMS['rel_error_target'])}/{len(rel_errors)}")
    
    # Extract just the physics results for saving
    results_data = [[qx, qy, pq] for _, qx, qy, pq, _, _ in results]
    
    q_time = time.time() - start_q_time
    print(f"  Processed {len(q_vectors)} Q-vectors in {q_time:.2f} seconds")
    
    # Save the results for this configuration
    output_path = output_subdir / config['output_file']
    np.savetxt(output_path, results_data, fmt='%.7e')
    print(f"  Results saved to {output_path}")
    
    # Create plots using simulation parameters
    print(f"  Generating plots for {config['output_file']}...")
    all_params = {**PARAMS, **PLOT_PARAMS}
    plot_files = plot_from_data_file(
        output_path,
        output_dir=output_subdir,
        params=all_params,
        plot_both=True,
        stretch_val=config['stretch'],
        phi0=PARAMS['phi0']
    )
    print(f"  Generated {len(plot_files)} plot file(s)")
    
    config_time = time.time() - config_start_time
    print(f"  Config {config_idx+1} completed in {config_time:.2f} seconds ({config_time/60:.2f} minutes)")
    
    return output_path


# --- Main Program Execution ---
if __name__ == "__main__":
    start_time = time.time()
    
    # Define input/output paths
    input_dir = Path('./input')
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    
    input_file = input_dir / 'Reversed_LowQMidQHighQ_Combined.dat'
    q_vectors = np.loadtxt(input_file)
    
    print(f"\nLoaded {len(q_vectors)} Q vectors from {input_file}")

    # Process stretch configurations sequentially
    # Q-vectors are parallelized within each config using multiprocessing Pool
    for i, config in enumerate(STRETCH_CONFIG):
        process_stretch_config(config, q_vectors, output_dir, i, len(STRETCH_CONFIG))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*40}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*40}")

