
import numpy as np
from numba import jit
from scipy.interpolate import interp1d

# =============================================================================
# Core Physics Functions (JIT-compiled)
# =============================================================================

@jit(nopython=True, fastmath=True)
def numba_j1(x):
    """
    Numba-compatible Bessel function of the first kind, order 1.
    """
    if x == 0.0:
        return 0.0
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + \
            y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + \
            y * (89965.84861 + y * (227.1267164 + y * 1.0))))
        return ans1 / ans2
    else:
        z = 8.0 / ax
        y = z * z
        xx = ax - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + \
            y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.200249054e-3 + \
            y * (0.8449199096e-5 + y * (-0.88228987e-6)))
        ans = np.sqrt(0.636619772 / ax) * (np.cos(xx) * ans1 - z * np.sin(xx) * ans2)
        return ans if x > 0 else -ans


@jit(nopython=True, fastmath=True)
def cylinder_form_factor(gam, q, r, l):
    """
    Calculates the form factor of a single cylinder.
    """
    if abs(gam - (np.pi / 2.0)) < 1e-9:
        qr = q * r
        return 2.0 * numba_j1(qr) / qr if abs(qr) > 1e-9 else 1.0

    term1 = q * r * np.sin(gam)
    term2 = (q * l / 2.0) * np.cos(gam)
    
    factor1 = 2.0 * numba_j1(term1) / term1 if abs(term1) > 1e-9 else 1.0
    factor2 = np.sin(term2) / term2 if abs(term2) > 1e-9 else 1.0
    
    return factor1 * factor2


@jit(nopython=True, fastmath=True)
def calculate_intensity_core(angles, q, r, l, n_cyl, cos_psi, sin_psi):
    """
    The performance-critical inner loop for calculating scattering intensity.
    Returns iq * sin_theta product (orientation probability applied separately).
    Used by both equilibrium and flow calculations.
    """
    theta = angles[:n_cyl]
    phi = angles[n_cyl:]

    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Calculate gamma for each cylinder with numerical safety clipping
    val = cos_psi * cos_phi * sin_theta + sin_psi * sin_theta * sin_phi
    val = np.clip(val, -1.0, 1.0)
    gamma = np.arccos(val)
    
    # Calculate form factor for each cylinder
    f1 = np.empty(n_cyl, dtype=np.float64)
    for i in range(n_cyl):
        f1[i] = cylinder_form_factor(gamma[i], q, r, l)
        
    s_x = np.zeros(n_cyl, dtype=np.float64)
    s_y = np.zeros(n_cyl, dtype=np.float64)
    q_dot_s = np.zeros(n_cyl, dtype=np.float64)

    for j in range(1, n_cyl):
        s_x[j] = s_x[j-1] + (cos_phi[j] * sin_theta[j] + cos_phi[j-1] * sin_theta[j-1])
        s_y[j] = s_y[j-1] + (sin_phi[j] * sin_theta[j] + sin_phi[j-1] * sin_theta[j-1])
        q_dot_s[j] = q * l / 2.0 * (cos_psi * s_x[j] + sin_psi * s_y[j])

    tem1 = f1[0]
    tem2 = 0.0
    for k in range(1, n_cyl):
        tem1 += f1[k] * np.cos(q_dot_s[k])
        tem2 += f1[k] * np.sin(q_dot_s[k])
        
    # Final intensity calculation
    iq = tem1**2 + tem2**2
    
    # Probability factor (sin_theta product)
    prob_factor = np.prod(sin_theta)
        
    return iq * prob_factor


@jit(nopython=True, fastmath=True)
def orientation_probability_vectorized(theta_array, phi_array, stretch, phi0):
    """
    Vectorized orientation probability for flow calculations.
    
    Args:
        theta_array: Array of theta angles (length n_cyl)
        phi_array: Array of phi angles (length n_cyl)
        stretch: Stretch parameter
        phi0: Phi0 parameter
        
    Returns:
        Array of probabilities (length n_cyl)
    """
    n = len(theta_array)
    prob_array = np.empty(n, dtype=np.float64)
    
    # Cap stretch value to prevent overflow when stretch approaches 1
    stretch_capped = min(stretch, 0.999999)

    if stretch_capped <= 1.0e-9:
        # Uniform distribution case
        prob_val = 1.0 / (4.0 * np.pi)
        for i in range(n):
            prob_array[i] = prob_val
        return prob_array

    # Numerically stable calculation
    X_num = 3.0 * stretch_capped - stretch_capped**3
    X_den = 1.0 - stretch_capped**2
    X = X_num / X_den
    norm_factor = 4.0 * np.pi * (1.0 / X) * np.sinh(X)
    
    # Compute probability for each cylinder
    for i in range(n):
        prob_unnormalized = np.exp(X * np.cos(phi_array[i] - phi0) * np.sin(theta_array[i]))
        prob_array[i] = prob_unnormalized / norm_factor
    
    return prob_array


@jit(nopython=True, fastmath=True)
def flow_integrand_kernel(x, q, radius, length, n_cyl, cos_psi, sin_psi, stretch, phi0):
    """
    JIT-compiled integrand kernel for flow calculations (monodisperse radius).

    Integration variables (in [0,1]):
        x[0:n_cyl]: theta angles
        x[n_cyl:]: phi angles
    """
    angles = np.empty(2 * n_cyl)
    angles[:n_cyl] = x[:n_cyl] * np.pi
    angles[n_cyl:] = x[n_cyl:] * (2.0 * np.pi)

    iq_core = calculate_intensity_core(
        angles, q, radius, length, n_cyl, cos_psi, sin_psi
    )

    theta = angles[:n_cyl]
    phi = angles[n_cyl:]
    prob = orientation_probability_vectorized(theta, phi, stretch, phi0)

    prob_product = 1.0
    for i in range(n_cyl):
        prob_product *= prob[i]

    angle_jacobian = (np.pi ** n_cyl) * ((2.0 * np.pi) ** n_cyl)
    return iq_core * prob_product * angle_jacobian


@jit(nopython=True, fastmath=True)
def flow_integrand_kernel_polydisperse(
    x, q, r_avg, sigma_r, length, n_cyl, cos_psi, sin_psi, stretch, phi0
):
    """
    JIT-compiled integrand kernel for flow calculations with radius polydispersity.

    Integration variables (in [0,1]):
        x[0]: radius (scaled)
        x[1:n_cyl+1]: theta angles
        x[n_cyl+1:]: phi angles
    """
    r_min = max(0.0, r_avg - 4.0 * sigma_r)
    r_max = r_avg + 4.0 * sigma_r
    radius = r_min + x[0] * (r_max - r_min)

    angles = np.empty(2 * n_cyl)
    angles[:n_cyl] = x[1:1 + n_cyl] * np.pi
    angles[n_cyl:] = x[1 + n_cyl:] * (2.0 * np.pi)

    iq_core = calculate_intensity_core(
        angles, q, radius, length, n_cyl, cos_psi, sin_psi
    )

    theta = angles[:n_cyl]
    phi = angles[n_cyl:]
    prob = orientation_probability_vectorized(theta, phi, stretch, phi0)

    prob_product = 1.0
    for i in range(n_cyl):
        prob_product *= prob[i]

    diff = radius - r_avg
    radius_prob = radius * radius * np.exp(-diff * diff / (2.0 * sigma_r * sigma_r))

    radius_jacobian = r_max - r_min
    angle_jacobian = (np.pi ** n_cyl) * ((2.0 * np.pi) ** n_cyl)
    jacobian = radius_jacobian * angle_jacobian

    return iq_core * prob_product * radius_prob * jacobian


@jit(nopython=True, fastmath=True)
def scattering_kernel_vectorized(angles, radius, length, n_cyl, q_array):
    """
    Calculate scattering intensity for all Q-values simultaneously.
    For equilibrium (isotropic), uses cos_psi=1.0, sin_psi=0.0.
    
    Args:
        angles: Array of [theta_1, ..., theta_n, phi_1, ..., phi_n]
        radius: Cylinder radius
        length: Cylinder length (Kuhn length)
        n_cyl: Number of connected cylinders
        q_array: Array of Q values to calculate
        
    Returns:
        Array of I(Q) values
    """
    n_q = len(q_array)
    cos_psi = 1.0
    sin_psi = 0.0
    
    theta = angles[:n_cyl]
    phi = angles[n_cyl:]
    
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Calculate gamma for each cylinder
    val = cos_psi * cos_phi * sin_theta + sin_psi * sin_theta * sin_phi
    val = np.clip(val, -1.0, 1.0)
    gamma = np.arccos(val)
    
    # Pre-compute position vectors
    s_x = np.zeros(n_cyl)
    s_y = np.zeros(n_cyl)
    for j in range(1, n_cyl):
        s_x[j] = s_x[j-1] + cos_phi[j] * sin_theta[j] + cos_phi[j-1] * sin_theta[j-1]
        s_y[j] = s_y[j-1] + sin_phi[j] * sin_theta[j] + sin_phi[j-1] * sin_theta[j-1]
    
    # Calculate intensity for each Q value
    iq_array = np.zeros(n_q)
    
    for iq in range(n_q):
        q = q_array[iq]
        
        # Form factor for each cylinder
        f1 = np.empty(n_cyl)
        for i in range(n_cyl):
            gam = gamma[i]
            if abs(gam - 1.5707963267948966) < 1e-9:  # pi/2
                qr = q * radius
                f1[i] = 2.0 * numba_j1(qr) / qr if abs(qr) > 1e-9 else 1.0
            else:
                term1 = q * radius * np.sin(gam)
                term2 = (q * length / 2.0) * np.cos(gam)
                factor1 = 2.0 * numba_j1(term1) / term1 if abs(term1) > 1e-9 else 1.0
                factor2 = np.sin(term2) / term2 if abs(term2) > 1e-9 else 1.0
                f1[i] = factor1 * factor2
        
        # Sum with phase factors
        tem1 = f1[0]
        tem2 = 0.0
        for k in range(1, n_cyl):
            q_dot_s = q * length / 2.0 * (cos_psi * s_x[k] + sin_psi * s_y[k])
            tem1 += f1[k] * np.cos(q_dot_s)
            tem2 += f1[k] * np.sin(q_dot_s)
        
        iq_array[iq] = tem1 * tem1 + tem2 * tem2
    
    prob_factor = np.prod(sin_theta)
    return iq_array * prob_factor


@jit(nopython=True, fastmath=True)
def equilibrium_integrand_kernel(x, r_avg, sigma_r, length, n_cyl, q_array):
    """
    JIT-compiled integrand kernel for equilibrium Vegas integration.
    Calculates scattering for all Q-values with radius polydispersity.
    
    Integration variables (in [0,1]):
        x[0]: radius (scaled)
        x[1:n_cyl+1]: theta angles
        x[n_cyl+1:]: phi angles
        
    Returns:
        Array of integrand values for each Q
    """
    # Transform radius
    r_min = max(0.0, r_avg - 4.0 * sigma_r)
    r_max = r_avg + 4.0 * sigma_r
    radius = r_min + x[0] * (r_max - r_min)
    
    # Transform angles
    angles = np.empty(2 * n_cyl)
    angles[:n_cyl] = x[1:1+n_cyl] * np.pi
    angles[n_cyl:] = x[1+n_cyl:] * (2.0 * np.pi)
    
    # Calculate scattering for all Q values
    iq = scattering_kernel_vectorized(angles, radius, length, n_cyl, q_array)
    
    # Uniform probability: (1/4*pi)^n_cyl
    prob0 = 1.0 / (4.0 * np.pi)
    prob_factor = prob0 ** n_cyl
    
    # Radius probability * r^2
    diff = radius - r_avg
    radius_prob = radius * radius * np.exp(-diff * diff / (2.0 * sigma_r * sigma_r))
    
    # Jacobian
    radius_jacobian = r_max - r_min
    angle_jacobian = (np.pi ** n_cyl) * ((2.0 * np.pi) ** n_cyl)
    jacobian = radius_jacobian * angle_jacobian
    
    # 1000x for numerical stability
    return iq * prob_factor * radius_prob * jacobian * 1000.0


# =============================================================================
# Utility Functions
# =============================================================================

def _get_max_rel_error(result):
    """
    Get maximum relative error from Vegas result.
    Handles both scalar RAvg and vector RAvgArray results.
    """
    # Check if result is iterable (RAvgArray for vectorized integrands)
    try:
        # Vector result - compute max relative error across all components
        rel_errors = []
        for r in result:
            if r.mean != 0:
                rel_errors.append(abs(r.sdev / r.mean))
        return max(rel_errors) if rel_errors else float('inf')
    except TypeError:
        # Scalar result
        if result.mean != 0:
            return abs(result.sdev / result.mean)
        return float('inf')


def integrate_adaptive(integrand, integ, nitn_min, neval, alpha, nproc, 
                      nitn_max=20, rel_error_target=0.15):
    """
    Adaptive Vegas integration: starts with nitn_min, adds more if not converged.
    
    Args:
        integrand: Function to integrate
        integ: Vegas Integrator object
        nitn_min: Minimum iterations
        neval: Evaluations per iteration
        alpha: Adaptation parameter
        nproc: Number of processes
        nitn_max: Maximum iterations
        rel_error_target: Target relative error
    
    Returns:
        (result, iterations_used)
    """
    result = integ(integrand, nitn=nitn_min, neval=neval, alpha=alpha, nproc=nproc)
    iterations_used = nitn_min
    
    rel_error = _get_max_rel_error(result)
    
    if rel_error < rel_error_target:
        return result, iterations_used
    
    while iterations_used < nitn_max and rel_error > rel_error_target:
        batch_size = min(3, nitn_max - iterations_used)
        result = integ(integrand, nitn=batch_size, neval=neval, alpha=alpha, nproc=nproc)
        iterations_used += batch_size
        
        rel_error = _get_max_rel_error(result)
    
    return result, iterations_used


def calculate_normalization(radius, sigma_r, n_points=500):
    """
    Calculate normalization integrals for radius distribution.
    
    INT1 = integral of exp(-(r-r0)^2 / (2*sigma^2))
    INT2 = integral of r^2 * exp(-(r-r0)^2 / (2*sigma^2))
    
    Returns: (r_sq_avg, int2)
    """
    r_min = max(0.0, radius - 5.0 * sigma_r)
    r_max = radius + 5.0 * sigma_r
    r = np.linspace(r_min, r_max, n_points)
    
    gaussian = np.exp(-(r - radius)**2 / (2.0 * sigma_r**2))
    int1 = np.trapz(gaussian, r)
    int2 = np.trapz(r**2 * gaussian, r)
    
    r_sq_avg = int2 / int1 if int1 > 0 else radius**2
    return r_sq_avg, int2


def apply_resolution_smearing(q_ideal, intensity_ideal, q_data, sigma_q):
    """
    Apply Gaussian resolution smearing to the ideal intensity curve.
    """
    intensity_smeared = np.zeros_like(q_data)
    interpolator = interp1d(
        q_ideal, intensity_ideal, 
        kind='linear', bounds_error=False, fill_value=0.0
    )
    
    n_smooth = 101
    
    for i, (q_i, sigma_i) in enumerate(zip(q_data, sigma_q)):
        q_range = np.linspace(q_i - 4*sigma_i, q_i + 4*sigma_i, n_smooth)
        intensity_at_q = interpolator(q_range)
        
        kernel = np.exp(-(q_range - q_i)**2 / (2.0 * sigma_i**2))
        kernel /= sigma_i * np.sqrt(2.0 * np.pi)
        
        intensity_smeared[i] = np.trapz(intensity_at_q * kernel, q_range)
    
    return intensity_smeared


def compute_cates_weights(lc, n_cyl_max):
    """
    Compute Cates exponential weights for chain-length polydispersity.

    Args:
        lc: Average cylinder number (Cates LC parameter)
        n_cyl_max: Maximum cylinder count to include

    Returns:
        (n_values, weights, n_weighted_sum)
    """
    if lc <= 0 or n_cyl_max <= 0:
        return None, None, None
    n_values = np.arange(1, n_cyl_max + 1, dtype=np.float64)
    weights = np.exp(-n_values / float(lc))
    n_weighted_sum = np.sum(n_values * weights)
    return n_values, weights, n_weighted_sum
