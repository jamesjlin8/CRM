#!/usr/bin/env python3
"""
Predict physical parameters from a single scattering pattern using PCA models.

Usage:
    python run_predict.py /path/to/pattern.dat [options]

Options:
    --pca-dir DIR            PCA artifacts directory (default: pca/pca12m)
    --models-dir DIR         Trained models directory (default: xgmodels/10modes12m)
    --results-dir DIR        Output directory (default: predictions/)
    --q-min Q                Exclude |q| < Q (default: 0.004)
    --q-max Q                Exclude |q| > Q from fit (default: None)
    --no-rescale             Skip affine rescaling (input already on simulation scale, for testing) (default: False)
    --rescale-background BG  Subtract constant BG before fit (same units as intensity) (default: None)
    --rescale-scale SCALE    Fix affine scale in fit (default: None, fit jointly with background)
    --beta BETA              RPA structure factor: model-side S(q) = 1/(1+beta*P(q)) (default: None)
    --ridge-lambda LAMBDA    Ridge regularization strength used in affine rescaling/PCA projection (default: 0.01)
    --smooth-sigma SIGMA     Mask-normalized smoothing on aligned intensity grid (default: 1.0)
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_training_q_values(pca_results_dir: Path, pca_model: dict, n_features: int) -> np.ndarray | None:
    """Load the training q-grid from PCA artifacts if available."""
    q_values = pca_model.get("q_values")
    if q_values is None:
        q_values = pca_model.get("q_values_")
    if q_values is None:
        for candidate in ("q_values.npy", "q_values.txt"):
            candidate_path = pca_results_dir / candidate
            if candidate_path.exists():
                q_values = np.load(candidate_path) if candidate_path.suffix == ".npy" else np.loadtxt(candidate_path)
                break
    if q_values is None:
        return None
    q_values = np.asarray(q_values).reshape(-1)
    if len(q_values) < n_features:
        raise ValueError(
            f"Training q-grid has {len(q_values)} points, but PCA expects {n_features}. "
            "Re-run run_pca_analysis.py to regenerate compatible results."
        )
    return q_values[:n_features]


def load_pattern_raw(pattern_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a .dat pattern file (qx, qy, p)."""
    data = np.loadtxt(pattern_path)
    if data.size == 0:
        raise ValueError(f"Empty file: {pattern_path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Invalid format in {pattern_path}: expected 3 columns, got {data.shape[1]}")
    return data[:, 0], data[:, 1], data[:, 2]


def load_pattern_sorted(
    pattern_path: Path,
    q_min_override: float | None = None,
    q_max_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a .dat pattern file and return q, p sorted by |q|.

    Drops non-finite / non-positive pixels and optionally applies
    q_min / q_max cutoffs.  Returns (q_sorted, p_sorted, q_min_valid).
    """
    qx, qy, p = load_pattern_raw(pattern_path)
    beamstop_r = float(q_min_override) if q_min_override is not None else 0.0

    q_all = np.sqrt(qx**2 + qy**2)
    valid = (p > 0) & np.isfinite(p) & (q_all >= beamstop_r)
    if q_max_override is not None:
        valid &= q_all <= float(q_max_override)

    p_v = p[valid]
    if len(p_v) == 0:
        raise ValueError(f"No valid intensities outside beamstop in {pattern_path}")
    q = np.sqrt(qx[valid]**2 + qy[valid]**2)
    order = np.argsort(q)
    q_min_valid = beamstop_r if beamstop_r > 0 else float(q[order[0]])
    return q[order], p_v[order], q_min_valid


def format_plot_filename(pattern_path: Path, model_label: str) -> str:
    return f"{model_label}_{pattern_path.stem}_pca_coefficients.png"


# ---------------------------------------------------------------------------
# Alignment & resampling
# ---------------------------------------------------------------------------

def _wrap_nematic_angle(angle: float) -> float:
    """Normalize a director angle to [-pi/2, pi/2)."""
    return float((angle + np.pi / 2) % np.pi - np.pi / 2)


def _sqrt_intensity_weights(p: np.ndarray) -> np.ndarray:
    """sqrt(I) weighting for annular-harmonic orientation (robust to hot pixels)."""
    return np.sqrt(np.maximum(np.asarray(p, dtype=float), 0.0))


def _principal_angle_from_harmonic(phi: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return nematic director angle and normalized second-harmonic strength."""
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return 0.0, 0.0
    c2 = float(np.sum(weights * np.cos(2 * phi)))
    s2 = float(np.sum(weights * np.sin(2 * phi)))
    return _wrap_nematic_angle(0.5 * np.arctan2(s2, c2)), float(np.hypot(c2, s2) / weight_sum)


def _harmonic_director_from_masked_pixels(
    qx: np.ndarray, qy: np.ndarray, p: np.ndarray, mask: np.ndarray,
) -> tuple[float, float]:
    """Single-pass sqrt-weighted 2φ director on masked pixels (fallback when annuli unusable)."""
    phi = np.arctan2(qy, qx)
    w = _sqrt_intensity_weights(p[mask])
    return _principal_angle_from_harmonic(phi[mask], w)


def calculate_annular_harmonic_angle(
    qx: np.ndarray,
    qy: np.ndarray,
    p: np.ndarray,
    q_min: float,
    q_max: float | None,
    n_q_bins: int = 24,
    n_phi_bins: int = 72,
    smooth_sigma: float = 0.0,
) -> tuple[float, dict[str, float | int | str]]:
    """Robust director estimate from annular azimuthal second harmonics (sqrt weights)."""
    q_mag = np.sqrt(qx**2 + qy**2)
    phi = np.arctan2(qy, qx)
    mask = (q_mag >= q_min) & (p > 0) & np.isfinite(p) & np.isfinite(q_mag) & np.isfinite(phi)
    if q_max is not None:
        mask &= q_mag <= q_max
    if not np.any(mask):
        return 0.0, {"method": "annular-harmonic", "pixels": 0, "rings": 0, "strength": 0.0, "coverage": 0.0}

    q_v = q_mag[mask]
    phi_v = phi[mask]
    weights_v = _sqrt_intensity_weights(p[mask])
    positive_weights = weights_v > 0
    q_v, phi_v, weights_v = q_v[positive_weights], phi_v[positive_weights], weights_v[positive_weights]
    if q_v.size == 0:
        return 0.0, {"method": "annular-harmonic", "pixels": 0, "rings": 0, "strength": 0.0, "coverage": 0.0}

    q_hi = float(q_max) if q_max is not None else float(q_v.max())
    q_lo = max(float(q_min), float(q_v.min()))
    if q_hi <= q_lo:
        angle, strength = _harmonic_director_from_masked_pixels(qx, qy, p, mask)
        return angle, {
            "method": "annular-harmonic", "pixels": int(mask.sum()), "rings": 0,
            "strength": strength, "coverage": 0.0,
        }

    q_edges = np.linspace(q_lo, q_hi, max(2, int(n_q_bins) + 1))
    phi_edges = np.linspace(-np.pi, np.pi, max(8, int(n_phi_bins) + 1))
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    z_sum = 0.0j
    ring_weight_sum = 0.0
    coverage_sum = 0.0
    rings_used = 0

    for lo, hi in zip(q_edges[:-1], q_edges[1:]):
        in_ring = (q_v >= lo) & (q_v < hi)
        if not np.any(in_ring):
            continue

        phi_idx = np.searchsorted(phi_edges, phi_v[in_ring], side="right") - 1
        phi_idx = np.clip(phi_idx, 0, len(phi_centers) - 1)
        angular_weights = np.bincount(phi_idx, weights=weights_v[in_ring], minlength=len(phi_centers))
        occupied = angular_weights > 0
        coverage = float(np.mean(occupied))
        if coverage < 0.25 or float(angular_weights.sum()) <= 0:
            continue

        if smooth_sigma > 0:
            angular_weights = gaussian_filter1d(angular_weights, sigma=float(smooth_sigma), mode="wrap")

        ring_angle, ring_strength = _principal_angle_from_harmonic(phi_centers, angular_weights)
        if ring_strength <= 0 or not np.isfinite(ring_strength):
            continue

        # Reliable anisotropic rings should dominate over weak or partially covered rings.
        ring_weight = float(angular_weights.sum()) * ring_strength**2 * coverage
        z_sum += ring_weight * np.exp(2j * ring_angle)
        ring_weight_sum += ring_weight
        coverage_sum += coverage
        rings_used += 1

    if ring_weight_sum <= 0 or rings_used == 0:
        angle, strength = _harmonic_director_from_masked_pixels(qx, qy, p, mask)
        return angle, {
            "method": "annular-harmonic", "pixels": int(q_v.size), "rings": 0,
            "strength": strength, "coverage": 0.0,
        }

    angle = _wrap_nematic_angle(0.5 * np.angle(z_sum))
    return angle, {
        "method": "annular-harmonic",
        "pixels": int(q_v.size),
        "rings": rings_used,
        "strength": float(np.abs(z_sum) / ring_weight_sum),
        "coverage": float(coverage_sum / rings_used),
    }


def rotate_coordinates(qx: np.ndarray, qy: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate raw q coordinates into the simulation frame."""
    cos0, sin0 = np.cos(angle), np.sin(angle)
    return qx * cos0 - qy * sin0, qx * sin0 + qy * cos0


def align_pattern_2d_to_master(
    qx: np.ndarray, qy: np.ndarray, p: np.ndarray,
    rotation_angle: float, qx_ref: np.ndarray, qy_ref: np.ndarray,
    beamstop_qmin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate raw coordinates, then interpolate experimental intensities onto the master grid."""
    qx_rot, qy_rot = rotate_coordinates(qx, qy, rotation_angle)
    aligned = griddata((qx_rot, qy_rot), p, (qx_ref, qy_ref), method="linear", fill_value=np.nan)
    aligned = np.asarray(aligned, dtype=float)
    interp_valid = np.isfinite(aligned)
    aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
    aligned = np.maximum(aligned, 0.0)
    if beamstop_qmin > 0:
        beamstop_mask = np.sqrt(qx_ref**2 + qy_ref**2) < beamstop_qmin
        aligned[beamstop_mask] = 0.0
        interp_valid &= ~beamstop_mask
    return aligned, interp_valid


def smooth_reference_grid_values(
    qx_ref: np.ndarray,
    qy_ref: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
    regions: list[tuple[int, int]],
    sigma: float,
) -> np.ndarray:
    """Mask-normalized Gaussian smoothing on regular detector regions."""
    if sigma <= 0:
        return values

    smoothed = values.copy()
    for s, e in regions:
        n = e - s
        side = int(round(np.sqrt(n)))
        if side * side != n:
            continue

        vals = values[s:e]
        valid = valid_mask[s:e].astype(float)
        qxr, qyr = qx_ref[s:e], qy_ref[s:e]
        transpose_grid = (qyr[:side].max() - qyr[:side].min()) > (qxr[:side].max() - qxr[:side].min())
        if transpose_grid:
            grid_2d = vals.reshape(side, side).T
            mask_2d = valid.reshape(side, side).T
            qx_axis = np.array([qxr[i * side : (i + 1) * side].mean() for i in range(side)])
            qy_axis = qyr[:side].copy()
        else:
            grid_2d = vals.reshape(side, side)
            mask_2d = valid.reshape(side, side)
            qx_axis = qxr[:side].copy()
            qy_axis = np.array([qyr[i * side : (i + 1) * side].mean() for i in range(side)])

        flip_x = qx_axis[-1] < qx_axis[0]
        flip_y = qy_axis[-1] < qy_axis[0]
        if flip_x:
            grid_2d = grid_2d[:, ::-1]
            mask_2d = mask_2d[:, ::-1]
        if flip_y:
            grid_2d = grid_2d[::-1, :]
            mask_2d = mask_2d[::-1, :]

        weighted = gaussian_filter(grid_2d * mask_2d, sigma=float(sigma), mode="nearest")
        norm = gaussian_filter(mask_2d, sigma=float(sigma), mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            grid_s = np.where(norm > 1e-12, weighted / norm, grid_2d)

        if flip_y:
            grid_s = grid_s[::-1, :]
        if flip_x:
            grid_s = grid_s[:, ::-1]
        flat_s = grid_s.T.reshape(-1) if transpose_grid else grid_s.reshape(-1)
        region_valid = valid_mask[s:e]
        region_out = smoothed[s:e].copy()
        region_out[region_valid] = np.maximum(flat_s[region_valid], 0.0)
        smoothed[s:e] = region_out

    return smoothed


def resample_pattern(
    q_sorted: np.ndarray, p_sorted: np.ndarray,
    n_features: int, q_train: np.ndarray | None,
) -> np.ndarray:
    """Resample a 1D pattern onto the training grid or truncate to PCA length."""
    if len(p_sorted) == 0:
        raise ValueError("Pattern has no points to resample.")
    if q_train is not None:
        if len(q_train) != n_features:
            raise ValueError(f"Training q-grid has {len(q_train)} points, but PCA expects {n_features}.")
        if len(p_sorted) == n_features and np.allclose(q_sorted, q_train):
            return p_sorted.copy()
        if len(p_sorted) == 1:
            return np.full(n_features, p_sorted[0])
        return np.interp(q_train, q_sorted, p_sorted, left=p_sorted[0], right=p_sorted[-1])
    if len(p_sorted) < n_features:
        raise ValueError(
            f"Pattern has {len(p_sorted)} points, but PCA expects {n_features}. "
            "Provide a pattern on the training q-grid or include q_values in pca_results."
        )
    return p_sorted[:n_features]


# ---------------------------------------------------------------------------
# PCA projection & reconstruction
# ---------------------------------------------------------------------------

def _rpa_structure_factor(P: np.ndarray, beta: float) -> np.ndarray:
    """S(q) = 1 / (1 + beta * max(P, 0))."""
    return 1.0 / (1.0 + beta * np.maximum(P, 0.0))


def _forward_structure_model(P: np.ndarray, beta: float | None) -> np.ndarray:
    """Apply the model-side structure factor to a reconstructed form factor."""
    if beta is None:
        return P
    return P * _rpa_structure_factor(P, beta)


def project_pattern_masked(
    aligned: np.ndarray, U: np.ndarray, mean: np.ndarray,
    n_components: int, valid_mask: np.ndarray,
    beta: float | None = None,
    max_iter: int = 200, tol: float = 1e-6,
) -> tuple[np.ndarray, int, dict]:
    """Project with missing-data mask via nonlinear least squares.

    Returns (alpha_row_vector, n_function_evaluations, fit_info).
    """
    U_k = U[:, :n_components]
    aligned_v = aligned[valid_mask]
    mean_v = mean[valid_mask]
    U_v = U_k[valid_mask, :]
    if aligned_v.size == 0:
        raise ValueError("No valid pixels available for PCA projection.")

    def residual(x: np.ndarray) -> np.ndarray:
        P_v = mean_v + U_v @ x
        return aligned_v - _forward_structure_model(P_v, beta)

    result = least_squares(
        residual,
        x0=np.zeros(n_components),
        max_nfev=max_iter,
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    fit_info = {
        "jac": result.jac,
        "fun": result.fun,
        "alpha_slice": slice(0, n_components),
        "n_data": aligned_v.size,
        "n_params": result.x.size,
    }
    return result.x.reshape(1, -1), int(result.nfev), fit_info


def project_with_rescaling(
    aligned: np.ndarray, U: np.ndarray, mean: np.ndarray,
    n_components: int,
    valid_mask: np.ndarray | None = None,
    ridge_lambda: float = 0.01,
    fixed_background: float | None = None,
    fixed_scale: float | None = None,
    beta: float | None = None,
    max_iter: int = 200, tol: float = 1e-6,
) -> tuple[np.ndarray, float, float, int, dict]:
    """Affine rescaling + PCA projection with optional model-side RPA.

    Minimizes ||scale*p + bg - forward(mean + U @ alpha, beta)||^2
    + lam*||alpha||^2. Scale and/or background may be fixed (not optimized).
    Returns (alpha, scale, background, n_function_evaluations, fit_info).
    """
    k_full = U.shape[1]
    if valid_mask is None:
        valid_mask = np.ones(len(aligned), dtype=bool)
    p_v = aligned[valid_mask]
    mean_v = mean[valid_mask]
    U_v = U[:, :k_full][valid_mask, :]
    if p_v.size == 0:
        raise ValueError("No valid pixels available for PCA projection.")

    model0_v = _forward_structure_model(mean_v, beta)
    finite_p = p_v[np.isfinite(p_v)]
    finite_model = model0_v[np.isfinite(model0_v)]
    p_med = float(np.median(finite_p)) if finite_p.size else 1.0
    model_med = float(np.median(finite_model)) if finite_model.size else 1.0
    scale_init = float(
        fixed_scale
        if fixed_scale is not None
        else (model_med / p_med if p_med != 0.0 else 1.0)
    )
    if not np.isfinite(scale_init) or scale_init == 0.0:
        scale_init = 1.0
    bg_init = float(
        fixed_background
        if fixed_background is not None
        else model_med - scale_init * p_med
    )
    ridge_weight = float(np.sqrt(ridge_lambda)) if ridge_lambda > 0 else 0.0

    def alpha_penalty(alpha: np.ndarray) -> np.ndarray:
        if ridge_weight == 0.0:
            return np.empty(0)
        return ridge_weight * alpha

    def unpack(x: np.ndarray) -> tuple[float, float, np.ndarray]:
        i = 0
        if fixed_scale is None:
            scale = float(x[i])
            i += 1
        else:
            scale = float(fixed_scale)
        if fixed_background is None:
            background = float(x[i])
            i += 1
        else:
            background = float(fixed_background)
        return scale, background, x[i:]

    def residual(x: np.ndarray) -> np.ndarray:
        scale, background, alpha = unpack(x)
        P_v = mean_v + U_v @ alpha
        data_resid = scale * p_v + background - _forward_structure_model(P_v, beta)
        return np.concatenate((data_resid, alpha_penalty(alpha)))

    x0_parts: list[np.ndarray] = []
    if fixed_scale is None:
        x0_parts.append(np.array([scale_init], dtype=float))
    if fixed_background is None:
        x0_parts.append(np.array([bg_init], dtype=float))
    x0_parts.append(np.zeros(k_full, dtype=float))
    x0 = np.concatenate(x0_parts)
    alpha_start = x0.size - k_full

    result = least_squares(
        residual,
        x0=x0,
        max_nfev=max_iter,
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    scale, background, alpha_flat = unpack(result.x)
    alpha = alpha_flat.reshape(1, -1)
    alpha_slice = slice(alpha_start, alpha_start + k_full)

    fit_info = {
        "jac": result.jac,
        "fun": result.fun,
        "alpha_slice": alpha_slice,
        "n_data": p_v.size,
        "n_params": result.x.size,
    }
    return alpha, scale, background, int(result.nfev), fit_info


def alpha_covariance_from_fit(fit_info: dict, n_components: int) -> np.ndarray:
    """Estimate local PCA-coefficient covariance from a nonlinear least-squares Jacobian."""
    jac = np.asarray(fit_info["jac"], dtype=float)
    fun = np.asarray(fit_info["fun"], dtype=float)
    n_data = int(fit_info["n_data"])
    n_params = int(fit_info["n_params"])
    alpha_slice = fit_info["alpha_slice"]

    data_resid = fun[:n_data]
    dof = max(1, n_data - n_params)
    sigma2 = float(np.sum(data_resid**2) / dof)
    cov_theta = sigma2 * np.linalg.pinv(jac.T @ jac, rcond=1e-12)
    cov_alpha = np.asarray(cov_theta[alpha_slice, alpha_slice], dtype=float)
    cov_alpha = cov_alpha[:n_components, :n_components]
    return 0.5 * (cov_alpha + cov_alpha.T)


def sample_alpha_from_covariance(
    alpha: np.ndarray,
    cov_alpha: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Draw coefficient-space perturbations from a PSD-clipped local covariance."""
    alpha_1d = np.asarray(alpha, dtype=float).reshape(-1)
    cov_alpha = np.asarray(cov_alpha, dtype=float)
    if cov_alpha.shape != (alpha_1d.size, alpha_1d.size):
        raise ValueError(
            f"Alpha covariance shape {cov_alpha.shape} does not match alpha size {alpha_1d.size}."
        )
    if not np.all(np.isfinite(cov_alpha)):
        return np.repeat(alpha_1d.reshape(1, -1), n_samples, axis=0)

    eigvals, eigvecs = np.linalg.eigh(cov_alpha)
    eigvals = np.clip(eigvals, 0.0, None)
    if eigvals.size == 0 or float(eigvals.max()) == 0.0:
        return np.repeat(alpha_1d.reshape(1, -1), n_samples, axis=0)

    draws = rng.normal(size=(n_samples, alpha_1d.size))
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    return alpha_1d + draws @ transform.T


def reconstruct_pattern(alpha: np.ndarray, U: np.ndarray, mean: np.ndarray, n_components: int) -> np.ndarray:
    """mean + alpha @ U^T."""
    return mean + (alpha.reshape(1, -1) @ U[:, :n_components].T).reshape(-1)


def sim_to_experimental(
    P_sim: np.ndarray, scale: float, bg_fit: float,
    bg_sub: float | None, beta: float | None,
) -> np.ndarray:
    """Map PCA reconstruction (form factor P in sim space) to raw experimental I.

    Path: RPA forward in sim space -> inverse affine -> add back subtracted bg.
    """
    I_sim = _forward_structure_model(P_sim, float(beta) if beta is not None else None)
    bg = bg_fit if bg_sub is None else 0.0
    I_exp = (I_sim - bg) / scale if scale != 0.0 else np.array(I_sim, dtype=float)
    if bg_sub is not None:
        I_exp = I_exp + float(bg_sub)
    return I_exp


# ---------------------------------------------------------------------------
# 2D gridding helpers
# ---------------------------------------------------------------------------

def detect_detector_regions(qx_ref: np.ndarray, qy_ref: np.ndarray) -> list[tuple[int, int]]:
    """Detect contiguous detector regions in a concatenated multi-detector grid."""
    n = len(qx_ref)
    for n_regions in range(1, 6):
        if n % n_regions != 0:
            continue
        chunk = n // n_regions
        side = int(round(np.sqrt(chunk)))
        if side * side != chunk:
            continue
        all_regular = True
        for r in range(n_regions):
            s = r * chunk
            qx_span = qx_ref[s : s + side].max() - qx_ref[s : s + side].min()
            qy_span = qy_ref[s : s + side].max() - qy_ref[s : s + side].min()
            full_qx = qx_ref[s : s + chunk].max() - qx_ref[s : s + chunk].min()
            full_qy = qy_ref[s : s + chunk].max() - qy_ref[s : s + chunk].min()
            if not (qx_span / max(full_qx, 1e-12) < 0.02 or qy_span / max(full_qy, 1e-12) < 0.02):
                all_regular = False
                break
        if all_regular:
            return [(r * chunk, (r + 1) * chunk) for r in range(n_regions)]
    return [(0, n)]


def grid_pattern_2d(
    qx: np.ndarray, qy: np.ndarray, values: np.ndarray,
    max_grid_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid scattered (qx, qy, values) data onto a 2D array for imshow."""
    unique_qx = np.unique(qx)
    unique_qy = np.unique(qy)

    if len(unique_qx) * len(unique_qy) <= 4 * len(qx):
        unique_qx.sort()
        unique_qy.sort()
        grid = np.full((len(unique_qy), len(unique_qx)), np.nan)
        grid[np.searchsorted(unique_qy, qy), np.searchsorted(unique_qx, qx)] = values
        return grid, unique_qx, unique_qy

    n_bins = min(max_grid_size, max(int(np.sqrt(len(qx))), 64))
    qx_edges = np.linspace(qx.min(), qx.max(), n_bins + 1)
    qy_edges = np.linspace(qy.min(), qy.max(), n_bins + 1)
    sum_grid, _, _ = np.histogram2d(qx, qy, bins=[qx_edges, qy_edges], weights=values)
    count_grid, _, _ = np.histogram2d(qx, qy, bins=[qx_edges, qy_edges])
    with np.errstate(invalid="ignore"):
        grid = np.where(count_grid > 0, sum_grid / count_grid, np.nan).T
    return grid, 0.5 * (qx_edges[:-1] + qx_edges[1:]), 0.5 * (qy_edges[:-1] + qy_edges[1:])


def grid_region_2d(
    qx: np.ndarray, qy: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid a single detector region into a 2D array (no binning)."""
    n = len(values)
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Region has {n} pixels which is not a perfect square.")

    if (qy[:side].max() - qy[:side].min()) > (qx[:side].max() - qx[:side].min()):
        grid_2d = values.reshape(side, side).T
        qx_axis = np.array([qx[i * side : (i + 1) * side].mean() for i in range(side)])
        qy_axis = qy[:side].copy()
    else:
        grid_2d = values.reshape(side, side)
        qx_axis = qx[:side].copy()
        qy_axis = np.array([qy[i * side : (i + 1) * side].mean() for i in range(side)])

    if qx_axis[-1] < qx_axis[0]:
        qx_axis = qx_axis[::-1]
        grid_2d = grid_2d[:, ::-1]
    if qy_axis[-1] < qy_axis[0]:
        qy_axis = qy_axis[::-1]
        grid_2d = grid_2d[::-1, :]
    return grid_2d, qx_axis, qy_axis


def _average_duplicate_q_for_interp(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (q, y) with q strictly increasing (mean-aggregated duplicates)."""
    m = np.isfinite(q) & np.isfinite(y)
    qv, yv = np.asarray(q[m], dtype=float), np.asarray(y[m], dtype=float)
    if qv.size == 0:
        return qv, yv
    order = np.argsort(qv)
    qv, yv = qv[order], yv[order]
    uq, inv = np.unique(qv, return_inverse=True)
    return uq, np.bincount(inv, weights=yv) / np.maximum(np.bincount(inv).astype(float), 1.0)


def _print_section(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict physical parameters from a single .dat pattern using PCA models."
    )
    parser.add_argument("pattern_file", type=str, help="Path to a .dat file with columns: qx qy p")
    parser.add_argument("--pca-dir", type=str, default="pca/pca12m", dest="pca_dir")
    parser.add_argument("--results-dir", type=str, default="predictions/", dest="results_dir")
    parser.add_argument("--models-dir", type=str, default="xgmodels/10modes12m")
    parser.add_argument("--no-rescale", action="store_true", help="Skip affine rescaling")
    parser.add_argument("--q-min", type=float, default=0.004, help="Beamstop radius (exclude |q| < value)")
    parser.add_argument("--q-max", type=float, default=None, dest="q_max", help="Exclude |q| > value")
    parser.add_argument("--rescale-background", type=float, default=None,
                        help="Constant background to subtract before fit (default: None)")
    parser.add_argument("--rescale-scale", type=float, default=None, dest="rescale_scale",
                        help="Fix affine scale in fit (default: None, fit jointly with background)")
    parser.add_argument("--beta", type=float, default=None,
                        help="RPA structure factor beta (model-side S = 1/(1+beta*P)) (default: None)")
    parser.add_argument("--ridge-lambda", type=float, default=0.01, dest="ridge_lambda",
                        help="Ridge regularization strength used in affine rescaling/PCA projection")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, dest="smooth_sigma",
                        help="Optional mask-normalized Gaussian sigma on aligned 2D intensity grid")
    args = parser.parse_args()

    # --- Validate arguments ---
    if args.q_min is not None and args.q_min < 0:
        raise ValueError(f"Invalid --q-min {args.q_min}: must be non-negative.")
    if args.q_max is not None and args.q_max <= 0:
        raise ValueError(f"Invalid --q-max {args.q_max}: must be positive.")
    if args.q_min is not None and args.q_max is not None and args.q_max <= args.q_min:
        raise ValueError(f"--q-max {args.q_max} must be greater than --q-min {args.q_min}.")
    if args.no_rescale and args.rescale_background is not None:
        raise ValueError("--rescale-background cannot be used with --no-rescale.")
    if args.no_rescale and args.rescale_scale is not None:
        raise ValueError("--rescale-scale cannot be used with --no-rescale.")
    if args.rescale_scale is not None and args.rescale_scale == 0:
        raise ValueError("--rescale-scale must be non-zero.")
    if args.beta is not None and args.beta == 0:
        args.beta = None
    if args.ridge_lambda < 0:
        raise ValueError(f"Invalid --ridge-lambda {args.ridge_lambda}: must be non-negative.")
    if args.smooth_sigma < 0:
        raise ValueError("--smooth-sigma must be non-negative.")

    beamstop_qmin = float(args.q_min) if args.q_min is not None else 0.0
    if beamstop_qmin > 0:
        print(f"Beamstop radius (manual): |q| < {beamstop_qmin:.6f}")
    if args.q_max is not None:
        print(f"|q| upper cutoff: |q| > {float(args.q_max):.6f} excluded")

    # --- Load PCA model & XGBoost models ---
    pattern_path = Path(args.pattern_file)
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

    pca_results_dir = Path(args.pca_dir)
    pca_file = pca_results_dir / "pca_components.pkl"
    models_dir = Path(args.models_dir) if args.models_dir else pca_results_dir / "models"
    model_plot_label = models_dir.name
    models_file = models_dir / "parameter_models.pkl"

    if not pca_file.exists():
        raise FileNotFoundError(f"PCA components not found: {pca_file}")
    if not models_file.exists():
        legacy = pca_results_dir / "parameter_models.pkl"
        if legacy.exists():
            models_file = legacy
        else:
            raise FileNotFoundError(f"Parameter models not found: {models_file}")

    with open(pca_file, "rb") as f:
        U, _S, pca_model = pickle.load(f)
    mean = pca_model["mean_"]
    n_features = U.shape[0]
    q_train = load_training_q_values(pca_results_dir, pca_model, n_features)

    qx_ref_path = pca_results_dir / "qx_ref.npy"
    qy_ref_path = pca_results_dir / "qy_ref.npy"
    have_2d_grid = qx_ref_path.exists() and qy_ref_path.exists()
    if have_2d_grid:
        qx_ref = np.load(qx_ref_path).reshape(-1)[:n_features]
        qy_ref = np.load(qy_ref_path).reshape(-1)[:n_features]
        q_mag_ref = np.sqrt(qx_ref**2 + qy_ref**2)
        sim_regions = detect_detector_regions(qx_ref, qy_ref)
    else:
        qx_ref = qy_ref = q_mag_ref = None

    with open(models_file, "rb") as f:
        model_bundle = pickle.load(f)
    n_components = min(model_bundle.get("n_components", U.shape[1]), U.shape[1])
    models = model_bundle.get("models", {})
    if not models:
        raise ValueError("No parameter models found in parameter_models.pkl")

    # ------------------------------------------------------------------
    # Align experimental pattern to master grid
    # ------------------------------------------------------------------
    _print_section("Pattern alignment")
    qx_raw, qy_raw, p_raw = load_pattern_raw(pattern_path)

    rotation_angle_deg = None
    if have_2d_grid:
        angle_q_min = beamstop_qmin
        angle_q_max = float(args.q_max) if args.q_max is not None else None
        p_for_angle = p_raw.astype(float, copy=True)
        if args.rescale_background is not None:
            p_for_angle -= float(args.rescale_background)

        phi_0, angle_stats = calculate_annular_harmonic_angle(
            qx_raw, qy_raw, p_for_angle,
            q_min=angle_q_min, q_max=angle_q_max,
        )
        rotation_angle = np.pi / 2 - phi_0
        rotation_angle_deg = float(np.degrees(rotation_angle))
        angle_q_max_label = f"{angle_q_max:.6g}" if angle_q_max is not None else "data max"
        print(
            f"Annular-harmonic orientation (q=[{angle_q_min:.6g}, {angle_q_max_label}]), "
            f"pixels={int(angle_stats.get('pixels', 0))}, rings={int(angle_stats.get('rings', 0))}, "
            f"R2={float(angle_stats.get('strength', 0.0)):.4g}, "
            f"coverage={float(angle_stats.get('coverage', 0.0)):.3g}"
        )
        print(
            f"Director phi_0 = {np.degrees(phi_0):.3f} deg, "
            f"raw-coordinate rotation = {rotation_angle_deg:.3f} deg"
        )
        aligned, interp_valid = align_pattern_2d_to_master(
            qx_raw, qy_raw, p_raw, rotation_angle, qx_ref, qy_ref, beamstop_qmin,
        )

        q_exp_min = float(np.sqrt(qx_raw**2 + qy_raw**2).min())
        q_exp_max = float(np.sqrt(qx_raw**2 + qy_raw**2).max())
        valid_mask = (
            interp_valid & np.isfinite(aligned) & (q_mag_ref >= beamstop_qmin)
            & (q_mag_ref >= q_exp_min) & (q_mag_ref <= q_exp_max)
        )
        if args.q_max is not None:
            valid_mask &= q_mag_ref <= float(args.q_max)
        if not np.any(valid_mask):
            valid_mask = interp_valid & np.isfinite(aligned) & (q_mag_ref >= beamstop_qmin)
            if args.q_max is not None:
                valid_mask &= q_mag_ref <= float(args.q_max)
        if args.smooth_sigma > 0:
            aligned = smooth_reference_grid_values(
                qx_ref, qy_ref, aligned, valid_mask, sim_regions,
                sigma=float(args.smooth_sigma),
            )
            print(f"Applied mask-normalized Gaussian smoothing: sigma={float(args.smooth_sigma):.3g} px")
    else:
        q_sorted, p_sorted, q_min_valid = load_pattern_sorted(
            pattern_path,
            q_min_override=beamstop_qmin if beamstop_qmin > 0 else None,
            q_max_override=float(args.q_max) if args.q_max is not None else None,
        )
        aligned = resample_pattern(q_sorted, p_sorted, n_features, q_train)
        if q_train is not None:
            valid_mask = (
                (q_train >= q_min_valid)
                & (q_train >= float(q_sorted[0]))
                & (q_train <= float(q_sorted[-1]))
            )
            if not np.any(valid_mask):
                valid_mask = q_train >= q_min_valid
        else:
            valid_mask = np.ones(n_features, dtype=bool)

    # ------------------------------------------------------------------
    # Preprocessing & PCA fit
    # ------------------------------------------------------------------
    _print_section("Preprocessing & PCA fit")
    aligned_for_fit = aligned.copy()
    if args.rescale_background is not None:
        aligned_for_fit -= float(args.rescale_background)
        print(f"Subtracted fixed background {float(args.rescale_background):.6g}")
    # Treat non-physical intensities as missing so they do not influence fitting.
    nonpositive_mask = np.isfinite(aligned_for_fit) & (aligned_for_fit <= 0.0)
    if np.any(nonpositive_mask):
        aligned_for_fit[nonpositive_mask] = np.nan
        valid_mask = valid_mask & (~nonpositive_mask)
        print(f"Marked {int(np.count_nonzero(nonpositive_mask))} non-positive intensities as N/A for fitting")

    beta = float(args.beta) if args.beta is not None else None
    fixed_bg = 0.0 if args.rescale_background is not None else None
    fixed_scale = float(args.rescale_scale) if args.rescale_scale is not None else None
    if fixed_scale is not None:
        print(f"Fixed affine scale: {fixed_scale:.6g}")

    if args.no_rescale:
        alpha_full, fit_evals, fit_info = project_pattern_masked(
            aligned_for_fit, U, mean, U.shape[1], valid_mask, beta=beta,
        )
        rescale_scale, rescale_bg = 1.0, 0.0
    else:
        alpha_full, rescale_scale, rescale_bg, fit_evals, fit_info = project_with_rescaling(
            aligned_for_fit, U, mean, n_components, valid_mask,
            ridge_lambda=float(args.ridge_lambda),
            fixed_background=fixed_bg,
            fixed_scale=fixed_scale,
            beta=beta,
        )
        print(f"Affine rescaling: scale={rescale_scale:.6g}, background={rescale_bg:.6g}")

    if beta is not None:
        print(f"Nonlinear RPA model-side fit used {fit_evals} function evaluation(s) (beta={beta:.6g})")

    alpha_xg = alpha_full[:, :n_components]

    # ------------------------------------------------------------------
    # Predict parameters
    # ------------------------------------------------------------------
    _print_section("PCA coefficients (XGBoost input)")
    param_order = ["radius", "length", "n_cyl", "stretch"]
    predictions: dict[str, float] = {}
    for param in param_order:
        if param in models and "model" in models[param]:
            predictions[param] = float(models[param]["model"].predict(alpha_xg)[0])
    print(alpha_xg.reshape(-1))

    # ------------------------------------------------------------------
    # Reconstruct and map back to experimental intensity
    # ------------------------------------------------------------------
    reconstructed_1d = reconstruct_pattern(alpha_full, U, mean, U.shape[1])
    recon_exp_1d = sim_to_experimental(reconstructed_1d, rescale_scale, rescale_bg, args.rescale_background, args.beta)

    aligned_masked = np.where(valid_mask, aligned, np.nan)
    recon_exp_masked = np.where(valid_mask, recon_exp_1d, np.nan)

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------
    N_MC = 100
    rng = np.random.default_rng(42)
    cov_alpha = alpha_covariance_from_fit(fit_info, n_components)
    alpha_sigma = np.sqrt(np.clip(np.diag(cov_alpha), 0.0, None))
    alpha_samples = sample_alpha_from_covariance(alpha_xg, cov_alpha, rng, N_MC)
    mc_pca: dict[str, np.ndarray] = {p: np.empty(N_MC) for p in predictions}
    for i in range(N_MC):
        alpha_pert = alpha_samples[i].reshape(1, -1)
        for param in predictions:
            mc_pca[param][i] = float(models[param]["model"].predict(alpha_pert)[0])

    boot_preds: dict[str, np.ndarray] = {}
    for param in predictions:
        ensemble = models[param].get("ensemble", [models[param]["model"]])
        boot_preds[param] = np.array([float(m.predict(alpha_xg)[0]) for m in ensemble])

    mc_stats: dict[str, dict] = {}
    for param in predictions:
        s_pca = float(np.std(mc_pca[param]))
        s_model = float(np.std(boot_preds[param]))
        s_total = float(np.sqrt(s_pca**2 + s_model**2))
        point = predictions[param]
        mc_stats[param] = {
            "s_pca": s_pca,
            "s_model": s_model,
            "sigma_total": s_total,
            "ci_lo": point - 1.96 * s_total,
            "ci_hi": point + 1.96 * s_total,
        }

    _print_section("Predicted parameters (95% CI)")
    for param in param_order:
        if param in mc_stats:
            s = mc_stats[param]
            print(f"  {param}: {predictions[param]:.6g}  (95% CI [{s['ci_lo']:.3g}, {s['ci_hi']:.3g}])")
    uncertainty_lines = [
        f"{p}: s_pca={mc_stats[p]['s_pca']:.3g}, s_model={mc_stats[p]['s_model']:.3g}"
        for p in param_order
        if p in mc_stats
    ]
    rmse_relative_error = float("nan")
    if q_train is not None and valid_mask.any():
        I_exp_diag = aligned_masked[valid_mask]
        recon_exp_diag = recon_exp_masked[valid_mask]
        positive = I_exp_diag > 0
        if positive.sum() > 0:
            rel_err_diag = (recon_exp_diag[positive] - I_exp_diag[positive]) / I_exp_diag[positive]
            finite_rel_err = rel_err_diag[np.isfinite(rel_err_diag)]
            if finite_rel_err.size > 0:
                rmse_relative_error = float(np.sqrt(np.mean(finite_rel_err**2)))
    print(f"  RMSE relative error: {rmse_relative_error:.6g}")

    # ------------------------------------------------------------------
    # Diagnostic data
    # ------------------------------------------------------------------
    per_q_diagnostic_plot_data = None
    if q_train is not None and valid_mask.any():
        q_diag = q_train[valid_mask]
        I_exp_diag = aligned_masked[valid_mask]
        recon_exp_diag = recon_exp_masked[valid_mask]
        positive = I_exp_diag > 0
        if positive.sum() > 0:
            per_q_diagnostic_plot_data = {
                "q_pos": q_diag[positive].copy(),
                "I_exp_pos": np.asarray(I_exp_diag[positive], dtype=float),
                "recon_exp_pos": np.asarray(recon_exp_diag[positive], dtype=float),
            }

    # ------------------------------------------------------------------
    # Prepare 2D grids for plotting
    # ------------------------------------------------------------------
    qx_raw, qy_raw, p_raw = load_pattern_raw(pattern_path)
    q_mag_raw = np.sqrt(qx_raw**2 + qy_raw**2)
    q_lo = q_train.min() if q_train is not None else q_mag_raw.min()
    q_hi = q_train.max() if q_train is not None else q_mag_raw.max()

    in_range = (q_mag_raw >= q_lo) & (q_mag_raw <= q_hi)
    original_grid, gqx, gqy = grid_pattern_2d(qx_raw[in_range], qy_raw[in_range], p_raw[in_range])

    interp_layers = recon_layers = recon_grid = recon_extent = None

    if have_2d_grid:
        region_order = sorted(sim_regions, key=lambda r: -(qx_ref[r[0]:r[1]].max() - qx_ref[r[0]:r[1]].min()))
        interp_layers, recon_layers = [], []
        for s, e in region_order:
            qxr, qyr = qx_ref[s:e], qy_ref[s:e]
            g_int, ux, uy = grid_region_2d(qxr, qyr, aligned_masked[s:e])
            ext = [ux[0], ux[-1], uy[0], uy[-1]]
            interp_layers.append((np.where(g_int > 0, g_int, np.nan), ext))
            g_rec, ux, uy = grid_region_2d(qxr, qyr, recon_exp_masked[s:e])
            recon_layers.append((np.where(g_rec > 0, g_rec, np.nan), ext))
    else:
        n_recon = max(len(gqx), len(gqy), 256)
        recon_qx = np.linspace(gqx[0], gqx[-1], n_recon)
        recon_qy = np.linspace(gqy[0], gqy[-1], n_recon)
        rqx_2d, rqy_2d = np.meshgrid(recon_qx, recon_qy)
        rq_mag = np.sqrt(rqx_2d**2 + rqy_2d**2)
        if q_train is not None:
            q_u, y_u = _average_duplicate_q_for_interp(q_train[valid_mask], recon_exp_1d[valid_mask])
            if q_u.size >= 2:
                recon_flat = np.interp(rq_mag.ravel(), q_u, y_u)
                in_q = (rq_mag.ravel() >= q_u.min()) & (rq_mag.ravel() <= q_u.max())
                recon_flat = np.where(in_q, recon_flat, np.nan)
            else:
                recon_flat = np.full(rq_mag.size, np.nan)
        else:
            q_s = np.sort(q_mag_raw)[:len(recon_exp_1d)]
            recon_flat = np.interp(rq_mag.ravel(), q_s, recon_exp_1d)
        recon_grid = np.where((rq_mag >= q_lo) & (rq_mag <= q_hi), recon_flat.reshape(rq_mag.shape), np.nan)
        recon_grid = np.where(recon_grid > 0, recon_grid, np.nan)
        recon_extent = [recon_qx[0], recon_qx[-1], recon_qy[0], recon_qy[-1]]

    original_grid = np.where(original_grid > 0, original_grid, np.nan)

    # Shared color limits
    color_pools = [original_grid[np.isfinite(original_grid)]]
    if have_2d_grid:
        color_pools.append(aligned_masked[(aligned_masked > 0) & np.isfinite(aligned_masked)])
        color_pools.append(recon_exp_masked[(recon_exp_masked > 0) & np.isfinite(recon_exp_masked)])
    else:
        color_pools.append(recon_grid[np.isfinite(recon_grid)])
    all_valid = np.concatenate([v for v in color_pools if v.size > 0])
    shared_vmin = max(all_valid.min(), 1e-10) if all_valid.size > 0 else 1e-6
    shared_vmax = all_valid.max() if all_valid.size > 0 else 1.0
    orig_extent = [gqx[0], gqx[-1], gqy[0], gqy[-1]]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-q diagnostic figure
    # ------------------------------------------------------------------
    diag_name = format_plot_filename(pattern_path, model_plot_label).replace(".png", "_per_q_diagnostic.png")
    diag_path = results_dir / diag_name
    can_2d = have_2d_grid and interp_layers is not None and recon_layers is not None

    if per_q_diagnostic_plot_data is not None or can_2d:
        diag_layout_kind: str = "1d"

        def _draw_per_q_1d_axes(ax_i, ax_e) -> None:
            d = per_q_diagnostic_plot_data
            assert d is not None
            q_p, Ie, Rc = (np.asarray(d[k], dtype=float) for k in ("q_pos", "I_exp_pos", "recon_exp_pos"))
            order = np.argsort(q_p)
            q_p, Ie, Rc = q_p[order], Ie[order], Rc[order]
            ax_i.loglog(q_p, Ie, ".", alpha=0.3, ms=1, label="Experimental", color="C0")
            ax_i.loglog(q_p, Rc, ".", alpha=0.3, ms=1, label="Reconstruction", color="C1")
            ax_i.set_ylabel("Intensity")
            ax_i.legend(loc="best", fontsize=8)
            ax_i.set_title("Per-q comparison")
            rel_err = (Rc - Ie) / Ie
            finite_e = rel_err[np.isfinite(rel_err)]
            rmse_l = float(np.sqrt(np.mean(finite_e**2))) if len(finite_e) > 0 else float("nan")
            mean_b = float(np.mean(finite_e)) if len(finite_e) > 0 else float("nan")
            ax_e.scatter(q_p, rel_err, s=1, alpha=0.4, color="C3")
            ax_e.axhline(0.0, color="k", ls="--", lw=1)
            ax_e.set_xlabel(r"|q| (Å$^{-1}$)")
            ax_e.set_ylabel("Relative Error")
            err99 = np.nanpercentile(np.abs(np.where(np.isfinite(rel_err), rel_err, np.nan)), 99)
            y_lim = min(2.0, max(0.5, err99 * 1.1)) if np.isfinite(err99) else 1.0
            ax_e.set_ylim(-y_lim, y_lim)
            error_summary = "\n".join(
                [f"Rel. err. RMSE = {rmse_l:.4f}", f"Mean bias = {mean_b:+.4f}", *uncertainty_lines]
            )
            ax_e.text(
                0.98, 0.03,
                error_summary,
                transform=ax_e.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )

        def _draw_2d_diff_axes(fig_2d, ax_a, ax_r, ax_d) -> None:
            def _cbar(im, ax, label):
                cb = fig_2d.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1))
                cb.set_label(label, fontsize=9)
                cb.ax.tick_params(labelsize=8)

            _qxl, _qyl = r"Q$_x$ (Å$^{-1}$)", r"Q$_y$ (Å$^{-1}$)"
            sn = LogNorm(vmin=shared_vmin, vmax=shared_vmax)

            im_a = None
            for g, ext in interp_layers:
                im_a = ax_a.imshow(g, extent=ext, origin="lower", aspect="equal", norm=sn, cmap="turbo")
            ax_a.set(xlabel=_qxl, ylabel=_qyl, title="Aligned",
                     xlim=(orig_extent[0], orig_extent[1]), ylim=(orig_extent[2], orig_extent[3]))
            _cbar(im_a, ax_a, "Intensity")

            im_r = None
            for g, ext in recon_layers:
                im_r = ax_r.imshow(g, extent=ext, origin="lower", aspect="equal", norm=sn, cmap="turbo")
            ax_r.set(xlabel=_qxl, title="Reconstructed",
                     xlim=(orig_extent[0], orig_extent[1]), ylim=(orig_extent[2], orig_extent[3]))
            ax_r.tick_params(axis="y", labelleft=False)
            _cbar(im_r, ax_r, "Intensity")

            dmax = np.nanmax(np.abs(recon_exp_masked - aligned_masked))
            if not np.isfinite(dmax) or dmax <= 0:
                dmax = 1.0
            dn = plt.Normalize(vmin=-dmax, vmax=dmax)
            im_d = None
            for (g_rec, ext), (g_aln, _) in zip(recon_layers, interp_layers):
                im_d = ax_d.imshow(g_aln - g_rec, extent=ext, origin="lower", aspect="equal", norm=dn, cmap="RdBu_r")
            ax_d.set(xlabel=_qxl, title="Difference",
                     xlim=(orig_extent[0], orig_extent[1]), ylim=(orig_extent[2], orig_extent[3]))
            ax_d.tick_params(axis="y", labelleft=False)
            _cbar(im_d, ax_d, "Δ intensity")

        if per_q_diagnostic_plot_data is not None and can_2d:
            diag_layout_kind = "combined"
            fig_d = plt.figure(figsize=(14, 13))
            gs_d = fig_d.add_gridspec(3, 1, height_ratios=[1.08, 1.0, 1.0], hspace=0.22)
            gs_2d = gs_d[0, 0].subgridspec(1, 3, wspace=0.28)
            _draw_2d_diff_axes(fig_d,
                               fig_d.add_subplot(gs_2d[0, 0]),
                               fig_d.add_subplot(gs_2d[0, 1]),
                               fig_d.add_subplot(gs_2d[0, 2]))
            _draw_per_q_1d_axes(fig_d.add_subplot(gs_d[1, 0]),
                                fig_d.add_subplot(gs_d[2, 0]))
            fig_d.suptitle(f"{pattern_path.stem}: reconstruction diagnostic", fontsize=12, fontweight="bold", y=0.99)
        elif per_q_diagnostic_plot_data is not None:
            fig_d, (ax_int, ax_err) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.22})
            _draw_per_q_1d_axes(ax_int, ax_err)
            fig_d.suptitle(f"{pattern_path.stem}: per-q diagnostic", fontsize=12, fontweight="bold", y=0.98)
        else:
            diag_layout_kind = "2d_only"
            fig_d, (ax_al, ax_rc, ax_df) = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"wspace": 0.28})
            _draw_2d_diff_axes(fig_d, ax_al, ax_rc, ax_df)
            fig_d.suptitle(f"{pattern_path.stem}: 2D difference", fontsize=12, fontweight="bold", y=0.98)

        fig_d.tight_layout(rect=[0.02, 0.02, 0.98, 0.99])
        if diag_layout_kind == "combined":
            fig_d.subplots_adjust(top=0.985)
        elif diag_layout_kind == "2d_only":
            fig_d.subplots_adjust(top=0.93)
        fig_d.savefig(diag_path, dpi=200, bbox_inches="tight")
        plt.close(fig_d)
        print(f"Saved per-q diagnostic to {diag_path}")

    # ------------------------------------------------------------------
    # Combined figure: 2D panels + PCA coefficients
    # ------------------------------------------------------------------
    n_cols_top = 3 if have_2d_grid else 2
    fig = plt.figure(figsize=(7 * n_cols_top, 10))
    gs = GridSpec(2, n_cols_top, height_ratios=[1.2, 1], hspace=0.30, wspace=0.35)

    ax_orig = fig.add_subplot(gs[0, 0])
    im_orig = ax_orig.imshow(original_grid, extent=orig_extent, origin="lower", aspect="equal",
                             norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax), cmap="turbo")
    ax_orig.set(xlabel=r"Q$_x$ (Å$^{-1}$)", ylabel=r"Q$_y$ (Å$^{-1}$)", title="Original Raw (unrotated)")
    fig.colorbar(im_orig, ax=ax_orig, label="Intensity", shrink=0.85)

    if interp_layers is not None:
        ax_interp = fig.add_subplot(gs[0, 1])
        shared_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)
        for g, ext in interp_layers:
            im_interp = ax_interp.imshow(g, extent=ext, origin="lower", aspect="equal", norm=shared_norm, cmap="turbo")
        ax_interp.set(xlabel=r"Q$_x$ (Å$^{-1}$)", ylabel=r"Q$_y$ (Å$^{-1}$)",
                      title="Aligned (raw q rotated to sim grid)")
        ax_interp.set_xlim(orig_extent[0], orig_extent[1])
        ax_interp.set_ylim(orig_extent[2], orig_extent[3])
        fig.colorbar(im_interp, ax=ax_interp, label="Intensity", shrink=0.85)

    ax_recon = fig.add_subplot(gs[0, n_cols_top - 1])
    recon_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)
    if recon_layers is not None:
        for g, ext in recon_layers:
            im_recon = ax_recon.imshow(g, extent=ext, origin="lower", aspect="equal", norm=recon_norm, cmap="turbo")
    else:
        im_recon = ax_recon.imshow(recon_grid, extent=recon_extent, origin="lower", aspect="equal",
                                   norm=recon_norm, cmap="turbo")
    ax_recon.set(xlabel=r"Q$_x$ (Å$^{-1}$)", ylabel=r"Q$_y$ (Å$^{-1}$)",
                 title=f"PCA Reconstruction ({U.shape[1]} modes)")
    ax_recon.set_xlim(orig_extent[0], orig_extent[1])
    ax_recon.set_ylim(orig_extent[2], orig_extent[3])
    fig.colorbar(im_recon, ax=ax_recon, label="Intensity", shrink=0.85)

    ax_coeff = fig.add_subplot(gs[1, :])
    coeffs = alpha_xg.reshape(-1)
    x = np.arange(1, len(coeffs) + 1)
    ax_coeff.bar(x, coeffs, yerr=alpha_sigma, capsize=3, color="tab:blue", ecolor="black", linewidth=0.8)
    ax_coeff.set(xlabel="PCA mode", ylabel="Coefficient", title="PCA Coefficients (XGBoost input)")
    ax_coeff.set_xticks(x)

    def _fmt(name):
        val = predictions.get(name, float("nan"))
        s = mc_stats.get(name)
        return f"{name}={val:.3g} [{s['ci_lo']:.3g}, {s['ci_hi']:.3g}]" if s else f"{name}={val:.3g}"

    param_lines = [_fmt(p) for p in param_order]
    if not args.no_rescale:
        if args.rescale_scale is not None:
            param_lines.append(f"scale_fix={float(args.rescale_scale):.3g}")
        else:
            param_lines.append(f"scale={rescale_scale:.3g}")
        param_lines.append(f"bg_sub={float(args.rescale_background):.3g}" if args.rescale_background is not None
                           else f"bg={rescale_bg:.3g}")
    if args.beta is not None:
        param_lines.append(f"beta={float(args.beta):.3g}")
    if rotation_angle_deg is not None:
        param_lines.append(f"rot={rotation_angle_deg:.1f}°")
    ax_coeff.text(0.98, 0.98, "\n".join(param_lines), transform=ax_coeff.transAxes,
                  ha="right", va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})

    fig.suptitle(pattern_path.stem, fontsize=12, fontweight="bold")
    coeff_plot_name = format_plot_filename(pattern_path, model_plot_label)
    fig.savefig(results_dir / coeff_plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {results_dir / coeff_plot_name}")


if __name__ == "__main__":
    main()
