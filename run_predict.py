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
    --beta BETA              Fixed interaction beta; omit to fit jointly (default: fit)
    --structure-factor {rpa,prism}
                             RPA S=1/(1+beta*P) or PRISM S=1/(1+beta*c(q)*P) (default: rpa)
    --prism-length L         Thin-rod length in Å for PRISM c(q); required with --structure-factor prism
    --ridge-lambda LAMBDA    Ridge regularization strength used in affine rescaling/PCA projection (default: 0.01)
    --smooth-sigma SIGMA     Mask-normalized smoothing on aligned intensity grid (default: 1.0)
    --plot-caxis-percentile LOW HIGH
                             Percentile clip on log10(I) for 2D color scale (default: 2 98)
    --plot-caxis-min VMIN    Fixed color scale minimum in log10(I) (overrides percentile low)
    --plot-caxis-max VMAX    Fixed color scale maximum in log10(I) (overrides percentile high)
    --no-plots               Skip figure generation (still saves JSON summary)
    --fast                   Skip Monte Carlo uncertainty propagation (point predictions only)
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import sici

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


def format_scattering_plot_filename(pattern_path: Path, model_label: str) -> str:
    return f"{model_label}_{pattern_path.stem}_scattering_patterns.png"


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

STRUCTURE_FACTOR_CHOICES = ("rpa", "prism")


def thin_rod_form_factor(q: np.ndarray, length: float) -> np.ndarray:
    """Normalized form factor of an infinitely thin rod of length L (Å).

    P_rod(q) = 2 Si(qL)/(qL) - [sin(qL/2)/(qL/2)]^2, with P_rod(0) = 1.
    Used as c(q) in the empirical PRISM structure factor
    (Arleth, Bergström & Pedersen; Pedersen & Schurtenberger).
    """
    q = np.asarray(q, dtype=float)
    x = np.abs(q) * float(length)
    out = np.ones_like(x, dtype=float)
    large = x > 1e-8
    xz = x[large]
    si, _ci = sici(xz)
    half = 0.5 * xz
    sinc_half = np.sin(half) / half
    out[large] = 2.0 * si / xz - sinc_half * sinc_half
    return np.clip(out, 0.0, 1.0)


def sas_structure_factor(
    P: np.ndarray,
    beta: float,
    c: np.ndarray | None = None,
) -> np.ndarray:
    """S(q) = 1 / (1 + beta * c * max(P, 0)).

    ``c`` is 1 for RPA (omit) and the thin-rod form factor for PRISM.
    """
    interaction = np.maximum(P, 0.0)
    if c is not None:
        interaction = np.asarray(c, dtype=float) * interaction
    return 1.0 / (1.0 + beta * interaction)


def _forward_structure_model(
    P: np.ndarray,
    beta: float | None,
    c: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the model-side structure factor to a reconstructed form factor."""
    if beta is None:
        return P
    return P * sas_structure_factor(P, beta, c)


def feature_q_grid(
    n_features: int,
    q_train: np.ndarray | None = None,
    q_mag_ref: np.ndarray | None = None,
) -> np.ndarray | None:
    """Per-feature |q| from the 2D master grid, else the 1D training q-grid."""
    if q_mag_ref is not None:
        q = np.asarray(q_mag_ref, dtype=float).reshape(-1)
    elif q_train is not None:
        q = np.asarray(q_train, dtype=float).reshape(-1)
    else:
        return None
    if len(q) < n_features:
        raise ValueError(
            f"q-grid has {len(q)} points, but PCA expects {n_features}."
        )
    return q[:n_features]


def resolve_structure_c(
    kind: str,
    q: np.ndarray | None,
    prism_length: float | None,
    *,
    apply_structure: bool,
) -> np.ndarray | None:
    """Return PRISM c(q), or None for RPA / form-factor-only."""
    kind = (kind or "rpa").lower()
    if kind not in STRUCTURE_FACTOR_CHOICES:
        raise ValueError(
            f"Unknown --structure-factor {kind!r}; use 'rpa' or 'prism'."
        )
    if not apply_structure or kind == "rpa":
        return None
    if q is None:
        raise ValueError(
            "PRISM structure factor requires a q-grid "
            "(qx_ref.npy/qy_ref.npy or q_values in PCA artifacts)."
        )
    if prism_length is None or float(prism_length) <= 0:
        raise ValueError(
            "PRISM structure factor requires --prism-length > 0 "
            "(infinitely thin rod length in Å)."
        )
    return thin_rod_form_factor(q, float(prism_length))


def structure_factor_label(kind: str) -> str:
    return "PRISM" if (kind or "rpa").lower() == "prism" else "RPA"


DEFAULT_BETA_INIT = 3
BETA_FIT_BOUNDS = (0.1, 10.0)


def project_pattern_masked(
    aligned: np.ndarray, U: np.ndarray, mean: np.ndarray,
    n_components: int, valid_mask: np.ndarray,
    beta: float | None = None,
    fit_beta: bool = False,
    beta_init: float = DEFAULT_BETA_INIT,
    beta_bounds: tuple[float, float] = BETA_FIT_BOUNDS,
    c: np.ndarray | None = None,
    ridge_lambda: float = 0.0,
    max_iter: int = 200, tol: float = 1e-6,
) -> tuple[np.ndarray, float | None, int, dict]:
    """Project with missing-data mask via nonlinear least squares.

    When ``fit_beta`` is True, beta is optimized jointly with alpha.
    ``c`` is the PRISM direct-correlation form factor (None = RPA).
    Optional ``ridge_lambda`` adds ||alpha||^2 regularization (same as rescaling path).
    Returns (alpha_row_vector, beta_used, n_function_evaluations, fit_info).
    """
    n_fit = min(int(n_components), U.shape[1])
    if n_fit < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")
    U_k = U[:, :n_fit]
    aligned_v = aligned[valid_mask]
    mean_v = mean[valid_mask]
    U_v = U_k[valid_mask, :]
    if aligned_v.size == 0:
        raise ValueError("No valid pixels available for PCA projection.")
    c_v = None if c is None else np.asarray(c, dtype=float)[valid_mask]

    apply_structure = fit_beta or beta is not None
    ridge_weight = float(np.sqrt(ridge_lambda)) if ridge_lambda > 0 else 0.0

    def alpha_penalty(alpha: np.ndarray) -> np.ndarray:
        if ridge_weight == 0.0:
            return np.empty(0)
        return ridge_weight * alpha

    def unpack(x: np.ndarray) -> tuple[float | None, np.ndarray]:
        if fit_beta:
            return float(x[0]), x[1:]
        return beta, x

    def residual(x: np.ndarray) -> np.ndarray:
        beta_val, alpha = unpack(x)
        P_v = mean_v + U_v @ alpha
        forward_beta = beta_val if apply_structure else None
        data_resid = aligned_v - _forward_structure_model(P_v, forward_beta, c_v)
        return np.concatenate((data_resid, alpha_penalty(alpha)))

    if fit_beta:
        x0 = np.concatenate(([float(beta_init)], np.zeros(n_fit, dtype=float)))
        lb = np.concatenate(([beta_bounds[0]], np.full(n_fit, -np.inf)))
        ub = np.concatenate(([beta_bounds[1]], np.full(n_fit, np.inf)))
        alpha_start = 1
        ls_bounds = (lb, ub)
    else:
        x0 = np.zeros(n_fit, dtype=float)
        alpha_start = 0
        ls_bounds = (-np.inf, np.inf)

    result = least_squares(
        residual,
        x0=x0,
        bounds=ls_bounds,
        max_nfev=max_iter,
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    beta_out, alpha_flat = unpack(result.x)
    fit_info = {
        "jac": result.jac,
        "fun": result.fun,
        "alpha_slice": slice(alpha_start, alpha_start + n_fit),
        "n_data": aligned_v.size,
        "n_params": result.x.size,
        "fit_beta": fit_beta,
        "beta": beta_out if apply_structure else None,
    }
    return alpha_flat.reshape(1, -1), beta_out if apply_structure else None, int(result.nfev), fit_info


def project_with_rescaling(
    aligned: np.ndarray, U: np.ndarray, mean: np.ndarray,
    n_components: int,
    valid_mask: np.ndarray | None = None,
    ridge_lambda: float = 0.01,
    fixed_background: float | None = None,
    fixed_scale: float | None = None,
    beta: float | None = None,
    fit_beta: bool = False,
    beta_init: float = DEFAULT_BETA_INIT,
    beta_bounds: tuple[float, float] = BETA_FIT_BOUNDS,
    c: np.ndarray | None = None,
    max_iter: int = 200, tol: float = 1e-6,
) -> tuple[np.ndarray, float, float, float | None, int, dict]:
    """Affine rescaling + PCA projection with optional model-side structure factor.

    Minimizes ||scale*p + bg - forward(mean + U[:, :k] @ alpha, beta, c)||^2
    + lam*||alpha||^2, with k = n_components (the XGBoost input size).
    Scale and/or background may be fixed (not optimized).
    When ``fit_beta`` is True, beta is optimized jointly with the other free params.
    ``c`` is the PRISM direct-correlation form factor (None = RPA).
    Returns (alpha, scale, background, beta_used, n_function_evaluations, fit_info).
    """
    n_fit = min(int(n_components), U.shape[1])
    if n_fit < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")
    if valid_mask is None:
        valid_mask = np.ones(len(aligned), dtype=bool)
    p_v = aligned[valid_mask]
    mean_v = mean[valid_mask]
    U_v = U[:, :n_fit][valid_mask, :]
    if p_v.size == 0:
        raise ValueError("No valid pixels available for PCA projection.")

    apply_structure = fit_beta or beta is not None
    c_v = None if c is None else np.asarray(c, dtype=float)[valid_mask]
    model_beta = float(beta_init) if fit_beta else beta
    model0_v = _forward_structure_model(mean_v, model_beta, c_v)
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

    def unpack(x: np.ndarray) -> tuple[float, float, float | None, np.ndarray]:
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
        if fit_beta:
            beta_val = float(x[i])
            i += 1
        else:
            beta_val = beta
        return scale, background, beta_val, x[i:]

    def residual(x: np.ndarray) -> np.ndarray:
        scale, background, beta_val, alpha = unpack(x)
        P_v = mean_v + U_v @ alpha
        forward_beta = beta_val if apply_structure else None
        data_resid = scale * p_v + background - _forward_structure_model(P_v, forward_beta, c_v)
        return np.concatenate((data_resid, alpha_penalty(alpha)))

    x0_parts: list[np.ndarray] = []
    lb_parts: list[np.ndarray] = []
    ub_parts: list[np.ndarray] = []
    if fixed_scale is None:
        x0_parts.append(np.array([scale_init], dtype=float))
        lb_parts.append(np.array([-np.inf]))
        ub_parts.append(np.array([np.inf]))
    if fixed_background is None:
        x0_parts.append(np.array([bg_init], dtype=float))
        lb_parts.append(np.array([-np.inf]))
        ub_parts.append(np.array([np.inf]))
    if fit_beta:
        x0_parts.append(np.array([float(beta_init)], dtype=float))
        lb_parts.append(np.array([beta_bounds[0]]))
        ub_parts.append(np.array([beta_bounds[1]]))
    x0_parts.append(np.zeros(n_fit, dtype=float))
    lb_parts.append(np.full(n_fit, -np.inf))
    ub_parts.append(np.full(n_fit, np.inf))
    x0 = np.concatenate(x0_parts)
    alpha_start = x0.size - n_fit
    ls_bounds = (np.concatenate(lb_parts), np.concatenate(ub_parts))

    result = least_squares(
        residual,
        x0=x0,
        bounds=ls_bounds,
        max_nfev=max_iter,
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    scale, background, beta_out, alpha_flat = unpack(result.x)
    alpha = alpha_flat.reshape(1, -1)
    alpha_slice = slice(alpha_start, alpha_start + n_fit)

    fit_info = {
        "jac": result.jac,
        "fun": result.fun,
        "alpha_slice": alpha_slice,
        "n_data": p_v.size,
        "n_params": result.x.size,
        "fit_beta": fit_beta,
        "beta": beta_out if apply_structure else None,
    }
    return alpha, scale, background, beta_out if apply_structure else None, int(result.nfev), fit_info


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
    c: np.ndarray | None = None,
) -> np.ndarray:
    """Map PCA reconstruction (form factor P in sim space) to raw experimental I.

    Path: structure-factor forward in sim space -> inverse affine -> add back subtracted bg.
    """
    I_sim = _forward_structure_model(
        P_sim, float(beta) if beta is not None else None, c,
    )
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


def _pixel_pitch(coords: np.ndarray) -> float:
    """Typical detector-pixel spacing, ignoring float jitter and inter-detector gaps."""
    uniq = np.unique(np.asarray(coords, dtype=float))
    uniq = uniq[np.isfinite(uniq)]
    if uniq.size < 2:
        return 1.0
    diffs = np.diff(uniq)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    span = float(uniq[-1] - uniq[0])
    # Gaps much smaller than one pixel of a filled pattern are coordinate jitter.
    nominal = span / max(np.sqrt(float(uniq.size)), 2.0)
    pixel_gaps = diffs[diffs >= 0.25 * nominal]
    if pixel_gaps.size == 0:
        return float(np.median(diffs))
    # The short gaps are the pixel pitch; longer ones are spaces between detectors.
    pitch_gaps = pixel_gaps[pixel_gaps <= np.percentile(pixel_gaps, 30)]
    if pitch_gaps.size == 0:
        pitch_gaps = pixel_gaps
    return float(np.median(pitch_gaps))


def _snap_to_pitch(coords: np.ndarray, pitch: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if not np.isfinite(pitch) or pitch <= 0:
        return coords
    origin = float(np.nanmin(coords))
    return origin + np.round((coords - origin) / pitch) * pitch


def _positive_image(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    return np.where(np.isfinite(grid) & (grid > 0), grid, np.nan)


def _mask_q_window(grid: np.ndarray, extent: list[float], q_lo: float | None, q_hi: float | None) -> np.ndarray:
    """Blank pixels outside the plotted |q| window. Beamstop and cuts stay empty."""
    if q_lo is None and q_hi is None:
        return grid
    ny, nx = grid.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    qmag = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
    masked = grid
    if q_lo is not None:
        masked = np.where(qmag >= q_lo, masked, np.nan)
    if q_hi is not None:
        masked = np.where(qmag <= q_hi, masked, np.nan)
    return masked


def _square_regions_or_none(qx: np.ndarray, qy: np.ndarray) -> list[tuple[int, int]] | None:
    """Return detector blocks only when each block is a filled square raster."""
    n = len(qx)
    regions = detect_detector_regions(qx, qy)
    if regions == [(0, n)]:
        side = int(round(np.sqrt(n)))
        if side * side != n:
            return None
        qx_span = float(qx[:side].max() - qx[:side].min())
        qy_span = float(qy[:side].max() - qy[:side].min())
        full_qx = float(np.max(qx) - np.min(qx))
        full_qy = float(np.max(qy) - np.min(qy))
        flat_row = (
            qx_span / max(full_qx, 1e-12) < 0.02
            or qy_span / max(full_qy, 1e-12) < 0.02
        )
        if not flat_row:
            return None
    return regions


def _mesh_layer(
    qx: np.ndarray, qy: np.ndarray, values: np.ndarray, dx: float, dy: float,
) -> tuple[np.ndarray, list[float]]:
    """Fill a regular mesh at the snapped pixel pitch (no empty-bin seams)."""
    x0, x1 = float(np.min(qx)), float(np.max(qx))
    y0, y1 = float(np.min(qy)), float(np.max(qy))
    nx = int(np.round((x1 - x0) / dx)) + 1 if dx > 0 else 1
    ny = int(np.round((y1 - y0) / dy)) + 1 if dy > 0 else 1
    nx = max(nx, 1)
    ny = max(ny, 1)
    if nx * ny > max(4_000_000, 8 * int(values.size)):
        grid, ux, uy = grid_pattern_2d(qx, qy, values)
        return _positive_image(grid), [float(ux[0]), float(ux[-1]), float(uy[0]), float(uy[-1])]
    ux = x0 + dx * np.arange(nx)
    uy = y0 + dy * np.arange(ny)
    flat = np.clip(np.rint((qy - y0) / dy).astype(int), 0, ny - 1) * nx
    flat = flat + np.clip(np.rint((qx - x0) / dx).astype(int), 0, nx - 1)
    sums = np.bincount(flat, weights=np.nan_to_num(values, nan=0.0), minlength=nx * ny)
    counts = np.bincount(flat, minlength=nx * ny).astype(float)
    averaged = np.full(nx * ny, np.nan)
    filled = counts > 0
    averaged[filled] = sums[filled] / counts[filled]
    grid = _positive_image(averaged.reshape(ny, nx))
    return grid, [float(ux[0]), float(ux[-1]), float(uy[0]), float(uy[-1])]


def pattern_imshow_layers(
    qx: np.ndarray,
    qy: np.ndarray,
    values: np.ndarray,
    q_lo: float | None = None,
    q_hi: float | None = None,
) -> tuple[list[tuple[np.ndarray, list[float]]], list[float]]:
    """Square detector blocks, or a pitch-snapped mesh, for seam-free imshow.

    Coordinates are snapped to the median pixel spacing so near-duplicate q
    values do not open empty rows. Non-positive pixels and the |q| window
    stay blank.
    """
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(values)
    qx, qy, values = qx[finite], qy[finite], values[finite]
    if qx.size == 0:
        empty = np.full((2, 2), np.nan)
        extent = [-1.0, 1.0, -1.0, 1.0]
        return [(empty, extent)], extent

    dx = _pixel_pitch(qx)
    dy = _pixel_pitch(qy)
    qx_s = _snap_to_pitch(qx, dx)
    qy_s = _snap_to_pitch(qy, dy)

    layers: list[tuple[np.ndarray, list[float]]] = []
    regions = _square_regions_or_none(qx_s, qy_s)
    if regions is not None:
        try:
            for s, e in regions:
                grid, ux, uy = grid_region_2d(qx_s[s:e], qy_s[s:e], values[s:e])
                extent = [float(ux[0]), float(ux[-1]), float(uy[0]), float(uy[-1])]
                layers.append((_mask_q_window(_positive_image(grid), extent, q_lo, q_hi), extent))
        except ValueError:
            layers = []
    if not layers:
        grid, extent = _mesh_layer(qx_s, qy_s, values, dx, dy)
        layers = [(_mask_q_window(grid, extent, q_lo, q_hi), extent)]

    x0 = min(ext[0] for _grid, ext in layers)
    x1 = max(ext[1] for _grid, ext in layers)
    y0 = min(ext[2] for _grid, ext in layers)
    y1 = max(ext[3] for _grid, ext in layers)
    return layers, [x0, x1, y0, y1]


def _shared_q_ticks(lo: float, hi: float, target: int = 5) -> np.ndarray:
    """0-centered 1-2-5 ticks shared by qx and qy on a square pattern."""
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        return np.array([lo])
    raw = span / max(target - 1, 1)
    mag = 10.0 ** np.floor(np.log10(max(raw, 1e-30)))
    step = mag
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * mag
        if span / step <= target:
            break
    if lo < 0.0 < hi:
        n = int(np.floor(max(abs(lo), abs(hi)) / step + 1e-9))
        ticks = step * np.arange(-n, n + 1)
    else:
        start = np.ceil((lo - 1e-12 * span) / step) * step
        ticks = np.arange(start, hi + 0.5 * step, step)
    ticks = np.asarray(ticks, dtype=float)
    ticks = ticks[(ticks >= lo - 1e-8 * span) & (ticks <= hi + 1e-8 * span)]
    if ticks.size == 0:
        ticks = np.array([0.5 * (lo + hi)])
    return ticks


def _style_sas_axes(ax, extent, *, title: str = "", ylabel: bool = True) -> None:
    """Square SAS panel with identical qx and qy tick marks and no grid."""
    x0, x1, y0, y1 = (float(v) for v in extent)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    span = max(x1 - x0, y1 - y0, 1e-12)
    ticks = _shared_q_ticks(min(x0, y0), max(x1, y1))
    shared = ticks[
        (ticks >= max(x0, y0) - 1e-6 * span)
        & (ticks <= min(x1, y1) + 1e-6 * span)
    ]
    if shared.size < 2:
        shared = ticks[(ticks >= x0 - 1e-6 * span) & (ticks <= x1 + 1e-6 * span)]
    ax.set_xticks(shared)
    ax.set_yticks(shared)
    ax.tick_params(direction="out", top=False, right=False, labelsize=8)
    ax.grid(False)
    ax.set_xlabel(r"Q$_x$ (Å$^{-1}$)")
    if ylabel:
        ax.set_ylabel(r"Q$_y$ (Å$^{-1}$)")
    else:
        ax.tick_params(axis="y", labelleft=False)
    if title:
        ax.set_title(title)


def _style_line_axes(ax) -> None:
    ax.grid(False)
    ax.tick_params(direction="out", top=False, right=False, labelsize=9, which="both")


def _style_decade_log_axis(ax, axis: str) -> None:
    """Label only powers of 10; keep minor ticks unlabeled."""
    axis_obj = ax.xaxis if axis == "x" else ax.yaxis
    axis_obj.set_major_locator(LogLocator(base=10.0))
    axis_obj.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis_obj.set_major_formatter(LogFormatterSciNotation())
    axis_obj.set_minor_formatter(NullFormatter())


def _imshow_layers(ax, layers, norm, *, cmap: str = "turbo", vmin: float | None = None, vmax: float | None = None):
    im = None
    for grid, ext in layers:
        kwargs = dict(
            extent=ext, origin="lower", aspect="equal", cmap=cmap, interpolation="antialiased",
        )
        if norm is not None:
            kwargs["norm"] = norm
        else:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        im = ax.imshow(grid, **kwargs)
    return im


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


def _robust_log_intensity_limits(
    arrays: list[np.ndarray],
    p_low: float = 2.0,
    p_high: float = 98.0,
    caxis_min_log10: float | None = None,
    caxis_max_log10: float | None = None,
) -> tuple[float, float]:
    """Percentile-based log intensity limits for shared 2D color scales."""
    chunks = []
    for arr in arrays:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=float).ravel()
        if flat.size > 0:
            chunks.append(flat)
    if not chunks:
        return 1e-6, 1.0

    pool = np.concatenate(chunks)
    valid = pool[np.isfinite(pool) & (pool > 0)]
    if valid.size == 0:
        return 1e-6, 1.0

    log_vals = np.log10(valid)
    if caxis_min_log10 is not None:
        lo_log = float(caxis_min_log10)
    else:
        lo_log = float(np.percentile(log_vals, p_low))
    if caxis_max_log10 is not None:
        hi_log = float(caxis_max_log10)
    else:
        hi_log = float(np.percentile(log_vals, p_high))
    if hi_log <= lo_log:
        lo_log, hi_log = float(log_vals.min()), float(log_vals.max())
    vmin = max(10.0 ** lo_log, 1e-10)
    vmax = max(10.0 ** hi_log, vmin * 1.01)
    return vmin, vmax


def _robust_symmetric_diff_limit(diff: np.ndarray, p_abs: float = 98.0) -> float:
    """Robust symmetric limit for difference-panel colormap."""
    abs_vals = np.abs(np.asarray(diff, dtype=float))
    finite = abs_vals[np.isfinite(abs_vals)]
    if finite.size == 0:
        return 1.0
    dmax = float(np.percentile(finite, p_abs))
    if not np.isfinite(dmax) or dmax <= 0:
        dmax = float(finite.max())
    return max(dmax, 1e-12)


def _attach_panel_colorbar(
    fig: plt.Figure,
    ax,
    mappable,
    *,
    label: str,
    log: bool = False,
) -> plt.colorbar:
    """Colorbar locked to one square SAS panel, same height as the image.

    Uses a divider axes so later layout passes cannot leave the bar floating.
    """
    ax.set_aspect("equal", adjustable="box")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8, direction="out")
    if log:
        cb.locator = LogLocator(base=10.0)
        cb.formatter = LogFormatterSciNotation()
        cb.update_ticks()
    return cb


@dataclass
class PredictionPlotGrids:
    """Layer stacks and color limits for predict scattering figures."""

    original_layers: list[tuple[np.ndarray, list[float]]]
    aligned_layers: list[tuple[np.ndarray, list[float]]]
    recon_layers: list[tuple[np.ndarray, list[float]]]
    diff_layers: list[tuple[np.ndarray, list[float]]]
    display_extent: list[float]
    shared_vmin: float
    shared_vmax: float
    diff_limit: float


def _layers_from_azimuthal_grid(
    q_train: np.ndarray,
    values_1d: np.ndarray,
    valid_mask: np.ndarray,
    extent_axes: list[float],
    q_lo: float,
    q_hi: float,
    *,
    n_recon: int = 256,
) -> list[tuple[np.ndarray, list[float]]]:
    """Build a single imshow layer by azimuthally symmetric |q| interpolation."""
    gqx = np.array([extent_axes[0], extent_axes[1]], dtype=float)
    gqy = np.array([extent_axes[2], extent_axes[3]], dtype=float)
    n_side = max(n_recon, len(gqx), len(gqy))
    recon_qx = np.linspace(gqx[0], gqx[-1], n_side)
    recon_qy = np.linspace(gqy[0], gqy[-1], n_side)
    rqx_2d, rqy_2d = np.meshgrid(recon_qx, recon_qy)
    rq_mag = np.sqrt(rqx_2d**2 + rqy_2d**2)
    q_u, y_u = _average_duplicate_q_for_interp(q_train[valid_mask], values_1d[valid_mask])
    if q_u.size >= 2:
        flat = np.interp(rq_mag.ravel(), q_u, y_u)
        in_q = (rq_mag.ravel() >= q_u.min()) & (rq_mag.ravel() <= q_u.max())
        flat = np.where(in_q, flat, np.nan)
    else:
        flat = np.full(rq_mag.size, np.nan)
    grid = flat.reshape(rq_mag.shape)
    grid = np.where((rq_mag >= q_lo) & (rq_mag <= q_hi), grid, np.nan)
    grid = np.where(grid > 0, grid, np.nan)
    extent = [recon_qx[0], recon_qx[-1], recon_qy[0], recon_qy[-1]]
    return [(grid, extent)]


def build_prediction_plot_grids(
    *,
    have_2d_grid: bool,
    qx_raw: np.ndarray,
    qy_raw: np.ndarray,
    p_raw: np.ndarray,
    qx_ref: np.ndarray | None,
    qy_ref: np.ndarray | None,
    q_mag_ref: np.ndarray | None,
    sim_regions,
    q_train: np.ndarray | None,
    valid_mask: np.ndarray,
    aligned_masked: np.ndarray,
    recon_exp_masked: np.ndarray,
    p_low: float,
    p_high: float,
    caxis_min_log10: float | None,
    caxis_max_log10: float | None,
) -> PredictionPlotGrids:
    q_mag_raw = np.sqrt(qx_raw**2 + qy_raw**2)
    q_lo = float(q_train.min()) if q_train is not None else float(q_mag_raw.min())
    q_hi = float(q_train.max()) if q_train is not None else float(q_mag_raw.max())

    in_range = (q_mag_raw >= q_lo) & (q_mag_raw <= q_hi) & np.isfinite(q_mag_raw)
    if np.any(in_range):
        display_extent = [
            float(np.min(qx_raw[in_range])),
            float(np.max(qx_raw[in_range])),
            float(np.min(qy_raw[in_range])),
            float(np.max(qy_raw[in_range])),
        ]
    else:
        display_extent = [
            float(np.min(qx_raw)), float(np.max(qx_raw)),
            float(np.min(qy_raw)), float(np.max(qy_raw)),
        ]

    original_layers: list[tuple[np.ndarray, list[float]]] = []
    aligned_layers: list[tuple[np.ndarray, list[float]]] = []
    recon_layers: list[tuple[np.ndarray, list[float]]] = []

    if have_2d_grid and qx_ref is not None and qy_ref is not None and q_mag_ref is not None:
        raw_on_ref = griddata(
            (qx_raw, qy_raw), p_raw, (qx_ref, qy_ref), method="linear", fill_value=np.nan,
        )
        raw_on_ref = np.asarray(raw_on_ref, dtype=float)
        raw_on_ref = np.where((q_mag_ref >= q_lo) & (q_mag_ref <= q_hi), raw_on_ref, np.nan)
        region_order = sorted(
            sim_regions,
            key=lambda r: -(qx_ref[r[0]:r[1]].max() - qx_ref[r[0]:r[1]].min()),
        )
        for s, e in region_order:
            qxr, qyr = qx_ref[s:e], qy_ref[s:e]
            g_raw, ux, uy = grid_region_2d(qxr, qyr, raw_on_ref[s:e])
            ext = [float(ux[0]), float(ux[-1]), float(uy[0]), float(uy[-1])]
            original_layers.append((_positive_image(g_raw), ext))
            g_aln, ux, uy = grid_region_2d(qxr, qyr, aligned_masked[s:e])
            ext = [float(ux[0]), float(ux[-1]), float(uy[0]), float(uy[-1])]
            aligned_layers.append((_positive_image(g_aln), ext))
            g_rec, ux, uy = grid_region_2d(qxr, qyr, recon_exp_masked[s:e])
            recon_layers.append((_positive_image(g_rec), ext))
    else:
        original_layers, _full_extent = pattern_imshow_layers(
            qx_raw, qy_raw, p_raw, q_lo=q_lo, q_hi=q_hi,
        )
        if q_train is not None and np.any(valid_mask):
            aligned_layers = _layers_from_azimuthal_grid(
                q_train, aligned_masked, valid_mask, display_extent, q_lo, q_hi,
            )
            recon_layers = _layers_from_azimuthal_grid(
                q_train, recon_exp_masked, valid_mask, display_extent, q_lo, q_hi,
            )
        else:
            aligned_layers = original_layers
            recon_layers = original_layers

    intensity_arrays: list[np.ndarray] = [grid for grid, _ext in original_layers]
    intensity_arrays.extend(grid for grid, _ext in aligned_layers)
    intensity_arrays.extend(grid for grid, _ext in recon_layers)
    shared_vmin, shared_vmax = _robust_log_intensity_limits(
        intensity_arrays,
        p_low=p_low,
        p_high=p_high,
        caxis_min_log10=caxis_min_log10,
        caxis_max_log10=caxis_max_log10,
    )

    diff_layers = [
        (g_aln - g_rec, ext)
        for (g_aln, ext), (g_rec, _ext2) in zip(aligned_layers, recon_layers)
    ]
    diff_sample = diff_layers[0][0] if diff_layers else recon_exp_masked - aligned_masked
    diff_limit = _robust_symmetric_diff_limit(diff_sample, p_abs=p_high)

    return PredictionPlotGrids(
        original_layers=original_layers,
        aligned_layers=aligned_layers,
        recon_layers=recon_layers,
        diff_layers=diff_layers,
        display_extent=display_extent,
        shared_vmin=shared_vmin,
        shared_vmax=shared_vmax,
        diff_limit=diff_limit,
    )


def save_scattering_patterns_figure(
    output_path: Path,
    *,
    pattern_stem: str,
    grids: PredictionPlotGrids,
) -> None:
    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.6),
        gridspec_kw={"wspace": 0.55, "left": 0.04, "right": 0.98, "top": 0.88, "bottom": 0.14},
    )
    log_norm = LogNorm(vmin=grids.shared_vmin, vmax=grids.shared_vmax)
    diff_norm = plt.Normalize(vmin=-grids.diff_limit, vmax=grids.diff_limit)
    extent = grids.display_extent
    titles = ("Original raw", "Aligned", "Reconstructed", "Difference")
    layer_sets = (
        grids.original_layers,
        grids.aligned_layers,
        grids.recon_layers,
        grids.diff_layers,
    )
    intensity_im = None
    for ax, layers, title in zip(axes, layer_sets, titles):
        if title == "Difference":
            im = _imshow_layers(ax, layers, diff_norm, cmap="RdBu_r")
            if im is not None:
                _attach_panel_colorbar(fig, ax, im, label="Δ intensity", log=False)
        else:
            im = _imshow_layers(ax, layers, log_norm)
            if title == "Reconstructed":
                intensity_im = im
        _style_sas_axes(ax, extent, title=title, ylabel=True)
    if intensity_im is not None:
        _attach_panel_colorbar(fig, axes[2], intensity_im, label="Intensity", log=True)
    fig.suptitle(f"{pattern_stem}: scattering patterns", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_coefficients_figure(
    output_path: Path,
    *,
    pattern_stem: str,
    per_q_data: dict | None,
    alpha_xg: np.ndarray,
    alpha_sigma: np.ndarray,
    predictions: dict[str, float],
    mc_stats: dict[str, dict],
    param_order: list[str],
    uncertainty_lines: list[str],
    n_components: int,
    no_rescale: bool,
    rescale_scale: float,
    rescale_scale_arg: float | None,
    rescale_background_arg: float | None,
    rescale_bg: float,
    beta: float | None,
    fit_beta: bool,
    rotation_angle_deg: float | None,
) -> None:
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.05, 1.0], hspace=0.28,
        left=0.10, right=0.97, top=0.93, bottom=0.07,
    )
    ax_coeff = fig.add_subplot(gs[0, 0])
    coeffs = alpha_xg.reshape(-1)
    x = np.arange(1, len(coeffs) + 1)
    ax_coeff.bar(
        x, coeffs, yerr=alpha_sigma, capsize=3,
        color="tab:blue", ecolor="black", linewidth=0.8,
    )
    ax_coeff.set(
        xlabel="PCA mode",
        ylabel="Coefficient",
        title=f"PCA coefficients ({n_components} modes, XGBoost input)",
    )
    ax_coeff.set_xticks(x)
    _style_line_axes(ax_coeff)

    def _fmt(name: str) -> str:
        val = predictions.get(name, float("nan"))
        s = mc_stats.get(name)
        return (
            f"{name}={val:.3g} [{s['ci_lo']:.3g}, {s['ci_hi']:.3g}]"
            if s else f"{name}={val:.3g}"
        )

    param_lines = [_fmt(p) for p in param_order if p in predictions]
    if not no_rescale:
        if rescale_scale_arg is not None:
            param_lines.append(f"scale_fix={float(rescale_scale_arg):.3g}")
        else:
            param_lines.append(f"scale={rescale_scale:.3g}")
        param_lines.append(
            f"bg_sub={float(rescale_background_arg):.3g}"
            if rescale_background_arg is not None
            else f"bg={rescale_bg:.3g}"
        )
    if beta is not None:
        beta_tag = "beta_fit" if fit_beta else "beta"
        param_lines.append(f"{beta_tag}={float(beta):.3g}")
    if rotation_angle_deg is not None:
        param_lines.append(f"rot={rotation_angle_deg:.1f}°")
    if param_lines:
        ax_coeff.text(
            0.98, 0.98, "\n".join(param_lines), transform=ax_coeff.transAxes,
            ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax_i = fig.add_subplot(gs[1, 0], sharex=None)
    ax_e = fig.add_subplot(gs[2, 0], sharex=ax_i)
    if per_q_data is not None:
        q_p = np.asarray(per_q_data["q_pos"], dtype=float)
        Ie = np.asarray(per_q_data["I_exp_pos"], dtype=float)
        Rc = np.asarray(per_q_data["recon_exp_pos"], dtype=float)
        order = np.argsort(q_p)
        q_p, Ie, Rc = q_p[order], Ie[order], Rc[order]
        ax_i.loglog(q_p, Ie, ".", alpha=0.35, ms=1, label="Experimental", color="C0")
        ax_i.loglog(q_p, Rc, ".", alpha=0.35, ms=1, label="Reconstruction", color="C1")
        ax_i.set_ylabel("Intensity")
        ax_i.legend(loc="best", fontsize=8, frameon=False)
        ax_i.set_title("Per-q comparison")
        _style_line_axes(ax_i)
        _style_decade_log_axis(ax_i, "x")
        _style_decade_log_axis(ax_i, "y")

        rel_err = (Rc - Ie) / Ie
        finite_e = rel_err[np.isfinite(rel_err)]
        rmse_l = float(np.sqrt(np.mean(finite_e**2))) if finite_e.size > 0 else float("nan")
        mean_b = float(np.mean(finite_e)) if finite_e.size > 0 else float("nan")
        ax_e.scatter(q_p, rel_err, s=1, alpha=0.4, color="C3")
        ax_e.axhline(0.0, color="k", ls="--", lw=1)
        ax_e.set_xscale("log")
        ax_e.set_xlabel(r"|q| (Å$^{-1}$)")
        ax_e.set_ylabel("Relative error")
        err99 = np.nanpercentile(np.abs(np.where(np.isfinite(rel_err), rel_err, np.nan)), 99)
        y_lim = min(2.0, max(0.5, err99 * 1.1)) if np.isfinite(err99) else 1.0
        ax_e.set_ylim(-y_lim, y_lim)
        _style_line_axes(ax_e)
        _style_decade_log_axis(ax_e, "x")
        error_summary = "\n".join(
            [f"Rel. err. RMSE = {rmse_l:.4f}", f"Mean bias = {mean_b:+.4f}", *uncertainty_lines]
        )
        ax_e.text(
            0.98, 0.03, error_summary, transform=ax_e.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    else:
        ax_i.set_visible(False)
        ax_e.text(
            0.5, 0.5, "Per-q diagnostics unavailable",
            transform=ax_e.transAxes, ha="center", va="center", fontsize=10,
        )
        ax_e.set_axis_off()

    fig.suptitle(f"{pattern_stem}: PCA coefficients & fit quality", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _print_section(title: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def _log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def structure_c_for_features(
    kind: str,
    prism_length: float | None,
    n_features: int,
    q_train: np.ndarray | None,
    q_mag_ref: np.ndarray | None,
    apply_structure: bool,
) -> np.ndarray | None:
    q = feature_q_grid(n_features, q_train=q_train, q_mag_ref=q_mag_ref)
    return resolve_structure_c(kind, q, prism_length, apply_structure=apply_structure)


def add_structure_factor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--beta", type=float, default=None,
        help=(
            "Fixed interaction beta (RPA: S=1/(1+beta*P); "
            "PRISM: S=1/(1+beta*c(q)*P)); omit to fit jointly; 0 for form factor only"
        ),
    )
    parser.add_argument(
        "--structure-factor",
        choices=STRUCTURE_FACTOR_CHOICES,
        default="rpa",
        dest="structure_factor",
        help="SAS structure factor: rpa or prism (default: rpa)",
    )
    parser.add_argument(
        "--prism-length",
        type=float,
        default=None,
        dest="prism_length",
        help="Thin-rod length L in Å for PRISM c(q); required with --structure-factor prism",
    )


def validate_structure_factor_args(
    structure_factor: str,
    prism_length: float | None,
    beta: float | None,
    fit_beta: bool,
) -> str:
    kind = (structure_factor or "rpa").lower()
    if kind not in STRUCTURE_FACTOR_CHOICES:
        raise ValueError(
            f"Unknown --structure-factor {kind!r}; use 'rpa' or 'prism'."
        )
    if prism_length is not None and prism_length <= 0:
        raise ValueError("--prism-length must be positive.")
    apply_structure = fit_beta or beta is not None
    if apply_structure and kind == "prism" and (prism_length is None or prism_length <= 0):
        raise ValueError(
            "PRISM structure factor requires --prism-length > 0 (thin-rod length in Å)."
        )
    return kind


def resolve_beta_settings(beta_arg: float | None) -> tuple[float | None, bool]:
    if beta_arg is not None and beta_arg < 0:
        raise ValueError(f"Invalid --beta {beta_arg}: must be non-negative.")
    if beta_arg is not None and beta_arg == 0:
        return None, False
    if beta_arg is not None:
        return float(beta_arg), False
    return None, True


@dataclass
class PredictBundle:
    U: np.ndarray
    mean: np.ndarray
    n_features: int
    q_train: np.ndarray | None
    qx_ref: np.ndarray | None
    qy_ref: np.ndarray | None
    q_mag_ref: np.ndarray | None
    have_2d_grid: bool
    sim_regions: list[tuple[int, int]]
    models: dict
    n_components: int
    model_plot_label: str
    pca_results_dir: Path
    models_dir: Path


@dataclass
class PredictOptions:
    beamstop_qmin: float
    q_max: float | None
    rescale_background: float | None
    rescale_scale: float | None
    beta: float | None
    fit_beta: bool
    no_rescale: bool
    ridge_lambda: float
    smooth_sigma: float
    structure_factor: str = "rpa"
    prism_length: float | None = None
    beta_init: float = DEFAULT_BETA_INIT


def load_predict_bundle(pca_dir: str | Path, models_dir: str | Path) -> PredictBundle:
    pca_results_dir = Path(pca_dir)
    pca_file = pca_results_dir / "pca_components.pkl"
    models_dir_path = Path(models_dir) if models_dir else pca_results_dir / "models"
    models_file = models_dir_path / "parameter_models.pkl"

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
        sim_regions = []

    with open(models_file, "rb") as f:
        model_bundle = pickle.load(f)
    n_components = min(model_bundle.get("n_components", U.shape[1]), U.shape[1])
    models = model_bundle.get("models", {})
    if not models:
        raise ValueError("No parameter models found in parameter_models.pkl")

    return PredictBundle(
        U=U,
        mean=mean,
        n_features=n_features,
        q_train=q_train,
        qx_ref=qx_ref,
        qy_ref=qy_ref,
        q_mag_ref=q_mag_ref,
        have_2d_grid=have_2d_grid,
        sim_regions=sim_regions,
        models=models,
        n_components=n_components,
        model_plot_label=models_dir_path.name,
        pca_results_dir=pca_results_dir,
        models_dir=models_dir_path,
    )


def predict_pattern(
    pattern_path: Path,
    bundle: PredictBundle,
    options: PredictOptions,
    *,
    quiet: bool = False,
    fast: bool = False,
) -> dict:
    """Predict parameters for one pattern. Returns a JSON-serializable summary dict."""
    pattern_path = Path(pattern_path)
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

    U = bundle.U
    mean = bundle.mean
    n_features = bundle.n_features
    q_train = bundle.q_train
    qx_ref = bundle.qx_ref
    qy_ref = bundle.qy_ref
    q_mag_ref = bundle.q_mag_ref
    have_2d_grid = bundle.have_2d_grid
    sim_regions = bundle.sim_regions
    models = bundle.models
    n_components = bundle.n_components

    beta = options.beta
    fit_beta = options.fit_beta
    beamstop_qmin = options.beamstop_qmin
    sf_kind = validate_structure_factor_args(
        options.structure_factor, options.prism_length, beta, fit_beta,
    )
    sf_label = structure_factor_label(sf_kind)
    c = structure_c_for_features(
        sf_kind, options.prism_length, n_features, q_train, q_mag_ref,
        apply_structure=fit_beta or beta is not None,
    )

    _print_section("Pattern alignment", quiet=quiet)
    qx_raw, qy_raw, p_raw = load_pattern_raw(pattern_path)

    rotation_angle_deg = None
    if have_2d_grid:
        angle_q_min = beamstop_qmin
        angle_q_max = float(options.q_max) if options.q_max is not None else None
        p_for_angle = p_raw.astype(float, copy=True)
        if options.rescale_background is not None:
            p_for_angle -= float(options.rescale_background)

        phi_0, angle_stats = calculate_annular_harmonic_angle(
            qx_raw, qy_raw, p_for_angle,
            q_min=angle_q_min, q_max=angle_q_max,
        )
        rotation_angle = np.pi / 2 - phi_0
        rotation_angle_deg = float(np.degrees(rotation_angle))
        angle_q_max_label = f"{angle_q_max:.6g}" if angle_q_max is not None else "data max"
        _log(
            f"Annular-harmonic orientation (q=[{angle_q_min:.6g}, {angle_q_max_label}]), "
            f"pixels={int(angle_stats.get('pixels', 0))}, rings={int(angle_stats.get('rings', 0))}, "
            f"R2={float(angle_stats.get('strength', 0.0)):.4g}, "
            f"coverage={float(angle_stats.get('coverage', 0.0)):.3g}",
            quiet=quiet,
        )
        _log(
            f"Director phi_0 = {np.degrees(phi_0):.3f} deg, "
            f"raw-coordinate rotation = {rotation_angle_deg:.3f} deg",
            quiet=quiet,
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
        if options.q_max is not None:
            valid_mask &= q_mag_ref <= float(options.q_max)
        if not np.any(valid_mask):
            valid_mask = interp_valid & np.isfinite(aligned) & (q_mag_ref >= beamstop_qmin)
            if options.q_max is not None:
                valid_mask &= q_mag_ref <= float(options.q_max)
        if options.smooth_sigma > 0:
            aligned = smooth_reference_grid_values(
                qx_ref, qy_ref, aligned, valid_mask, sim_regions,
                sigma=float(options.smooth_sigma),
            )
            _log(
                f"Applied mask-normalized Gaussian smoothing: sigma={float(options.smooth_sigma):.3g} px",
                quiet=quiet,
            )
    else:
        q_sorted, p_sorted, q_min_valid = load_pattern_sorted(
            pattern_path,
            q_min_override=beamstop_qmin if beamstop_qmin > 0 else None,
            q_max_override=float(options.q_max) if options.q_max is not None else None,
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

    _print_section("Preprocessing & PCA fit", quiet=quiet)
    aligned_for_fit = aligned.copy()
    if options.rescale_background is not None:
        aligned_for_fit -= float(options.rescale_background)
        _log(f"Subtracted fixed background {float(options.rescale_background):.6g}", quiet=quiet)
    nonpositive_mask = np.isfinite(aligned_for_fit) & (aligned_for_fit <= 0.0)
    if np.any(nonpositive_mask):
        aligned_for_fit[nonpositive_mask] = np.nan
        valid_mask = valid_mask & (~nonpositive_mask)
        _log(
            f"Marked {int(np.count_nonzero(nonpositive_mask))} non-positive intensities as N/A for fitting",
            quiet=quiet,
        )

    fixed_bg = 0.0 if options.rescale_background is not None else None
    fixed_scale = float(options.rescale_scale) if options.rescale_scale is not None else None
    if fixed_scale is not None:
        _log(f"Fixed affine scale: {fixed_scale:.6g}", quiet=quiet)
    if fit_beta:
        extra = (
            f", thin-rod L={options.prism_length:g} Å" if sf_kind == "prism" else ""
        )
        _log(
            f"Fitting {sf_label} beta jointly (init={float(options.beta_init):g}, "
            f"bounds={BETA_FIT_BOUNDS}){extra}",
            quiet=quiet,
        )
    elif beta is not None:
        extra = (
            f", thin-rod L={options.prism_length:g} Å" if sf_kind == "prism" else ""
        )
        _log(f"Using fixed {sf_label} beta: {beta:g}{extra}", quiet=quiet)
    else:
        _log("Form factor only (no structure factor)", quiet=quiet)

    _log(
        f"Fitting {n_components} PCA modes (XGBoost n_components; PCA basis has {U.shape[1]})",
        quiet=quiet,
    )
    if options.no_rescale:
        alpha_full, beta, fit_evals, fit_info = project_pattern_masked(
            aligned_for_fit, U, mean, n_components, valid_mask,
            beta=beta, fit_beta=fit_beta, c=c,
            beta_init=float(options.beta_init),
            ridge_lambda=float(options.ridge_lambda),
        )
        rescale_scale, rescale_bg = 1.0, 0.0
    else:
        alpha_full, rescale_scale, rescale_bg, beta, fit_evals, fit_info = project_with_rescaling(
            aligned_for_fit, U, mean, n_components, valid_mask,
            ridge_lambda=float(options.ridge_lambda),
            fixed_background=fixed_bg,
            fixed_scale=fixed_scale,
            beta=beta,
            fit_beta=fit_beta,
            beta_init=float(options.beta_init),
            c=c,
        )
        _log(f"Affine rescaling: scale={rescale_scale:.6g}, background={rescale_bg:.6g}", quiet=quiet)

    if beta is not None:
        beta_label = "Fitted" if fit_beta else "Fixed"
        _log(f"{beta_label} {sf_label} beta: {beta:.6g} ({fit_evals} function evaluation(s))", quiet=quiet)

    alpha_xg = alpha_full[:, :n_components]
    param_order = ["radius", "length", "n_cyl", "stretch"]
    predictions: dict[str, float] = {}
    for param in param_order:
        if param in models and "model" in models[param]:
            predictions[param] = float(models[param]["model"].predict(alpha_xg)[0])

    _print_section("PCA coefficients (XGBoost input)", quiet=quiet)
    _log(str(alpha_xg.reshape(-1)), quiet=quiet)

    reconstructed_1d = reconstruct_pattern(alpha_full, U, mean, n_components)
    recon_exp_1d = sim_to_experimental(
        reconstructed_1d, rescale_scale, rescale_bg, options.rescale_background, beta, c,
    )
    aligned_masked = np.where(valid_mask, aligned, np.nan)
    recon_exp_masked = np.where(valid_mask, recon_exp_1d, np.nan)

    mc_stats: dict[str, dict] = {}
    alpha_sigma = np.zeros(n_components, dtype=float)
    if fast:
        for param in predictions:
            point = predictions[param]
            mc_stats[param] = {
                "s_pca": 0.0,
                "s_model": 0.0,
                "sigma_total": 0.0,
                "ci_lo": point,
                "ci_hi": point,
            }
    else:
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

    _print_section("Predicted parameters (95% CI)", quiet=quiet)
    for param in param_order:
        if param in mc_stats:
            s = mc_stats[param]
            _log(
                f"  {param}: {predictions[param]:.6g}  (95% CI [{s['ci_lo']:.3g}, {s['ci_hi']:.3g}])",
                quiet=quiet,
            )

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
    _log(f"  RMSE relative error: {rmse_relative_error:.6g}", quiet=quiet)

    return {
        "pattern_file": str(pattern_path),
        "pattern_stem": pattern_path.stem,
        "pca_dir": str(bundle.pca_results_dir),
        "models_dir": str(bundle.models_dir),
        "rescale_scale": float(rescale_scale),
        "rescale_background": float(options.rescale_background) if options.rescale_background is not None else None,
        "rescale_background_fit": float(rescale_bg),
        "beta": float(beta) if beta is not None else None,
        "beta_fit": bool(fit_beta),
        "structure_factor": sf_kind,
        "prism_length": (
            float(options.prism_length)
            if sf_kind == "prism" and options.prism_length is not None
            else None
        ),
        "q_min": float(beamstop_qmin),
        "q_max": float(options.q_max) if options.q_max is not None else None,
        "n_components": int(n_components),
        "rotation_angle_deg": rotation_angle_deg,
        "rmse_relative_error": rmse_relative_error,
        "pca_coefficients": alpha_xg.reshape(-1).tolist(),
        "predictions": {
            p: {
                "value": float(predictions[p]),
                "ci_lo": float(mc_stats[p]["ci_lo"]),
                "ci_hi": float(mc_stats[p]["ci_hi"]),
                "sigma_total": float(mc_stats[p]["sigma_total"]),
                "s_pca": float(mc_stats[p]["s_pca"]),
                "s_model": float(mc_stats[p]["s_model"]),
            }
            for p in predictions
        },
        "_alpha_xg": alpha_xg,
        "_alpha_sigma": alpha_sigma,
        "_aligned_masked": aligned_masked,
        "_recon_exp_masked": recon_exp_masked,
        "_valid_mask": valid_mask,
        "_recon_exp_1d": recon_exp_1d,
    }


def summary_for_json(result: dict) -> dict:
    """Drop internal ndarray fields before writing JSON."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


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
    add_structure_factor_arguments(parser)
    parser.add_argument("--ridge-lambda", type=float, default=0.01, dest="ridge_lambda",
                        help="Ridge regularization strength used in affine rescaling/PCA projection")
    parser.add_argument("--beta-init", type=float, default=DEFAULT_BETA_INIT, dest="beta_init",
                        help="Initial beta when fitting jointly (default: 3)")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, dest="smooth_sigma",
                        help="Optional mask-normalized Gaussian sigma on aligned 2D intensity grid")
    parser.add_argument("--no-plots", action="store_true", dest="no_plots",
                        help="Skip figure generation (still saves JSON summary)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip Monte Carlo uncertainty propagation (point predictions only)")
    parser.add_argument(
        "--plot-caxis-percentile",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=[2.0, 98.0],
        dest="plot_caxis_percentile",
        help="Percentile clip on log10(I) for 2D intensity color scale (default: 2 98)",
    )
    parser.add_argument(
        "--plot-caxis-min",
        type=float,
        default=None,
        dest="plot_caxis_min",
        help="Fixed color scale minimum in log10(I) (overrides percentile low bound)",
    )
    parser.add_argument(
        "--plot-caxis-max",
        type=float,
        default=None,
        dest="plot_caxis_max",
        help="Fixed color scale maximum in log10(I) (overrides percentile high bound)",
    )
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
    if args.beta is not None and args.beta < 0:
        raise ValueError(f"Invalid --beta {args.beta}: must be non-negative.")
    beta, fit_beta = resolve_beta_settings(args.beta)
    sf_kind = validate_structure_factor_args(
        args.structure_factor, args.prism_length, beta, fit_beta,
    )
    sf_label = structure_factor_label(sf_kind)
    if args.ridge_lambda < 0:
        raise ValueError(f"Invalid --ridge-lambda {args.ridge_lambda}: must be non-negative.")
    if args.smooth_sigma < 0:
        raise ValueError("--smooth-sigma must be non-negative.")
    p_low, p_high = args.plot_caxis_percentile
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(
            f"Invalid --plot-caxis-percentile {args.plot_caxis_percentile}: "
            "require 0 <= LOW < HIGH <= 100."
        )
    if args.plot_caxis_min is not None and args.plot_caxis_max is not None:
        if args.plot_caxis_max <= args.plot_caxis_min:
            raise ValueError(
                f"--plot-caxis-max {args.plot_caxis_max} must be greater than "
                f"--plot-caxis-min {args.plot_caxis_min} (log10 units)."
            )

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

    c = structure_c_for_features(
        sf_kind, args.prism_length, n_features, q_train, q_mag_ref,
        apply_structure=fit_beta or beta is not None,
    )

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

    fixed_bg = 0.0 if args.rescale_background is not None else None
    fixed_scale = float(args.rescale_scale) if args.rescale_scale is not None else None
    if fixed_scale is not None:
        print(f"Fixed affine scale: {fixed_scale:.6g}")
    if fit_beta:
        extra = f", thin-rod L={args.prism_length:g} Å" if sf_kind == "prism" else ""
        print(
            f"Fitting {sf_label} beta jointly (init={float(args.beta_init):g}, "
            f"bounds={BETA_FIT_BOUNDS}){extra}"
        )
    elif beta is not None:
        extra = f", thin-rod L={args.prism_length:g} Å" if sf_kind == "prism" else ""
        print(f"Using fixed {sf_label} beta: {beta:g}{extra}")
    else:
        print("Form factor only (no structure factor)")

    print(f"Fitting {n_components} PCA modes (XGBoost n_components; PCA basis has {U.shape[1]})")
    if args.no_rescale:
        alpha_full, beta, fit_evals, fit_info = project_pattern_masked(
            aligned_for_fit, U, mean, n_components, valid_mask,
            beta=beta, fit_beta=fit_beta, c=c,
            beta_init=float(args.beta_init),
            ridge_lambda=float(args.ridge_lambda),
        )
        rescale_scale, rescale_bg = 1.0, 0.0
    else:
        alpha_full, rescale_scale, rescale_bg, beta, fit_evals, fit_info = project_with_rescaling(
            aligned_for_fit, U, mean, n_components, valid_mask,
            ridge_lambda=float(args.ridge_lambda),
            fixed_background=fixed_bg,
            fixed_scale=fixed_scale,
            beta=beta,
            fit_beta=fit_beta,
            beta_init=float(args.beta_init),
            c=c,
        )
        print(f"Affine rescaling: scale={rescale_scale:.6g}, background={rescale_bg:.6g}")

    if beta is not None:
        beta_label = "Fitted" if fit_beta else "Fixed"
        print(f"{beta_label} {sf_label} beta: {beta:.6g} ({fit_evals} function evaluation(s))")

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
    reconstructed_1d = reconstruct_pattern(alpha_full, U, mean, n_components)
    recon_exp_1d = sim_to_experimental(
        reconstructed_1d, rescale_scale, rescale_bg, args.rescale_background, beta, c,
    )

    aligned_masked = np.where(valid_mask, aligned, np.nan)
    recon_exp_masked = np.where(valid_mask, recon_exp_1d, np.nan)

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------
    mc_stats: dict[str, dict] = {}
    alpha_sigma = np.zeros(n_components, dtype=float)
    if args.fast:
        for param in predictions:
            point = predictions[param]
            mc_stats[param] = {
                "s_pca": 0.0,
                "s_model": 0.0,
                "sigma_total": 0.0,
                "ci_lo": point,
                "ci_hi": point,
            }
    else:
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

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_name = format_plot_filename(pattern_path, model_plot_label).replace(
        "_pca_coefficients.png", "_summary.json"
    )
    summary_path = results_dir / summary_name
    summary_payload = {
        "pattern_file": str(pattern_path),
        "pattern_stem": pattern_path.stem,
        "pca_dir": str(pca_results_dir),
        "models_dir": str(models_dir),
        "rescale_scale": float(rescale_scale),
        "rescale_background": float(args.rescale_background) if args.rescale_background is not None else None,
        "rescale_background_fit": float(rescale_bg),
        "beta": float(beta) if beta is not None else None,
        "beta_fit": bool(fit_beta),
        "structure_factor": sf_kind,
        "prism_length": (
            float(args.prism_length)
            if sf_kind == "prism" and args.prism_length is not None
            else None
        ),
        "q_min": float(beamstop_qmin),
        "q_max": float(args.q_max) if args.q_max is not None else None,
        "n_components": int(n_components),
        "rotation_angle_deg": rotation_angle_deg,
        "rmse_relative_error": rmse_relative_error,
        "pca_coefficients": alpha_xg.reshape(-1).tolist(),
        "predictions": {
            p: {
                "value": float(predictions[p]),
                "ci_lo": float(mc_stats[p]["ci_lo"]),
                "ci_hi": float(mc_stats[p]["ci_hi"]),
                "sigma_total": float(mc_stats[p]["sigma_total"]),
                "s_pca": float(mc_stats[p]["s_pca"]),
                "s_model": float(mc_stats[p]["s_model"]),
            }
            for p in predictions
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved summary to {summary_path}")

    if args.no_plots:
        return

    per_q_plot_data = None
    if q_train is not None and valid_mask.any():
        q_diag = q_train[valid_mask]
        I_exp_diag = aligned_masked[valid_mask]
        recon_exp_diag = recon_exp_masked[valid_mask]
        positive = I_exp_diag > 0
        if positive.sum() > 0:
            per_q_plot_data = {
                "q_pos": q_diag[positive].copy(),
                "I_exp_pos": np.asarray(I_exp_diag[positive], dtype=float),
                "recon_exp_pos": np.asarray(recon_exp_diag[positive], dtype=float),
            }

    plot_grids = build_prediction_plot_grids(
        have_2d_grid=have_2d_grid,
        qx_raw=qx_raw,
        qy_raw=qy_raw,
        p_raw=p_raw,
        qx_ref=qx_ref,
        qy_ref=qy_ref,
        q_mag_ref=q_mag_ref,
        sim_regions=sim_regions if have_2d_grid else [],
        q_train=q_train,
        valid_mask=valid_mask,
        aligned_masked=aligned_masked,
        recon_exp_masked=recon_exp_masked,
        p_low=p_low,
        p_high=p_high,
        caxis_min_log10=args.plot_caxis_min,
        caxis_max_log10=args.plot_caxis_max,
    )

    scatter_path = results_dir / format_scattering_plot_filename(pattern_path, model_plot_label)
    save_scattering_patterns_figure(
        scatter_path,
        pattern_stem=pattern_path.stem,
        grids=plot_grids,
    )
    print(f"Saved scattering patterns to {scatter_path}")

    coeff_path = results_dir / format_plot_filename(pattern_path, model_plot_label)
    save_coefficients_figure(
        coeff_path,
        pattern_stem=pattern_path.stem,
        per_q_data=per_q_plot_data,
        alpha_xg=alpha_xg,
        alpha_sigma=alpha_sigma,
        predictions=predictions,
        mc_stats=mc_stats,
        param_order=param_order,
        uncertainty_lines=uncertainty_lines,
        n_components=n_components,
        no_rescale=args.no_rescale,
        rescale_scale=rescale_scale,
        rescale_scale_arg=args.rescale_scale,
        rescale_background_arg=args.rescale_background,
        rescale_bg=rescale_bg,
        beta=beta,
        fit_beta=fit_beta,
        rotation_angle_deg=rotation_angle_deg,
    )
    print(f"Saved PCA coefficients figure to {coeff_path}")


if __name__ == "__main__":
    main()
