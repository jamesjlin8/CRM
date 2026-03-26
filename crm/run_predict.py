#!/usr/bin/env python3
"""
Predict physical parameters from a single scattering pattern using PCA models.

If the input pattern is on a different q-grid, the pattern is interpolated onto
the training grid when available in the PCA results. Otherwise, the pattern is
required to be on the same grid length (or longer, in which case it is truncated).

Default:
    python run_predict.py --pca-dir pca_results --results-dir predict_results --models-dir 20modes --q-min 0

Usage:
    python run_predict.py /path/to/pattern.dat [--pca-dir DIR] [--results-dir DIR] [--models-dir DIR] [--q-min Q] [--no-rescale]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import pickle
from scipy.interpolate import griddata


def calculate_principal_angle(
    qx: np.ndarray,
    qy: np.ndarray,
    p: np.ndarray,
    q_min: float = 0.005,
    q_max: float = 0.01,
) -> float:
    """Maier-Saupe principal orientation angle from weighted second moment of intensity.

    phi = arctan2(qy, qx); phi_0 = 0.5 * arctan2(sum(p*sin(2*phi)), sum(p*cos(2*phi)))
    Only pixels with q_min <= |q| <= q_max and finite p > 0 are used.
    Returns phi_0 in radians.
    """
    q_mag = np.sqrt(qx**2 + qy**2)
    phi = np.arctan2(qy, qx)
    mask = (q_mag >= q_min) & (q_mag <= q_max) & (p > 0) & np.isfinite(p)
    if not np.any(mask):
        return 0.0
    p_v = p[mask]
    phi_v = phi[mask]
    num = np.sum(p_v * np.sin(2 * phi_v))
    den = np.sum(p_v * np.cos(2 * phi_v))
    phi_0 = 0.5 * np.arctan2(num, den)
    return float(phi_0)


def align_pattern_2d_to_master(
    qx: np.ndarray,
    qy: np.ndarray,
    p: np.ndarray,
    phi_0: float,
    qx_ref: np.ndarray,
    qy_ref: np.ndarray,
    beamstop_qmin: float,
) -> np.ndarray:
    """Map experimental pattern onto the master grid by querying exp at R(phi_0) @ (qx_ref, qy_ref).

    So we sample the raw (qx, qy, p) at positions that align the principal axis with
    the training frame. Returns a 1D array of length len(qx_ref). Beamstop region
    (|q_ref| < beamstop_qmin) is set to 0.
    """
    cos0 = np.cos(phi_0)
    sin0 = np.sin(phi_0)
    # Query positions in experimental frame: (qx_ref, qy_ref) rotated by +phi_0
    qx_query = qx_ref * cos0 - qy_ref * sin0
    qy_query = qx_ref * sin0 + qy_ref * cos0
    aligned = griddata(
        (qx, qy),
        p,
        (qx_query, qy_query),
        method="linear",
        fill_value=0.0,
    )
    aligned = np.asarray(aligned, dtype=float)
    aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
    # Remove beamstop region on the master grid according to the user-chosen radius
    q_mag_ref = np.sqrt(qx_ref**2 + qy_ref**2)
    if beamstop_qmin > 0:
        aligned[q_mag_ref < beamstop_qmin] = 0.0
    return aligned


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
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a .dat pattern file and return q, p sorted by |q|.

    Beamstop pixels are excluded in two stages:
    1. Pixels with p <= 0 or non-finite are always dropped.
    2. Pixels with |q| below the *manual* beamstop radius are dropped so that
       noisy shadow pixels cannot corrupt the PCA projection. When
       q_min_override is None, no beamstop masking is applied (only the p > 0
       and finite filter is used).

    Returns (q_sorted, p_sorted, q_min_valid).
    """
    qx, qy, p = load_pattern_raw(pattern_path)

    # Manual-only beamstop: if q_min_override is provided, use it; otherwise,
    # do not apply any additional |q|-based beamstop cutoff.
    if q_min_override is not None:
        beamstop_r = float(q_min_override)
        print(f"Beamstop radius (manual): {beamstop_r:.6f}")
    else:
        beamstop_r = 0.0
        print("Beamstop radius: 0 (no beamstop masking; using all valid pixels).")

    q_all = np.sqrt(qx**2 + qy**2)
    valid = (p > 0) & np.isfinite(p) & (q_all >= beamstop_r)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"Beamstop filter: excluded {n_dropped} pixels")

    qx_v, qy_v, p_v = qx[valid], qy[valid], p[valid]
    if len(p_v) == 0:
        raise ValueError(f"No valid intensities outside beamstop in {pattern_path}")
    q = np.sqrt(qx_v**2 + qy_v**2)
    q_sorted_idx = np.argsort(q)
    q_min_valid = float(beamstop_r) if beamstop_r > 0 else float(q[q_sorted_idx[0]])
    return (q[q_sorted_idx], p_v[q_sorted_idx], q_min_valid)


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
    if len(q_values) > n_features:
        q_values = q_values[:n_features]
    return q_values


def resample_pattern(
    q_sorted: np.ndarray,
    p_sorted: np.ndarray,
    n_features: int,
    q_train: np.ndarray | None
) -> np.ndarray:
    """Resample a pattern onto the training grid or truncate to PCA feature length.

    When the input grid matches the training grid (same number of points and
    identical sorted |q| values), the intensities are returned as-is so that
    angular (anisotropic) structure is preserved.  A 1-D ``np.interp`` fallback
    is only used when the grids genuinely differ.
    """
    if len(p_sorted) == 0:
        raise ValueError("Pattern has no points to resample.")
    if q_train is not None:
        if len(q_train) != n_features:
            raise ValueError(
                f"Training q-grid has {len(q_train)} points, but PCA expects {n_features}."
            )
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
    if len(p_sorted) > n_features:
        return p_sorted[:n_features]
    return p_sorted


def project_pattern_masked(
    aligned: np.ndarray,
    U: np.ndarray,
    mean: np.ndarray,
    n_components: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Project a pattern with missing data (e.g. beamstop) via least squares.

    Only the pixels indicated by *valid_mask* participate. Uses the normal
    equations alpha = inv(U_v.T @ U_v) @ U_v.T @ centered_v.
    """
    centered = aligned - mean
    U_v = U[valid_mask, :n_components]
    centered_v = centered[valid_mask]
    M = U_v.T @ U_v
    rhs = U_v.T @ centered_v
    alpha = np.linalg.solve(M, rhs)
    return alpha.reshape(1, -1)


def project_with_rescaling(
    aligned: np.ndarray,
    U: np.ndarray,
    mean: np.ndarray,
    n_components: int,
    valid_mask: np.ndarray | None = None,
    ridge_lambda: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Jointly solve for affine rescaling and PCA coefficients with Ridge on alpha.

    Minimizes ||scale * p + background - mean - U @ alpha||^2 + lam*||alpha||^2
    over pixels in *valid_mask*. Uses (A.T @ A + R) @ x = A.T @ mean_v with R
    zero except R[2:,2:] = lam*I so only alpha is regularized.
    Returns (alpha_row_vector, scale, background).
    """
    k_full = U.shape[1]
    if valid_mask is None:
        valid_mask = np.ones(len(aligned), dtype=bool)

    p_v = aligned[valid_mask]
    mean_v = mean[valid_mask]
    U_v = U[valid_mask, :k_full]

    n_valid = int(valid_mask.sum())
    A = np.empty((n_valid, k_full + 2))
    A[:, 0] = p_v
    A[:, 1] = 1.0
    A[:, 2:] = -U_v

    M = A.T @ A
    M[2:, 2:] += ridge_lambda * np.eye(k_full)
    rhs = A.T @ mean_v
    x = np.linalg.solve(M, rhs)
    scale = float(x[0])
    background = float(x[1])
    alpha = x[2:].reshape(1, -1)
    return alpha, scale, background


def reconstruct_pattern(
    alpha: np.ndarray, U: np.ndarray, mean: np.ndarray, n_components: int
) -> np.ndarray:
    """Reconstruct a 1D pattern from PCA coefficients: mean + alpha @ U^T."""
    return mean + (alpha.reshape(1, -1) @ U[:, :n_components].T).reshape(-1)


def grid_pattern_2d(
    qx: np.ndarray, qy: np.ndarray, values: np.ndarray,
    max_grid_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid scattered (qx, qy, values) data onto a 2D array for imshow.

    For data already on a regular grid (e.g. experimental raster scans),
    uses fast direct indexing.  For scattered data (e.g. simulation output
    sorted by |q|), bins onto a regular grid of at most *max_grid_size*
    pixels per side.

    Returns (grid, qx_axis, qy_axis).
    """
    unique_qx = np.unique(qx)
    unique_qy = np.unique(qy)

    # Heuristic: if the outer-product grid is compact, data is on a regular grid
    if len(unique_qx) * len(unique_qy) <= 4 * len(qx):
        unique_qx.sort()
        unique_qy.sort()
        grid = np.full((len(unique_qy), len(unique_qx)), np.nan)
        x_idx = np.searchsorted(unique_qx, qx)
        y_idx = np.searchsorted(unique_qy, qy)
        grid[y_idx, x_idx] = values
        return grid, unique_qx, unique_qy

    # Scattered data — bin onto a regular grid
    n_bins = min(max_grid_size, max(int(np.sqrt(len(qx))), 64))
    qx_edges = np.linspace(qx.min(), qx.max(), n_bins + 1)
    qy_edges = np.linspace(qy.min(), qy.max(), n_bins + 1)

    sum_grid, _, _ = np.histogram2d(qx, qy, bins=[qx_edges, qy_edges], weights=values)
    count_grid, _, _ = np.histogram2d(qx, qy, bins=[qx_edges, qy_edges])

    with np.errstate(invalid="ignore"):
        grid = np.where(count_grid > 0, sum_grid / count_grid, np.nan)

    # histogram2d returns shape (n_qx, n_qy); transpose so rows = qy
    grid = grid.T

    qx_centers = 0.5 * (qx_edges[:-1] + qx_edges[1:])
    qy_centers = 0.5 * (qy_edges[:-1] + qy_edges[1:])
    return grid, qx_centers, qy_centers


def detect_detector_regions(
    qx_ref: np.ndarray, qy_ref: np.ndarray,
) -> list[tuple[int, int]]:
    """Detect contiguous detector regions in a concatenated multi-detector grid.

    Tries splitting into N equal-size chunks (N = 1..5) and verifies each chunk
    is a perfect-square number of pixels whose coordinates form an approximate
    rectangular grid (one axis is nearly constant within the first sqrt(N)
    elements).  Falls back to a single region spanning the full array.
    """
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
            qx_first = qx_ref[s : s + side]
            qy_first = qy_ref[s : s + side]
            qx_span = qx_first.max() - qx_first.min()
            qy_span = qy_first.max() - qy_first.min()
            full_qx = qx_ref[s : s + chunk].max() - qx_ref[s : s + chunk].min()
            full_qy = qy_ref[s : s + chunk].max() - qy_ref[s : s + chunk].min()
            slow_ok = (
                qx_span / max(full_qx, 1e-12) < 0.02
                or qy_span / max(full_qy, 1e-12) < 0.02
            )
            if not slow_ok:
                all_regular = False
                break
        if all_regular:
            return [(r * chunk, (r + 1) * chunk) for r in range(n_regions)]
    return [(0, n)]


def grid_region_2d(
    qx: np.ndarray, qy: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid a single detector region into a 2D array (no binning).

    Determines the fast/slow axes from the coordinate layout and reshapes
    directly.  Returns (grid, qx_axis, qy_axis) with grid shaped (ny, nx)
    for use with ``imshow(origin='lower')``.
    """
    n = len(values)
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError(
            f"Region has {n} pixels which is not a perfect square; "
            "cannot reshape into a 2D grid."
        )

    qx_var = qx[:side].max() - qx[:side].min()
    qy_var = qy[:side].max() - qy[:side].min()

    if qy_var > qx_var:
        # qy is fast axis (varies within each row of reshaped array);
        # qx is slow axis (approximately constant per row).
        grid_raw = values.reshape(side, side)  # (n_qx_groups, n_qy_per_group)
        grid_2d = grid_raw.T                   # (ny, nx) for imshow
        qx_axis = np.array([qx[i * side : (i + 1) * side].mean() for i in range(side)])
        qy_axis = qy[:side].copy()
    else:
        # qx is fast axis; qy is slow axis.
        grid_2d = values.reshape(side, side)   # already (ny, nx)
        qx_axis = qx[:side].copy()
        qy_axis = np.array([qy[i * side : (i + 1) * side].mean() for i in range(side)])

    if qx_axis[-1] < qx_axis[0]:
        qx_axis = qx_axis[::-1]
        grid_2d = grid_2d[:, ::-1]
    if qy_axis[-1] < qy_axis[0]:
        qy_axis = qy_axis[::-1]
        grid_2d = grid_2d[::-1, :]

    return grid_2d, qx_axis, qy_axis


def format_plot_filename(pattern_path: Path, model_label: str) -> str:
    """``{model_label}_{input_stem}_pca_coefficients.png``."""
    return f"{model_label}_{pattern_path.stem}_pca_coefficients.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict physical parameters from a single .dat pattern using PCA models."
    )
    parser.add_argument(
        "pattern_file",
        type=str,
        help="Path to a .dat file with columns: qx qy p"
    )
    parser.add_argument(
        "--pca-dir",
        type=str,
        default="pca_results",
        dest="pca_dir",
        help="Directory containing PCA artifacts (default: pca_results)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="predict_results",
        dest="results_dir",
        help="Directory to save plots and outputs (default: predict_results)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="20modes",
        help="Directory containing trained models (default: 20modes)"
    )
    parser.add_argument(
        "--no-rescale",
        action="store_true",
        help="Disable affine rescaling (use when input is already on the simulation scale)"
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=None,
        help="Manual beamstop radius override (exclude |q| < this value)"
    )
    args = parser.parse_args()

    # Resolve beamstop radius for this run. If the user supplies --q-min, that
    # value is used everywhere; otherwise, no |q|-based beamstop masking is applied.
    if args.q_min is not None and args.q_min < 0:
        raise ValueError(f"Invalid --q-min {args.q_min}: must be non-negative.")
    beamstop_qmin = float(args.q_min) if args.q_min is not None else 0.0
    if beamstop_qmin > 0:
        print(f"Beamstop radius (manual): |q| < {beamstop_qmin:.6f} will be treated as beamstop.")
    else:
        print("Beamstop radius: 0 (no |q|-based beamstop masking).")

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
        legacy_file = pca_results_dir / "parameter_models.pkl"
        if legacy_file.exists():
            print(
                f"Warning: Models not found in {models_dir}; falling back to {legacy_file}."
            )
            models_file = legacy_file
        else:
            raise FileNotFoundError(
                f"Parameter models not found: {models_file}. "
                "Run run_pca_model.py to generate them."
            )

    with open(pca_file, "rb") as f:
        U, _S, pca_model = pickle.load(f)
    mean = pca_model["mean_"]
    n_features = U.shape[0]
    q_train = load_training_q_values(pca_results_dir, pca_model, n_features)

    # Load 2D spatial metadata (master grid the PCA was trained on).
    qx_ref_path = pca_results_dir / "qx_ref.npy"
    qy_ref_path = pca_results_dir / "qy_ref.npy"
    have_2d_grid = qx_ref_path.exists() and qy_ref_path.exists()
    sim_regions = None
    if have_2d_grid:
        qx_ref = np.load(qx_ref_path).reshape(-1)[:n_features]
        qy_ref = np.load(qy_ref_path).reshape(-1)[:n_features]
        q_mag_ref = np.sqrt(qx_ref**2 + qy_ref**2)
        sim_regions = detect_detector_regions(qx_ref, qy_ref)
        print(f"Simulation grid: {n_features} pixels in {len(sim_regions)} detector region(s)")
        for i, (s, e) in enumerate(sim_regions):
            q_lo = q_mag_ref[s:e].min()
            q_hi = q_mag_ref[s:e].max()
            side = int(round(np.sqrt(e - s)))
            print(f"  Region {i}: {side}x{side} = {e - s} pixels, "
                  f"|q| in [{q_lo:.5f}, {q_hi:.5f}]")
    else:
        qx_ref = qy_ref = q_mag_ref = None

    with open(models_file, "rb") as f:
        model_bundle = pickle.load(f)

    n_components = model_bundle.get("n_components", U.shape[1])
    n_components = min(n_components, U.shape[1])
    models = model_bundle.get("models", {})
    if not models:
        raise ValueError("No parameter models found in parameter_models.pkl")

    # ------------------------------------------------------------------
    # Load raw pattern and align to master 2D grid (when available).
    # ------------------------------------------------------------------
    qx_raw, qy_raw, p_raw = load_pattern_raw(pattern_path)

    rotation_angle_deg = None
    if have_2d_grid:
        phi_0 = calculate_principal_angle(qx_raw, qy_raw, p_raw)
        # Rotate so principal axis is vertical: use rotation = 90° - phi_0
        rotation_angle = np.pi / 2 - phi_0
        rotation_angle_deg = float(np.degrees(rotation_angle))
        print(
            f"Maier-Saupe phi_0 = {np.degrees(phi_0):.3f} deg, "
            f"rotation (90° - phi_0) = {rotation_angle_deg:.3f} deg"
        )
        aligned = align_pattern_2d_to_master(
            qx_raw,
            qy_raw,
            p_raw,
            rotation_angle,
            qx_ref,
            qy_ref,
            beamstop_qmin,
        )
        q_exp_min = float(np.sqrt(qx_raw**2 + qy_raw**2).min())
        q_exp_max = float(np.sqrt(qx_raw**2 + qy_raw**2).max())
        valid_mask = (
            np.isfinite(aligned)
            & (q_mag_ref >= beamstop_qmin)
            & (q_mag_ref >= q_exp_min)
            & (q_mag_ref <= q_exp_max)
        )
        if not np.any(valid_mask):
            valid_mask = np.isfinite(aligned) & (q_mag_ref >= beamstop_qmin)
    else:
        (q_sorted, p_sorted, q_min_valid) = load_pattern_sorted(
            pattern_path,
            q_min_override=beamstop_qmin if beamstop_qmin > 0 else None,
        )
        if q_train is None:
            print("Warning: training q-grid not found; falling back to truncate/strict behavior.")
        aligned = resample_pattern(q_sorted, p_sorted, n_features, q_train)
        if q_train is not None:
            q_exp_min = float(q_sorted[0])
            q_exp_max = float(q_sorted[-1])
            valid_mask = (
                (q_train >= q_min_valid)
                & (q_train >= q_exp_min)
                & (q_train <= q_exp_max)
            )
            if not np.any(valid_mask):
                valid_mask = q_train >= q_min_valid
        else:
            valid_mask = np.ones(n_features, dtype=bool)

    n_masked = int((~valid_mask).sum())
    if n_masked > 0:
        print(f"Masking {n_masked} training-grid points "
              f"(beamstop / outside experimental coverage)")

    if args.no_rescale:
        alpha_full = project_pattern_masked(aligned, U, mean, U.shape[1], valid_mask)
        rescale_scale, rescale_bg = 1.0, 0.0
    else:
        alpha_full, rescale_scale, rescale_bg = project_with_rescaling(
            aligned, U, mean, n_components, valid_mask
        )
        print(f"Affine rescaling: scale={rescale_scale:.6g}, background={rescale_bg:.6g}")
        if rescale_scale <= 0:
            print("Warning: estimated scale <= 0; predictions may be unreliable.")
    alpha_xg = alpha_full[:, :n_components]

    predictions = {}
    model_metadata = {}
    param_order = ["radius", "length", "n_cyl", "stretch"]
    for param in param_order:
        if param not in models:
            continue
        model_entry = models[param]
        if "model" not in model_entry:
            raise ValueError(f"Model entry for '{param}' is missing the 'model' field.")
        model = model_entry["model"]
        predictions[param] = float(model.predict(alpha_xg)[0])
        model_metadata[param] = {
            "model_type": model_entry.get("model_type"),
            "r2_score": model_entry.get("r2_score")
        }

    print("PCA coefficients (first n_components used for XGBoost):")
    print(alpha_xg.reshape(-1))
    print("Predicted parameters:")
    print(f"  n_components: {n_components}")
    for param in param_order:
        if param in predictions:
            print(f"  {param}: {predictions[param]:.6g}")
            meta = model_metadata.get(param, {})
            model_type = meta.get("model_type")
            r2_score = meta.get("r2_score")
            if model_type is not None or r2_score is not None:
                parts = []
                if model_type is not None:
                    parts.append(f"model={model_type}")
                if r2_score is not None:
                    parts.append(f"r2={r2_score:.3f}")
                print(f"    ({', '.join(parts)})")

    # ------------------------------------------------------------------
    # Reconstruct the scattering pattern from PCA coefficients
    # ------------------------------------------------------------------
    reconstructed_1d = reconstruct_pattern(alpha_full, U, mean, U.shape[1])

    # Map reconstruction back to experimental intensity scale so that
    # the 2D comparison plot uses the same units as the input pattern.
    if rescale_scale != 0:
        recon_exp_1d = (reconstructed_1d - rescale_bg) / rescale_scale
    else:
        recon_exp_1d = reconstructed_1d

    # ------------------------------------------------------------------
    # Per-q comparison diagnostic: ratio = recon_exp / I_exp
    # If the model were perfect per-pixel (with or without affine rescaling),
    # ratio would be 1 everywhere. Large variation with q suggests the fit is
    # dominated by one q-region (e.g. high-q) and a weighted fit may help.
    # This diagnostic is produced for both rescaled and non-rescaled fits.
    # ------------------------------------------------------------------
    if q_train is not None and valid_mask.any():
        q_diag = q_train[valid_mask]
        I_exp_diag = aligned[valid_mask]
        recon_exp_diag = recon_exp_1d[valid_mask]
        positive = I_exp_diag > 0
        if positive.sum() > 0:
            q_pos = q_diag[positive]
            ratio = recon_exp_diag[positive] / I_exp_diag[positive]

            # Summary by q terciles (low / mid / high q)
            n = len(q_pos)
            idx_sorted = np.argsort(q_pos)
            q_sorted_diag = q_pos[idx_sorted]
            ratio_sorted = ratio[idx_sorted]
            third = max(1, n // 3)
            low_slice = ratio_sorted[:third]
            mid_slice = ratio_sorted[third : 2 * third]
            high_slice = ratio_sorted[2 * third :]
            q_low = q_sorted_diag[:third].mean()
            q_mid = q_sorted_diag[third : 2 * third].mean()
            q_high = q_sorted_diag[2 * third :].mean()
            mean_low = np.nanmean(low_slice)
            mean_mid = np.nanmean(mid_slice)
            mean_high = np.nanmean(high_slice)
            std_low = np.nanstd(low_slice)
            std_mid = np.nanstd(mid_slice)
            std_high = np.nanstd(high_slice)

            print("Per-q fit ratio (recon_exp / I_exp); ideal = 1.0:")
            print(f"  low  q (q≈{q_low:.4f}): mean={mean_low:.4f}, std={std_low:.4f}")
            print(f"  mid  q (q≈{q_mid:.4f}): mean={mean_mid:.4f}, std={std_mid:.4f}")
            print(f"  high q (q≈{q_high:.4f}): mean={mean_high:.4f}, std={std_high:.4f}")
            ratio_span = max(mean_low, mean_mid, mean_high) - min(mean_low, mean_mid, mean_high)
            if ratio_span > 0.5 or np.nanstd(ratio) > 0.5:
                print("  -> Large variation with q: fit may be dominated by high-q or background-dominated regions.")
            else:
                print("  -> Ratio fairly uniform with q; unweighted (or current) fit is consistent across range.")

            # Diagnostic plot: intensity vs q and relative error vs q
            plots_dir_diag = Path(args.results_dir)
            plots_dir_diag.mkdir(parents=True, exist_ok=True)
            fig_diag, (ax_int, ax_err) = plt.subplots(
                2, 1, figsize=(8, 8), sharex=True
            )
            ax_int.loglog(q_pos, I_exp_diag[positive], ".", alpha=0.3, ms=1, label="Experimental", color="C0")
            ax_int.loglog(q_pos, recon_exp_diag[positive], ".", alpha=0.3, ms=1, label="Reconstruction (exp scale)", color="C1")
            ax_int.set_ylabel("Intensity")
            ax_int.legend(loc="best", fontsize=8)
            ax_int.set_title("Per-q comparison (training grid, valid points)")

            rel_error = (recon_exp_diag[positive] - I_exp_diag[positive]) / I_exp_diag[positive]
            finite_err = rel_error[np.isfinite(rel_error)]
            rmse = float(np.sqrt(np.mean(finite_err**2))) if len(finite_err) > 0 else float("nan")
            mean_bias = float(np.mean(finite_err)) if len(finite_err) > 0 else float("nan")

            ax_err.scatter(q_pos, rel_error, s=1, alpha=0.4, color="C3")
            ax_err.axhline(0.0, color="k", ls="--", lw=1, label="ideal = 0")
            ax_err.set_xlabel(r"|q| (Å$^{-1}$)")
            ax_err.set_ylabel("Relative Error\n(recon − I_exp) / I_exp")
            err_abs_99 = np.nanpercentile(np.abs(np.where(np.isfinite(rel_error), rel_error, np.nan)), 99)
            y_lim = min(2.0, max(0.5, err_abs_99 * 1.1)) if np.isfinite(err_abs_99) else 1.0
            ax_err.set_ylim(-y_lim, y_lim)
            ax_err.legend(loc="upper left", fontsize=8)
            ax_err.text(
                0.98, 0.95,
                f"RMSE = {rmse:.4f}\nMean bias = {mean_bias:+.4f}",
                transform=ax_err.transAxes,
                ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

            fig_diag.suptitle(f"{pattern_path.stem}: affine rescaling per-q diagnostic")
            fig_diag.tight_layout()
            diag_name = format_plot_filename(
                pattern_path, model_plot_label
            ).replace(".png", "_per_q_diagnostic.png")
            fig_diag.savefig(plots_dir_diag / diag_name, dpi=150, bbox_inches="tight")
            plt.close(fig_diag)
            print(f"Saved per-q diagnostic to {plots_dir_diag / diag_name}")

    # Load raw 2D pattern for visualization
    qx_raw, qy_raw, p_raw = load_pattern_raw(pattern_path)
    q_mag_raw = np.sqrt(qx_raw**2 + qy_raw**2)

    # Restrict to the q-range covered by the PCA training grid (no extrapolation)
    if q_train is not None:
        q_lo, q_hi = q_train.min(), q_train.max()
    else:
        q_lo, q_hi = q_mag_raw.min(), q_mag_raw.max()

    in_range = (q_mag_raw >= q_lo) & (q_mag_raw <= q_hi)
    qx_clip = qx_raw[in_range]
    qy_clip = qy_raw[in_range]
    p_clip = p_raw[in_range]

    # Grid original pattern (qx = horizontal, qy = vertical)
    original_grid, gqx, gqy = grid_pattern_2d(qx_clip, qy_clip, p_clip)

    # Build per-region 2D layers for the simulation grid panels.
    # Each detector region is gridded independently at its native resolution
    # (no binning / no mixing of pixel sizes across detector regions).
    # Layers are sorted coarsest-first so finer regions paint on top.
    interp_layers = None
    recon_layers = None
    recon_grid = None
    recon_extent = None

    if have_2d_grid:
        region_order = sorted(
            sim_regions,
            key=lambda r: -(qx_ref[r[0]:r[1]].max() - qx_ref[r[0]:r[1]].min()),
        )
        interp_layers = []
        recon_layers = []
        for s, e in region_order:
            qxr, qyr = qx_ref[s:e], qy_ref[s:e]
            g_int, ux, uy = grid_region_2d(qxr, qyr, aligned[s:e])
            g_int = np.where(g_int > 0, g_int, np.nan)
            ext = [ux[0], ux[-1], uy[0], uy[-1]]
            interp_layers.append((g_int, ext))

            g_rec, ux, uy = grid_region_2d(qxr, qyr, recon_exp_1d[s:e])
            g_rec = np.where(g_rec > 0, g_rec, np.nan)
            recon_layers.append((g_rec, ext))
    else:
        n_recon = max(len(gqx), len(gqy), 256)
        recon_qx = np.linspace(gqx[0], gqx[-1], n_recon)
        recon_qy = np.linspace(gqy[0], gqy[-1], n_recon)
        rqx_2d, rqy_2d = np.meshgrid(recon_qx, recon_qy)
        rq_mag = np.sqrt(rqx_2d**2 + rqy_2d**2)

        if q_train is not None:
            recon_flat = np.interp(rq_mag.ravel(), q_train, recon_exp_1d)
        else:
            q_sorted_input = np.sort(q_mag_raw)
            if len(q_sorted_input) > len(recon_exp_1d):
                q_sorted_input = q_sorted_input[:len(recon_exp_1d)]
            recon_flat = np.interp(rq_mag.ravel(), q_sorted_input, recon_exp_1d)
        recon_grid = recon_flat.reshape(rq_mag.shape)
        recon_grid = np.where(
            (rq_mag >= q_lo) & (rq_mag <= q_hi), recon_grid, np.nan
        )
        recon_grid = np.where(recon_grid > 0, recon_grid, np.nan)
        recon_extent = [recon_qx[0], recon_qx[-1], recon_qy[0], recon_qy[-1]]

    original_grid = np.where(original_grid > 0, original_grid, np.nan)

    # Shared color limits across all panels
    valid_orig = original_grid[np.isfinite(original_grid)]
    color_pools = [valid_orig]
    if have_2d_grid:
        pos_aligned = aligned[(aligned > 0) & np.isfinite(aligned)]
        pos_recon = recon_exp_1d[(recon_exp_1d > 0) & np.isfinite(recon_exp_1d)]
        color_pools.extend([pos_aligned, pos_recon])
    else:
        color_pools.append(recon_grid[np.isfinite(recon_grid)])
    all_valid = np.concatenate([v for v in color_pools if v.size > 0])
    if all_valid.size > 0:
        shared_vmin = max(all_valid.min(), 1e-10)
        shared_vmax = all_valid.max()
    else:
        shared_vmin, shared_vmax = 1e-6, 1.0

    orig_extent = [gqx[0], gqx[-1], gqy[0], gqy[-1]]

    # ------------------------------------------------------------------
    # Combined figure: 2D patterns (top) + PCA coefficients (bottom)
    # ------------------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_cols_top = 3 if have_2d_grid else 2
    fig = plt.figure(figsize=(7 * n_cols_top, 10))
    gs = GridSpec(
        2, n_cols_top,
        height_ratios=[1.2, 1],
        hspace=0.30,
        wspace=0.35,
    )

    # --- Panel 1: Original raw pattern (unrotated) ---
    ax_orig = fig.add_subplot(gs[0, 0])
    im_orig = ax_orig.imshow(
        original_grid,
        extent=orig_extent,
        origin="lower",
        aspect="equal",
        norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax),
        cmap="turbo",
    )
    ax_orig.set_xlabel(r"Q$_x$ (Å$^{-1}$)")
    ax_orig.set_ylabel(r"Q$_y$ (Å$^{-1}$)")
    ax_orig.set_title("Original Raw (unrotated)")
    fig.colorbar(im_orig, ax=ax_orig, label="Intensity", shrink=0.85)

    # --- Panel 2: Rotated, interpolated pattern on simulation grid ---
    if interp_layers is not None:
        ax_interp = fig.add_subplot(gs[0, 1])
        shared_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)
        for g, ext in interp_layers:
            im_interp = ax_interp.imshow(
                g, extent=ext, origin="lower", aspect="equal",
                norm=shared_norm, cmap="turbo",
            )
        ax_interp.set_xlabel(r"Q$_x$ (Å$^{-1}$)")
        ax_interp.set_ylabel(r"Q$_y$ (Å$^{-1}$)")
        ax_interp.set_title("Aligned (rotated + on sim. grid)")
        ax_interp.set_xlim(orig_extent[0], orig_extent[1])
        ax_interp.set_ylim(orig_extent[2], orig_extent[3])
        fig.colorbar(im_interp, ax=ax_interp, label="Intensity", shrink=0.85)

    # --- Panel 3: PCA reconstruction ---
    ax_recon = fig.add_subplot(gs[0, n_cols_top - 1])
    recon_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)
    if recon_layers is not None:
        for g, ext in recon_layers:
            im_recon = ax_recon.imshow(
                g, extent=ext, origin="lower", aspect="equal",
                norm=recon_norm, cmap="turbo",
            )
    else:
        im_recon = ax_recon.imshow(
            recon_grid, extent=recon_extent, origin="lower", aspect="equal",
            norm=recon_norm, cmap="turbo",
        )
    ax_recon.set_xlabel(r"Q$_x$ (Å$^{-1}$)")
    ax_recon.set_ylabel(r"Q$_y$ (Å$^{-1}$)")
    ax_recon.set_title(f"PCA Reconstruction ({U.shape[1]} modes)")
    ax_recon.set_xlim(orig_extent[0], orig_extent[1])
    ax_recon.set_ylim(orig_extent[2], orig_extent[3])
    fig.colorbar(im_recon, ax=ax_recon, label="Intensity", shrink=0.85)

    # --- Bottom: PCA coefficients bar chart ---
    ax_coeff = fig.add_subplot(gs[1, :])
    coeffs = alpha_full.reshape(-1)[:20]
    x = np.arange(1, len(coeffs) + 1)
    ax_coeff.bar(x, coeffs, color="tab:blue")
    ax_coeff.set_xlabel("PCA mode")
    ax_coeff.set_ylabel("Coefficient")
    ax_coeff.set_title("PCA Coefficients")
    ax_coeff.set_xticks(x)

    param_lines = [
        f"radius={predictions.get('radius', float('nan')):.3g}",
        f"length={predictions.get('length', float('nan')):.3g}",
        f"n_cyl={predictions.get('n_cyl', float('nan')):.3g}",
        f"stretch={predictions.get('stretch', float('nan')):.3g}",
    ]
    if not args.no_rescale:
        param_lines.append(f"scale={rescale_scale:.3g}")
        param_lines.append(f"bg={rescale_bg:.3g}")
    if rotation_angle_deg is not None:
        param_lines.append(f"rot={rotation_angle_deg:.1f}°")
    param_text = "\n".join(param_lines)
    ax_coeff.text(
        0.98, 0.98, param_text,
        transform=ax_coeff.transAxes,
        ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle(pattern_path.stem, fontsize=12, fontweight="bold")

    coeff_plot_name = format_plot_filename(pattern_path, model_plot_label)
    fig.savefig(results_dir / coeff_plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {results_dir / coeff_plot_name}")


if __name__ == "__main__":
    main()
