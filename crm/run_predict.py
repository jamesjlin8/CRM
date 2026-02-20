#!/usr/bin/env python3
"""
Predict physical parameters from a single scattering pattern using PCA models.
The pattern is oriented by the dominant scattering direction before projection.

If the input pattern is on a different q-grid, the pattern is interpolated onto
the training grid when available in the PCA results. Otherwise, the pattern is
required to be on the same grid length (or longer, in which case it is truncated).

Usage:
    python run_predict.py /path/to/pattern.dat
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import pickle


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


def estimate_orientation_angle(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray) -> float:
    """Estimate preferred orientation angle using second angular moment."""
    mask = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(intensity)
    if not np.any(mask):
        return 0.0
    qx = qx[mask]
    qy = qy[mask]
    intensity = intensity[mask]
    weights = np.clip(intensity, 0, None)
    if np.all(weights == 0):
        return 0.0
    theta = np.arctan2(qy, qx)
    c = np.sum(weights * np.cos(2.0 * theta))
    s = np.sum(weights * np.sin(2.0 * theta))
    if c == 0 and s == 0:
        return 0.0
    return 0.5 * np.arctan2(s, c)


def rotate_pattern(qx: np.ndarray, qy: np.ndarray, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate qx, qy coordinates by angle_rad."""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    qx_rot = qx * cos_a - qy * sin_a
    qy_rot = qx * sin_a + qy * cos_a
    return qx_rot, qy_rot


def load_pattern_sorted(pattern_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a .dat pattern file and return q, p sorted by |q| after orientation."""
    qx, qy, p = load_pattern_raw(pattern_path)
    orientation_angle = estimate_orientation_angle(qx, qy, p)
    qx_rot, qy_rot = rotate_pattern(qx, qy, -orientation_angle)
    q = np.sqrt(qx_rot**2 + qy_rot**2)
    q_sorted_idx = np.argsort(q)
    return q[q_sorted_idx], p[q_sorted_idx], orientation_angle


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
    """Resample a pattern onto the training grid or truncate to PCA feature length."""
    if len(p_sorted) == 0:
        raise ValueError("Pattern has no points to resample.")
    if q_train is not None:
        if len(q_train) != n_features:
            raise ValueError(
                f"Training q-grid has {len(q_train)} points, but PCA expects {n_features}."
            )
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


def project_pattern(
    q_sorted: np.ndarray,
    p_sorted: np.ndarray,
    U: np.ndarray,
    mean: np.ndarray,
    n_components: int,
    q_train: np.ndarray | None
) -> np.ndarray:
    """Project a single pattern into PCA space and return a row vector."""
    n_features = U.shape[0]
    aligned = resample_pattern(q_sorted, p_sorted, n_features, q_train)
    centered = aligned - mean
    return (centered @ U[:, :n_components]).reshape(1, -1)


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


def format_prediction_filename(pattern_path: Path, predictions: dict) -> str:
    """Build a filename similar to input structure, using predicted parameters."""
    def fmt_int(val: float | None, fallback: str) -> str:
        if val is None:
            return fallback
        return f"{int(round(val))}"

    def fmt_float(val: float | None, fallback: str) -> str:
        if val is None:
            return fallback
        return f"{val:.2f}".rstrip("0").rstrip(".") or "0"

    stretch = fmt_float(predictions.get("stretch"), "na")
    n_cyl = fmt_int(predictions.get("n_cyl"), "na")
    radius = fmt_int(predictions.get("radius"), "na")
    length = fmt_int(predictions.get("length"), "na")
    return f"{pattern_path.stem}_Pred_St{stretch}_{n_cyl}cyl_{radius}r_{length}l_pca_coefficients.png"


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
        "--pca-results-dir",
        type=str,
        default="pca_results",
        help="Directory containing PCA artifacts (default: pca_results)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="10modes",
        help="Directory containing trained models (default: 10modes)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write predictions as JSON (includes model metadata)"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="predict_results",
        help="Directory to save plots (default: predict_results)"
    )
    args = parser.parse_args()

    pattern_path = Path(args.pattern_file)
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

    pca_results_dir = Path(args.pca_results_dir)
    pca_file = pca_results_dir / "pca_components.pkl"
    models_dir = Path(args.models_dir) if args.models_dir else pca_results_dir / "models"
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
    q_train = load_training_q_values(pca_results_dir, pca_model, U.shape[0])

    with open(models_file, "rb") as f:
        model_bundle = pickle.load(f)

    n_components = model_bundle.get("n_components", U.shape[1])
    n_components = min(n_components, U.shape[1])
    models = model_bundle.get("models", {})
    if not models:
        raise ValueError("No parameter models found in parameter_models.pkl")

    q_sorted, p_sorted, orientation_angle = load_pattern_sorted(pattern_path)
    if q_train is None:
        print("Warning: training q-grid not found; falling back to truncate/strict behavior.")
    aligned = resample_pattern(q_sorted, p_sorted, U.shape[0], q_train)
    alpha = project_pattern(q_sorted, p_sorted, U, mean, n_components, q_train)

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
        predictions[param] = float(model.predict(alpha)[0])
        model_metadata[param] = {
            "model_type": model_entry.get("model_type"),
            "r2_score": model_entry.get("r2_score")
        }

    print(f"Estimated orientation angle: {np.degrees(orientation_angle):.2f} degrees")
    print("PCA coefficients:")
    print(alpha.reshape(-1))
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
    reconstructed_1d = reconstruct_pattern(alpha, U, mean, n_components)

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
    q_mag_clip = q_mag_raw[in_range]

    # Grid original pattern at its actual data points (preserves beamstop, etc.)
    original_grid, gqx, gqy = grid_pattern_2d(qx_clip, qy_clip, p_clip)

    # Build a complete regular grid for the reconstruction so the PCA model
    # fills the entire q-range (including beamstop / missing-data regions).
    n_recon = max(len(gqx), len(gqy), 256)
    recon_qx = np.linspace(gqx[0], gqx[-1], n_recon)
    recon_qy = np.linspace(gqy[0], gqy[-1], n_recon)
    rqx_2d, rqy_2d = np.meshgrid(recon_qx, recon_qy)
    rq_mag = np.sqrt(rqx_2d**2 + rqy_2d**2)

    # Evaluate the 1D PCA reconstruction at every grid pixel's |q|
    if q_train is not None:
        recon_flat = np.interp(rq_mag.ravel(), q_train, reconstructed_1d)
    else:
        q_sorted_input = np.sort(q_mag_raw)
        if len(q_sorted_input) > len(reconstructed_1d):
            q_sorted_input = q_sorted_input[:len(reconstructed_1d)]
        recon_flat = np.interp(rq_mag.ravel(), q_sorted_input, reconstructed_1d)
    recon_grid = recon_flat.reshape(rq_mag.shape)

    # Mask pixels outside the training q-range
    recon_grid = np.where(
        (rq_mag >= q_lo) & (rq_mag <= q_hi), recon_grid, np.nan
    )

    # Replace non-positive values with NaN for log-scale display
    original_grid = np.where(original_grid > 0, original_grid, np.nan)
    recon_grid = np.where(recon_grid > 0, recon_grid, np.nan)

    # Shared color limits across both 2D panels
    valid_orig = original_grid[np.isfinite(original_grid)]
    valid_recon = recon_grid[np.isfinite(recon_grid)]
    all_valid = np.concatenate(
        [v for v in (valid_orig, valid_recon) if v.size > 0]
    )
    if all_valid.size > 0:
        shared_vmin = max(all_valid.min(), 1e-10)
        shared_vmax = all_valid.max()
    else:
        shared_vmin, shared_vmax = 1e-6, 1.0

    orig_extent = [gqx[0], gqx[-1], gqy[0], gqy[-1]]
    recon_extent = [recon_qx[0], recon_qx[-1], recon_qy[0], recon_qy[-1]]

    # ------------------------------------------------------------------
    # Combined figure: 2D patterns (top) + PCA coefficients (bottom)
    # ------------------------------------------------------------------
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(
        2, 2,
        height_ratios=[1.2, 1],
        hspace=0.30,
        wspace=0.35,
    )

    # --- Top-left: original pattern (clipped to PCA q-range) ---
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
    ax_orig.set_title("Original Pattern")
    fig.colorbar(im_orig, ax=ax_orig, label="Intensity", shrink=0.85)

    # --- Top-right: PCA-reconstructed pattern (complete grid, no beamstop) ---
    ax_recon = fig.add_subplot(gs[0, 1])
    im_recon = ax_recon.imshow(
        recon_grid,
        extent=recon_extent,
        origin="lower",
        aspect="equal",
        norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax),
        cmap="turbo",
    )
    ax_recon.set_xlabel(r"Q$_x$ (Å$^{-1}$)")
    ax_recon.set_ylabel(r"Q$_y$ (Å$^{-1}$)")
    ax_recon.set_title(f"PCA Reconstruction ({n_components} modes)")
    fig.colorbar(im_recon, ax=ax_recon, label="Intensity", shrink=0.85)

    # --- Bottom: PCA coefficients bar chart ---
    ax_coeff = fig.add_subplot(gs[1, :])
    coeffs = alpha.reshape(-1)[:20]
    x = np.arange(1, len(coeffs) + 1)
    ax_coeff.bar(x, coeffs, color="tab:blue")
    ax_coeff.set_xlabel("PCA mode")
    ax_coeff.set_ylabel("Coefficient")
    ax_coeff.set_title("PCA Coefficients")
    ax_coeff.set_xticks(x)

    param_text = (
        f"radius={predictions.get('radius', float('nan')):.3g}\n"
        f"length={predictions.get('length', float('nan')):.3g}\n"
        f"n_cyl={predictions.get('n_cyl', float('nan')):.3g}\n"
        f"stretch={predictions.get('stretch', float('nan')):.3g}"
    )
    ax_coeff.text(
        0.98, 0.98, param_text,
        transform=ax_coeff.transAxes,
        ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle(pattern_path.stem, fontsize=12, fontweight="bold")

    coeff_plot_name = format_prediction_filename(pattern_path, predictions)
    fig.savefig(plots_dir / coeff_plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {plots_dir / coeff_plot_name}")

    if args.output_json:
        output_path = Path(args.output_json)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "orientation_angle_deg": float(np.degrees(orientation_angle)),
                    "model_info": {
                        "n_components": n_components,
                        "parameters": model_metadata
                    }
                },
                f,
                indent=2
            )
        print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
