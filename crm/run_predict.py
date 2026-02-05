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

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    coeffs = alpha.reshape(-1)[:20]
    plt.figure(figsize=(8, 4))
    x = np.arange(1, len(coeffs) + 1)
    plt.bar(x, coeffs, color="tab:blue")
    plt.xlabel("PCA mode")
    plt.ylabel("coefficient")
    plt.title("PCA coefficients (modes 1-10)")
    plt.xticks(x)
    param_text = (
        f"radius={predictions.get('radius', float('nan')):.3g}\n"
        f"length={predictions.get('length', float('nan')):.3g}\n"
        f"n_cyl={predictions.get('n_cyl', float('nan')):.3g}\n"
        f"stretch={predictions.get('stretch', float('nan')):.3g}"
    )
    plt.gca().text(
        0.98,
        0.98,
        param_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
    )
    plt.tight_layout()
    coeff_plot_name = format_prediction_filename(pattern_path, predictions)
    plt.savefig(plots_dir / coeff_plot_name, dpi=300, bbox_inches="tight")
    plt.close()

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
