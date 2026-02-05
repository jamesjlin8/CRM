#!/usr/bin/env python3
"""
Train XGBoost regression models to predict physical parameters from PCA scores.

This script consumes PCA artifacts produced by run_pca_analysis.py and trains
XGBoost regressor models to predict metadata parameters from PCA coefficients.

Usage:
    python run_pca_model.py --pca-results-dir pca_results --model-components 10
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


def parse_metadata_from_filename(filename: str) -> dict:
    """
    Parse physical parameters from filename.
    Expected format: PHIO_0_St{stretch}_{n_cyl}cyl_{radius}r_{length}l.dat
    Example: PHIO_0_St0.05_10cyl_10r_100l.dat
    """
    pattern = r"St([\d.]+)_(\d+)cyl_(\d+)r_(\d+)l"
    match = re.search(pattern, filename)
    if match:
        stretch = float(match.group(1))
        n_cyl = int(match.group(2))
        radius = int(match.group(3))
        length = int(match.group(4))
        return {
            "stretch": stretch,
            "n_cyl": n_cyl,
            "radius": radius,
            "length": length,
        }
    return {
        "stretch": None,
        "n_cyl": None,
        "radius": None,
        "length": None,
    }


def load_scattering_patterns(
    data_dir: Path, max_files: Optional[int] = None
) -> Tuple[np.ndarray, list, np.ndarray, pd.DataFrame]:
    """
    Load scattering patterns from .dat files in the specified directory.

    Returns:
        intensity_matrix: Array of shape (n_patterns, n_q_points)
        file_names: List of source file names
        q_values: Q-values corresponding to the columns (sorted by q magnitude)
        metadata: DataFrame with columns: stretch, n_cyl, radius, length
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    dat_files = sorted(data_dir.rglob("*.dat"))
    dat_files = [f for f in dat_files if not f.name.endswith("_fit.dat")]
    if max_files is not None:
        dat_files = dat_files[:max_files]
    if not dat_files:
        raise ValueError(f"No .dat files found in {data_dir}")

    print(f"Found {len(dat_files)} .dat files to process")

    patterns = []
    file_names = []
    metadata_list = []
    q_values_ref = None

    for idx, dat_file in enumerate(dat_files):
        try:
            data = np.loadtxt(dat_file)
            if data.size == 0:
                print(f"Warning: Empty file: {dat_file}")
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 3:
                print(
                    f"Warning: Invalid format in {dat_file}: expected 3 columns, got {data.shape[1]}"
                )
                continue

            qx = data[:, 0]
            qy = data[:, 1]
            p = data[:, 2]
            q = np.sqrt(qx**2 + qy**2)
            q_sorted_idx = np.argsort(q)
            q_sorted = q[q_sorted_idx]
            p_sorted = p[q_sorted_idx]

            if idx == 0:
                q_values_ref = q_sorted

            patterns.append(p_sorted)
            file_names.append(dat_file.name)
            metadata_list.append(parse_metadata_from_filename(dat_file.name))
        except Exception as exc:
            print(f"Warning: Error loading {dat_file}: {exc}")
            continue

    if not patterns:
        raise ValueError("No valid patterns loaded")

    min_length = min(len(p) for p in patterns)
    patterns = [p[:min_length] for p in patterns]
    if q_values_ref is not None:
        q_values_ref = q_values_ref[:min_length]

    intensity_matrix = np.array(patterns)
    metadata_df = pd.DataFrame(metadata_list)

    print(
        f"Loaded {intensity_matrix.shape[0]} patterns with {intensity_matrix.shape[1]} q-points each"
    )
    return intensity_matrix, file_names, q_values_ref, metadata_df


def load_pca_components(pca_results_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    pca_file = pca_results_dir / "pca_components.pkl"
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA components not found: {pca_file}")
    with open(pca_file, "rb") as f:
        U, S, pca_model = pickle.load(f)
    return U, S, pca_model


def align_intensity_to_pca(intensity_matrix: np.ndarray, mean: np.ndarray) -> np.ndarray:
    n_features = mean.shape[0]
    if intensity_matrix.shape[1] > n_features:
        return intensity_matrix[:, :n_features]
    if intensity_matrix.shape[1] < n_features:
        raise ValueError(
            f"Intensity matrix has {intensity_matrix.shape[1]} features, but PCA expects {n_features}."
        )
    return intensity_matrix


XGB_BASE_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "tree_method": "hist",
}

XGB_BEST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def model_importance(model: XGBRegressor) -> np.ndarray:
    return np.asarray(model.feature_importances_).reshape(-1)


def train_parameter_models(
    alpha: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int,
    n_splits: int = 5,
) -> tuple[dict, dict]:
    n_components = min(n_components, alpha.shape[1])
    param_cols = ["stretch", "n_cyl", "radius", "length"]
    models = {}
    cv_summary = {"per_param": {}}
    importance_results = {}

    for param in param_cols:
        if param not in metadata.columns:
            continue

        y = metadata[param].values
        valid_mask = ~pd.isna(y)
        if valid_mask.sum() < 5:
            print(f"Skipping {param}: insufficient valid samples")
            continue

        X = alpha[valid_mask, :n_components]
        y_clean = y[valid_mask]

        folds = min(n_splits, X.shape[0])
        if folds < 2:
            print(f"Skipping {param}: not enough samples for CV")
            continue

        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        r2_scores = []
        rmse_scores = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]
            model = XGBRegressor(**XGB_BASE_PARAMS, **XGB_BEST_PARAMS)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        cv_summary["per_param"][param] = {
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores, ddof=1)) if len(r2_scores) > 1 else 0.0,
            "rmse_mean": float(np.mean(rmse_scores)),
            "rmse_std": float(np.std(rmse_scores, ddof=1)) if len(rmse_scores) > 1 else 0.0,
            "n_splits": folds,
            "params": dict(XGB_BASE_PARAMS, **XGB_BEST_PARAMS),
        }

        best_model = XGBRegressor(**XGB_BASE_PARAMS, **XGB_BEST_PARAMS)
        best_model.fit(X, y_clean)
        y_pred = best_model.predict(X)

        models[param] = {
            "model_type": "XGBoost",
            "r2_score": cv_summary["per_param"][param]["r2_mean"],
            "model": best_model,
            "params": dict(XGB_BASE_PARAMS, **XGB_BEST_PARAMS),
        }

        importance_results[param] = {
            "r2_score": cv_summary["per_param"][param]["r2_mean"],
            "importance": model_importance(best_model),
            "model_type": "XGBoost",
            "params": dict(XGB_BASE_PARAMS, **XGB_BEST_PARAMS),
        }

    cv_summary = cv_summary["per_param"]

    model_bundle = {
        "n_components": n_components,
        "models": models,
        "cv_summary": cv_summary,
    }
    return model_bundle, importance_results


def print_cv_summary(cv_summary: dict) -> None:
    if not cv_summary:
        print("No cross-validation results to report.")
        return
    print("\nCross-validation summary:")
    for param, metrics in cv_summary.items():
        r2_mean = metrics.get("r2_mean")
        r2_std = metrics.get("r2_std")
        rmse_mean = metrics.get("rmse_mean")
        rmse_std = metrics.get("rmse_std")
        print(
            f"  {param}: "
            f"R2={r2_mean:.4f}±{r2_std:.4f}, "
            f"RMSE={rmse_mean:.4g}±{rmse_std:.4g}"
        )


def plot_predictive_importance(results: dict, output_path: Path) -> None:
    param_cols = ["stretch", "n_cyl", "radius", "length"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    for idx, param in enumerate(param_cols):
        param_results = results.get(param)
        if not param_results:
            axes[idx].text(
                0.5,
                0.5,
                f"No XGBoost result\nfor {param}",
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )
            axes[idx].set_title(param.capitalize(), fontsize=12, fontweight="bold")
            continue

        importance = param_results["importance"]
        pc_labels = [f"PC{i+1}" for i in range(len(importance))]
        axes[idx].barh(pc_labels, importance)
        axes[idx].set_xlabel("Feature Importance", fontsize=11)
        r2_score = param_results.get("r2_score")
        r2_text = f"{r2_score:.3f}" if r2_score is not None else "n/a"
        axes[idx].set_title(
            f"{param.capitalize()} (R²={r2_text}, XGBoost)",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PCA-based parameter models using saved PCA components."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output",
        help="Directory containing .dat files (default: output)",
    )
    parser.add_argument(
        "--pca-results-dir",
        type=str,
        default="pca_results",
        help="Directory containing PCA artifacts (default: pca_results)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="10modes",
        help="Directory to save model outputs (default: 10modes)",
    )
    parser.add_argument(
        "--model-components",
        type=int,
        default=10,
        help="Number of PCA components to use for modeling (default: 10)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits (default: 5)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pca_results_dir = Path(args.pca_results_dir)
    models_dir = Path(args.models_dir) if args.models_dir else pca_results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    U, _S, pca_model = load_pca_components(pca_results_dir)
    mean_intensity = pca_model["mean_"]

    intensity_matrix, file_names, _q_values, metadata = load_scattering_patterns(input_dir)
    intensity_matrix = align_intensity_to_pca(intensity_matrix, mean_intensity)

    n_components = min(args.model_components, U.shape[1])
    centered_data = intensity_matrix - mean_intensity
    alpha = centered_data @ U[:, :n_components]

    if metadata is None or metadata.empty:
        raise ValueError("No metadata available; cannot train parameter models.")

    metadata = metadata.reset_index(drop=True)

    model_bundle, importance_results = train_parameter_models(
        alpha, metadata, n_components=n_components, n_splits=args.n_splits
    )
    print_cv_summary(model_bundle.get("cv_summary", {}))

    models_path = models_dir / "parameter_models.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Saved parameter models to {models_path}")

    gb_importance_path = models_dir / "predictive_importance_xgboost.png"
    plot_predictive_importance(importance_results, gb_importance_path)
    print(f"Saved predictive importance plots to {gb_importance_path}")

    importance_pickle = models_dir / "predictive_importance_results.pkl"
    with open(importance_pickle, "wb") as f:
        pickle.dump(importance_results, f)

    summary_path = models_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(model_bundle.get("cv_summary", {}), f, indent=2)



if __name__ == "__main__":
    main()
