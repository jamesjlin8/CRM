#!/usr/bin/env python3
"""
Train XGBoost regression models to predict physical parameters from PCA scores.
Use --random-search to run RandomizedSearchCV and refresh xgboost_best_params.json.

Default:
    python run_xg_model.py --pca-dir pca_results --model-components 20 --n-splits 5 --models-dir 20modes

Usage:
    python run_xg_model.py [--pca-dir DIR] [--models-dir DIR] [--model-components N] [--n-splits K]
    python run_xg_model.py ... --random-search
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
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.utils import resample
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


def _parse_n_modes_from_rom_path(rom_path: Path) -> Optional[int]:
    """Extract n_modes from reduced_order_model_{n}modes.pkl."""
    match = re.search(r"reduced_order_model_(\d+)modes\.pkl", rom_path.name)
    return int(match.group(1)) if match else None


def load_reduced_order_model(
    pca_results_dir: Path, n_components: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load PCA coefficients and file names from the reduced-order model in pca_results.
    Build metadata from file names. Uses alpha[:, :n_components] for the requested components.

    Returns:
        alpha: Array of shape (n_patterns, n_components)
        metadata: DataFrame with columns stretch, n_cyl, radius, length (from filenames)
    """
    pca_results_dir = Path(pca_results_dir)
    if not pca_results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pca_results_dir}")

    rom_files = sorted(pca_results_dir.glob("reduced_order_model_*modes.pkl"))
    if not rom_files:
        raise FileNotFoundError(
            f"No reduced_order_model_*modes.pkl found in {pca_results_dir}. "
            "Run run_pca_analysis.py first."
        )

    # Prefer ROM with n_modes >= n_components; use smallest such
    candidates = []
    for rom_path in rom_files:
        n_modes = _parse_n_modes_from_rom_path(rom_path)
        if n_modes is not None and n_modes >= n_components:
            candidates.append((n_modes, rom_path))
    if not candidates:
        n_modes_first = _parse_n_modes_from_rom_path(rom_files[0])
        if n_modes_first is not None and n_modes_first < n_components:
            raise ValueError(
                f"ROM has n_modes={n_modes_first}, but --model-components={n_components}. "
                "Use model_components <= n_modes or re-run run_pca_analysis.py."
            )
        chosen_path = rom_files[0]
    else:
        candidates.sort(key=lambda x: x[0])
        chosen_path = candidates[0][1]

    with open(chosen_path, "rb") as f:
        rom = pickle.load(f)
    alpha_full = rom["alpha"]
    file_names = rom["file_names"]
    n_modes_rom = rom.get("n_modes", alpha_full.shape[1])
    alpha = alpha_full[:, : min(n_components, alpha_full.shape[1])]

    metadata_list = [parse_metadata_from_filename(name) for name in file_names]
    metadata_df = pd.DataFrame(metadata_list)
    print(
        f"Loaded ROM from {chosen_path.name}: {alpha.shape[0]} patterns, "
        f"using {alpha.shape[1]} components (ROM has {n_modes_rom} modes)"
    )
    return alpha, metadata_df


XGB_BASE_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "tree_method": "hist",
}

# Search space for RandomizedSearchCV (shared across stretch / n_cyl / radius / length).
XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": list(range(100, 501, 25)),
    "max_depth": list(range(3, 12)),
    "learning_rate": np.round(np.logspace(-2.0, -0.45, 28), 6).tolist(),
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8],
    "gamma": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "reg_alpha": [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
}


def _params_to_serializable(params: dict) -> dict:
    return {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in params.items()
    }


def load_best_params_from_json(models_dir: Path) -> Optional[dict]:
    best_params_path = models_dir / "xgboost_best_params.json"
    if not best_params_path.exists():
        return None

    try:
        with open(best_params_path, "r") as f:
            loaded = json.load(f)
    except Exception as exc:
        print(
            f"Error: failed to read {best_params_path} ({exc}). "
            "Falling back to random search."
        )
        return None

    if not isinstance(loaded, dict):
        print(
            f"Error: {best_params_path} does not contain a JSON object. "
            "Falling back to random search."
        )
        return None

    return loaded


def run_xgb_randomized_search(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_iter: int,
    random_state: int,
) -> tuple[dict, XGBRegressor, RandomizedSearchCV]:
    """Return merged best params dict, refit best_estimator_, and fitted search object."""
    folds = min(n_splits, X.shape[0])
    if folds < 2:
        raise ValueError("RandomizedSearchCV requires at least 2 CV folds.")
    kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    base_est = XGBRegressor(**XGB_BASE_PARAMS)
    search = RandomizedSearchCV(
        estimator=base_est,
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=kfold,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    merged = {**XGB_BASE_PARAMS, **search.best_params_}
    merged = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in merged.items()}
    return merged, search.best_estimator_, search


def model_importance(model: XGBRegressor) -> np.ndarray:
    return np.asarray(model.feature_importances_).reshape(-1)


def train_parameter_models(
    alpha: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int,
    n_splits: int = 5,
    best_params_by_param: Optional[dict] = None,
    *,
    use_random_search: bool = False,
    n_random_iter: int = 50,
    random_search_random_state: int = 42,
) -> tuple[dict, dict]:
    n_components = min(n_components, alpha.shape[1])
    param_cols = ["stretch", "n_cyl", "radius", "length"]
    models = {}
    cv_summary = {}
    importance_results = {}

    for param in param_cols:
        if param not in metadata.columns:
            continue
        if not use_random_search and (
            best_params_by_param is None or param not in best_params_by_param
        ):
            print(f"Skipping {param}: no fixed hyperparameters defined")
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

        search_obj: Optional[RandomizedSearchCV] = None
        if use_random_search:
            print(
                f"  {param}: RandomizedSearchCV (n_iter={n_random_iter}, "
                f"{folds}-fold, neg_RMSE)..."
            )
            best_params, best_model, search_obj = run_xgb_randomized_search(
                X,
                y_clean,
                n_splits=n_splits,
                n_iter=n_random_iter,
                random_state=random_search_random_state,
            )
            best_neg_rmse = float(search_obj.best_score_)
            print(
                f"  {param}: best CV mean neg_RMSE = {best_neg_rmse:.6g} "
                f"(RMSE ≈ {-best_neg_rmse:.6g})"
            )
        else:
            best_params = {**XGB_BASE_PARAMS, **best_params_by_param[param]}
            best_model = XGBRegressor(**best_params)
            best_model.fit(X, y_clean)
            best_neg_rmse = None

        best_params_serializable = _params_to_serializable(best_params)

        # Bootstrap ensemble for input-dependent uncertainty estimation
        N_BOOTSTRAP = 100
        ensemble = []
        for b in range(N_BOOTSTRAP):
            X_boot, y_boot = resample(X, y_clean, random_state=b)
            m = XGBRegressor(**best_params)
            m.fit(X_boot, y_boot)
            ensemble.append(m)
        print(f"  {param}: trained {N_BOOTSTRAP}-model bootstrap ensemble")

        # Compute CV metrics for reporting
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        r2_scores = []
        rmse_scores = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]
            fold_model = XGBRegressor(**best_params)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        cv_entry = {
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores, ddof=1)) if len(r2_scores) > 1 else 0.0,
            "rmse_mean": float(np.mean(rmse_scores)),
            "rmse_std": float(np.std(rmse_scores, ddof=1)) if len(rmse_scores) > 1 else 0.0,
            "n_splits": folds,
            "params": best_params_serializable,
            "random_search": use_random_search,
        }
        if use_random_search and search_obj is not None:
            cv_entry["random_search_n_iter"] = n_random_iter
            cv_entry["random_search_best_neg_rmse"] = float(search_obj.best_score_)
        cv_summary[param] = cv_entry

        models[param] = {
            "model_type": "XGBoost",
            "r2_score": cv_summary[param]["r2_mean"],
            "model": best_model,
            "ensemble": ensemble,
            "params": best_params,
        }

        mean_imp = np.mean([model_importance(m) for m in ensemble], axis=0)
        importance_results[param] = {
            "r2_score": cv_summary[param]["r2_mean"],
            "importance": mean_imp,
            "model_type": "XGBoost",
            "params": best_params,
        }

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
        description="Train PCA-based parameter models using the reduced-order model from pca_results."
    )
    parser.add_argument(
        "--pca-dir",
        type=str,
        default="pca_results",
        dest="pca_dir",
        help="Directory containing PCA artifacts and reduced-order model (default: pca_results)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="20modes",
        help="Directory to save model outputs (default: 20modes)",
    )
    parser.add_argument(
        "--model-components",
        type=int,
        default=20,
        help="Number of PCA components to use for modeling (default: 20)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits for reporting (default: 5)",
    )
    parser.add_argument(
        "--random-search",
        action="store_true",
        help="Force RandomizedSearchCV per parameter (otherwise load xgboost_best_params.json from --models-dir).",
    )
    parser.add_argument(
        "--random-search-iter",
        type=int,
        default=50,
        dest="random_search_iter",
        help="RandomizedSearchCV trials per parameter (default: 50).",
    )
    args = parser.parse_args()
    if args.random_search_iter < 1:
        raise ValueError("--random-search-iter must be >= 1.")

    pca_results_dir = Path(args.pca_dir)
    models_dir = Path(args.models_dir) if args.models_dir else pca_results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    alpha, metadata = load_reduced_order_model(pca_results_dir, args.model_components)
    n_components = alpha.shape[1]

    if metadata.empty:
        raise ValueError("No metadata available; cannot train parameter models.")

    metadata = metadata.reset_index(drop=True)

    loaded_best_params = load_best_params_from_json(models_dir)
    use_random_search = args.random_search
    if not use_random_search and loaded_best_params is None:
        print(
            f"Error: {models_dir / 'xgboost_best_params.json'} not found. "
            "Performing random search instead."
        )
        use_random_search = True

    model_bundle, importance_results = train_parameter_models(
        alpha,
        metadata,
        n_components=n_components,
        n_splits=args.n_splits,
        best_params_by_param=loaded_best_params,
        use_random_search=use_random_search,
        n_random_iter=args.random_search_iter,
    )
    print_cv_summary(model_bundle.get("cv_summary", {}))

    cv_summary = model_bundle.get("cv_summary", {})
    best_params_path = models_dir / "xgboost_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(
            {p: cv_summary[p]["params"] for p in cv_summary},
            f,
            indent=2,
        )
    if use_random_search:
        print(
            f"Saved hyperparameters to {best_params_path} "
            "for future runs."
        )

    models_path = models_dir / "parameter_models.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(model_bundle, f)

    gb_importance_path = models_dir / "predictive_importance_xgboost.png"
    plot_predictive_importance(importance_results, gb_importance_path)

    importance_pickle = models_dir / "predictive_importance_results.pkl"
    with open(importance_pickle, "wb") as f:
        pickle.dump(importance_results, f)

    summary_path = models_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(model_bundle.get("cv_summary", {}), f, indent=2)

    print(f"Saved data to {models_dir}/")

if __name__ == "__main__":
    main()
