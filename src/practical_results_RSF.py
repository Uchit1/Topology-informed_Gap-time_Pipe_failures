"""
Train RSF on all synthetic 2007-2019 gap-time intervals and generate
temporal prediction outputs for all pipes in 2020-2024.

Inputs:
    outputs/survival_data/gap_time_train_intervals_2007_2019_with_embeddings.csv
    outputs/survival_data/eval_window_2020_2024_with_embeddings.csv

Outputs:
    outputs/rsf_temporal_survival_plots_synthetic/
        temporal_predictions_all_pipes.csv
        summary_metrics.csv
        survival_by_risk_groups.png
        survival_by_material.png
        survival_by_installation_year_bins.png

This is a temporal split setup:
    training period:   2007-2019
    prediction period: 2020-2024
    test pipes:        all eligible pipes in the 2020-2024 evaluation file
"""

from __future__ import annotations

import json
import re
import bisect
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_CSV = (
    ROOT_DIR
    / "outputs"
    / "survival_data"
    / "gap_time_train_intervals_2007_2019_with_embeddings.csv"
)

EVAL_CSV = (
    ROOT_DIR
    / "outputs"
    / "survival_data"
    / "eval_window_2020_2024_with_embeddings.csv"
)

OUT_DIR = ROOT_DIR / "outputs" / "rsf_temporal_survival_plots_synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "pipe_id"

SEED = 42

TRAIN_START = 2007
TRAIN_END = 2019
TEST_START = 2020
TEST_END = 2024

TEST_HORIZON_YEARS = 5.0
TOP_FRACTION = 0.20

# Hyperparameter tuning
USE_HYPERPARAMETER_TUNING = True
N_CV_FOLDS = 5
N_ITER = 19

# Event rows can be upweighted. Use 1.0 for no event weighting.
ALPHA_EVENT_WEIGHT = 1.0

# Survival-plot settings
T_PRED_MAX = 10.0
N_RISK_GROUPS = 3


# ---------------------------------------------------------------------
# 2. Input variables
# ---------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "Soil",
    "Material",
    "Landuse",
    "Road Intersections",
    "Near Railways",
]

NUMERIC_BASE_FEATURES = [
    "Length",
    "Diameter",
    "Elevation",
    "GWT",
    "Service Connections",
    "Fire Hydrants",
    "Outlets",
    "Air Release Valve",
    "age_start",
    "num_prev_failures",
]

GRAPH_SCALARS = [
    "khop_fail_sum",
]


# ---------------------------------------------------------------------
# 3. Basic helpers
# ---------------------------------------------------------------------

def read_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found:\n{path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def detect_embedding_columns(df: pd.DataFrame) -> List[str]:
    emb_cols = [c for c in df.columns if re.match(r"^emb_\d+$", str(c))]

    if not emb_cols:
        raise ValueError("No embedding columns found. Expected emb_00, emb_01, ...")

    return sorted(emb_cols, key=lambda c: int(str(c).split("_")[1]))


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_survival_object(df: pd.DataFrame):
    return Surv.from_arrays(
        event=df["event"].astype(bool).to_numpy(),
        time=df["duration"].astype(float).to_numpy(),
    )


def make_row_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df[ID_COL].astype(str).value_counts()

    weights = (
        df[ID_COL]
        .astype(str)
        .map(1.0 / counts)
        .to_numpy(dtype=float)
        .copy()
    )

    weights[~np.isfinite(weights)] = 0.0
    weights = np.maximum(weights, 1e-12)

    if ALPHA_EVENT_WEIGHT != 1.0:
        event_mask = df["event"].astype(bool).to_numpy()
        weights[event_mask] *= ALPHA_EVENT_WEIGHT

    return weights


def build_preprocessor(
    categorical_features: List[str],
    numeric_features: List[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                make_one_hot_encoder(),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )


def build_rsf_pipeline(
    categorical_features: List[str],
    numeric_features: List[str],
    params: Dict | None = None,
) -> Pipeline:
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True,
        }

    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    rsf = RandomSurvivalForest(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        n_jobs=-1,
        random_state=SEED,
    )

    return Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("rsf", rsf),
        ]
    )


# ---------------------------------------------------------------------
# 4. RSF prediction helpers
# ---------------------------------------------------------------------

def rsf_predict_survival_functions(pipe: Pipeline, X_df: pd.DataFrame):
    preprocessor = pipe.named_steps["preproc"]
    rsf = pipe.named_steps["rsf"]

    X_transformed = preprocessor.transform(X_df)

    return rsf.predict_survival_function(
        X_transformed,
        return_array=False,
    )


def cumulative_hazard_risk_scores(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    observed_times: np.ndarray,
) -> np.ndarray:
    preprocessor = pipe.named_steps["preproc"]
    rsf = pipe.named_steps["rsf"]

    X_transformed = preprocessor.transform(X_eval)

    chfs = rsf.predict_cumulative_hazard_function(
        X_transformed,
        return_array=False,
    )

    def hazard_at_time(chf, t):
        times = chf.x
        values = chf.y

        idx = bisect.bisect_right(times, float(t)) - 1

        if idx < 0:
            return 0.0

        idx = min(idx, len(values) - 1)

        return float(values[idx])

    return np.asarray(
        [
            hazard_at_time(chf, t)
            for chf, t in zip(chfs, observed_times)
        ],
        dtype=float,
    )


def failure_probability_at_horizon(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    horizon_years: float,
) -> np.ndarray:
    survival_functions = rsf_predict_survival_functions(pipe, X_eval)

    def survival_at_time(sf, t):
        times = sf.x
        values = sf.y

        idx = bisect.bisect_right(times, float(t)) - 1

        if idx < 0:
            return 1.0

        idx = min(idx, len(values) - 1)

        return float(values[idx])

    return np.asarray(
        [
            1.0 - survival_at_time(sf, horizon_years)
            for sf in survival_functions
        ],
        dtype=float,
    )


def evaluate_survival_grid(
    pipe: Pipeline,
    X: pd.DataFrame,
    t_grid: np.ndarray,
) -> np.ndarray:
    survival_functions = rsf_predict_survival_functions(pipe, X)

    def survival_on_grid(sf):
        times = sf.x
        values = sf.y

        out = []

        for t in t_grid:
            idx = bisect.bisect_right(times, float(t)) - 1

            if idx < 0:
                out.append(1.0)
            else:
                idx = min(idx, len(values) - 1)
                out.append(float(values[idx]))

        return np.asarray(out, dtype=float)

    return np.vstack([survival_on_grid(sf) for sf in survival_functions])


def harrell_c_index(
    y_event: np.ndarray,
    y_time: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    result = concordance_index_censored(
        event_indicator=np.asarray(y_event, dtype=bool),
        event_time=np.asarray(y_time, dtype=float),
        estimate=np.asarray(risk_scores, dtype=float),
    )

    return float(result[0])


def top20_recall_from_probability(
    y_true: np.ndarray,
    failure_probability: np.ndarray,
    top_fraction: float,
) -> Tuple[float, int, int, int]:
    y_true = np.asarray(y_true, dtype=int)
    failure_probability = np.asarray(failure_probability, dtype=float)

    n = len(failure_probability)
    k = max(1, int(round(top_fraction * n)))

    top_idx = np.argsort(failure_probability)[::-1][:k]

    total_failures = int(y_true.sum())
    captured_failures = int(y_true[top_idx].sum())

    recall = (
        captured_failures / total_failures
        if total_failures > 0
        else 0.0
    )

    return float(recall), int(k), captured_failures, total_failures


# ---------------------------------------------------------------------
# 5. Hyperparameter tuning using Harrell C-index
# ---------------------------------------------------------------------

def sample_rsf_param_sets(n_iter: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)

    max_features_pool = np.array(
        ["sqrt", "log2", 0.35, 0.50, 0.75],
        dtype=object,
    )

    param_sets = []

    for _ in range(max(1, n_iter)):
        param_sets.append(
            {
                "n_estimators": int(rng.integers(100, 501)),
                "max_depth": int(rng.integers(3, 16)),
                "min_samples_split": int(rng.integers(4, 31)),
                "min_samples_leaf": int(rng.integers(3, 16)),
                "max_features": rng.choice(max_features_pool),
                "bootstrap": True,
            }
        )

    param_sets.append(
        {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True,
        }
    )

    return param_sets


def make_pipe_level_cv_splits(
    df_train: pd.DataFrame,
    n_folds: int,
    seed: int,
):
    pipe_event = (
        df_train.groupby(ID_COL)["event"]
        .max()
        .astype(int)
    )

    df_cv = df_train.copy()
    df_cv["__pipe_event__"] = df_cv[ID_COL].map(pipe_event).astype(int)

    groups = df_cv[ID_COL].astype(str).to_numpy()
    y_strat = df_cv["__pipe_event__"].astype(int).to_numpy()

    class_counts = pipe_event.value_counts()

    if len(class_counts) >= 2 and class_counts.min() >= 2:
        usable_folds = min(n_folds, int(class_counts.min()))
        usable_folds = max(2, usable_folds)

        splitter = StratifiedGroupKFold(
            n_splits=usable_folds,
            shuffle=True,
            random_state=seed,
        )

        return list(splitter.split(df_cv, y_strat, groups=groups))

    unique_pipes = df_train[ID_COL].nunique()
    usable_folds = min(n_folds, unique_pipes)

    if usable_folds < 2:
        raise ValueError("At least two unique pipes are needed for CV tuning.")

    print(
        "Warning: StratifiedGroupKFold was not possible. "
        "Using GroupKFold instead."
    )

    splitter = GroupKFold(n_splits=usable_folds)

    return list(splitter.split(df_cv, y_strat, groups=groups))


def tune_rsf_hyperparameters_harrell_c(
    df_train: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
    feature_cols: List[str],
) -> Tuple[Dict, pd.DataFrame]:
    print("\nStarting RSF hyperparameter tuning using Harrell C-index...")

    param_sets = sample_rsf_param_sets(
        n_iter=N_ITER,
        seed=SEED,
    )

    folds = make_pipe_level_cv_splits(
        df_train=df_train,
        n_folds=N_CV_FOLDS,
        seed=SEED,
    )

    tuning_records = []

    best_params = None
    best_mean_c = -np.inf

    for rank, params in enumerate(param_sets):
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            df_tr = df_train.iloc[train_idx].copy()
            df_val = df_train.iloc[val_idx].copy()

            X_tr = df_tr[feature_cols].copy()
            y_tr = make_survival_object(df_tr)
            w_tr = make_row_weights(df_tr)

            X_val = df_val[feature_cols].copy()

            y_val_event = df_val["event"].astype(bool).to_numpy()
            y_val_time = df_val["duration"].astype(float).to_numpy()

            try:
                pipe = build_rsf_pipeline(
                    categorical_features=categorical_features,
                    numeric_features=numeric_features,
                    params=params,
                )

                pipe.fit(
                    X_tr,
                    y_tr,
                    rsf__sample_weight=w_tr,
                )

                val_risk = cumulative_hazard_risk_scores(
                    pipe=pipe,
                    X_eval=X_val,
                    observed_times=y_val_time,
                )

                c_val = harrell_c_index(
                    y_event=y_val_event,
                    y_time=y_val_time,
                    risk_scores=val_risk,
                )

                fold_scores.append(float(c_val))

            except Exception as error:
                print(
                    f"Warning: candidate {rank}, fold {fold_idx} failed: {error}"
                )
                fold_scores.append(float("nan"))

        valid_scores = [s for s in fold_scores if np.isfinite(s)]

        mean_c = float(np.mean(valid_scores)) if valid_scores else float("nan")
        std_c = (
            float(np.std(valid_scores, ddof=1))
            if len(valid_scores) > 1
            else float("nan")
        )

        tuning_records.append(
            {
                "rank": rank,
                "params": params,
                "fold_harrell_c": fold_scores,
                "mean_harrell_c": mean_c,
                "std_harrell_c": std_c,
            }
        )

        print(
            f"Candidate {rank + 1}/{len(param_sets)} | "
            f"mean Harrell C = {mean_c:.4f}"
        )

        if np.isfinite(mean_c) and mean_c > best_mean_c:
            best_mean_c = mean_c
            best_params = params

    if best_params is None:
        raise RuntimeError("No successful RSF hyperparameter candidate was fitted.")

    tuning_log = pd.DataFrame(tuning_records)

    tuning_log.to_json(
        OUT_DIR / "rsf_hyperparameter_tuning_harrell_c.json",
        orient="records",
        indent=2,
    )

    tuning_log.to_csv(
        OUT_DIR / "rsf_hyperparameter_tuning_harrell_c.csv",
        index=False,
    )

    with open(OUT_DIR / "best_rsf_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\nBest RSF hyperparameters:")
    print(best_params)
    print(f"Best mean CV Harrell C-index: {best_mean_c:.4f}")

    return best_params, tuning_log


# ---------------------------------------------------------------------
# 6. Survival plotting helpers
# ---------------------------------------------------------------------

def km_line_on_grid(
    durations: np.ndarray,
    events: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    km_t, km_s = kaplan_meier_estimator(
        events.astype(bool),
        durations.astype(float),
    )

    out = []

    for t in grid:
        idx = bisect.bisect_right(km_t, float(t)) - 1

        if idx < 0:
            out.append(1.0)
        else:
            idx = min(idx, len(km_s) - 1)
            out.append(float(km_s[idx]))

    return np.asarray(out, dtype=float)


def plot_survival_by_risk_groups(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    df_eval: pd.DataFrame,
    out_path: Path,
    test_horizon_years: float = 5.0,
    t_pred_max: float = 10.0,
    n_groups: int = 3,
) -> None:
    t_grid = np.linspace(0.0, float(t_pred_max), 101)
    idx_observed = t_grid <= float(test_horizon_years)
    t_grid_observed = t_grid[idx_observed]

    survival_matrix = evaluate_survival_grid(pipe, X_eval, t_grid)

    df_plot = df_eval.copy()

    if "failure_prob_5yr" not in df_plot.columns:
        raise KeyError("df_eval must contain failure_prob_5yr.")

    try:
        df_plot["risk_group"] = (
            pd.qcut(
                df_plot["failure_prob_5yr"],
                q=n_groups,
                labels=False,
                duplicates="drop",
            )
            + 1
        )
    except Exception:
        ranked = df_plot["failure_prob_5yr"].rank(method="first")
        df_plot["risk_group"] = (
            pd.qcut(
                ranked,
                q=n_groups,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

    groups = sorted(df_plot["risk_group"].dropna().unique())

    plt.figure(figsize=(8, 6))

    plt.axvspan(
        float(test_horizon_years),
        float(t_pred_max),
        color="tab:blue",
        alpha=0.06,
        zorder=0,
    )

    plt.axvline(
        float(test_horizon_years),
        linestyle="--",
        linewidth=1,
        color="black",
        zorder=3,
    )

    palette = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]
    ymins = []

    for i, group in enumerate(groups):
        mask = (df_plot["risk_group"] == group).to_numpy()

        if not mask.any():
            continue

        color = palette[i % len(palette)]

        group_survival = survival_matrix[mask, :]
        median_survival = np.nanmedian(group_survival, axis=0)
        q25, q75 = np.nanpercentile(group_survival, [25, 75], axis=0)

        plt.fill_between(
            t_grid[idx_observed],
            q25[idx_observed],
            q75[idx_observed],
            color=color,
            alpha=0.15,
            linewidth=0,
            step="post",
        )

        plt.plot(
            t_grid,
            median_survival,
            linestyle="--",
            linewidth=2,
            color=color,
            drawstyle="steps-post",
            label=f"Group {int(group)} — predicted median (n={int(mask.sum())})",
        )

        durations = df_plot.loc[mask, "duration"].to_numpy(float)
        events = df_plot.loc[mask, "event"].to_numpy(bool)

        km_curve = km_line_on_grid(
            durations=durations,
            events=events,
            grid=t_grid_observed,
        )

        km_full = np.full_like(t_grid, np.nan, dtype=float)
        km_full[idx_observed] = km_curve

        plt.plot(
            t_grid,
            km_full,
            linestyle="-",
            linewidth=2,
            color=color,
            drawstyle="steps-post",
            label=(
                f"Group {int(group)} — KM "
                f"(0–{int(test_horizon_years)}y, "
                f"events={int(events.sum())})"
            ),
        )

        ymins.append(np.nanmin([median_survival[idx_observed].min(), km_curve.min()]))

    ymin = max(0.0, min(ymins) - 0.03) if ymins else 0.0

    plt.xlabel("Years since 2020", fontsize=14)
    plt.ylabel("Survival S(t)", fontsize=14)
    plt.title("Survival by predicted 5-year risk groups", fontsize=16)
    plt.ylim(ymin, 1.02)
    plt.xlim(0.0, float(t_pred_max))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_survival_by_material(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    df_eval: pd.DataFrame,
    out_path: Path,
    material_col: str = "Material",
    test_horizon_years: float = 5.0,
    t_pred_max: float = 10.0,
    max_materials: int = 4,
) -> None:
    if material_col not in df_eval.columns:
        raise KeyError(f"Column '{material_col}' not found in evaluation data.")

    t_grid = np.linspace(0.0, float(t_pred_max), 101)
    idx_observed = t_grid <= float(test_horizon_years)
    t_grid_observed = t_grid[idx_observed]

    survival_matrix = evaluate_survival_grid(pipe, X_eval, t_grid)

    df_plot = df_eval.copy()
    material_raw = df_plot[material_col].astype("string").fillna("Unknown")

    top_materials = material_raw.value_counts().nlargest(max_materials).index.tolist()
    df_plot["material_group"] = np.where(
        material_raw.isin(top_materials),
        material_raw,
        np.nan,
    )

    groups = [
        g for g in top_materials
        if (df_plot["material_group"] == g).any()
    ]

    plt.figure(figsize=(8, 6))

    plt.axvspan(
        float(test_horizon_years),
        float(t_pred_max),
        color="tab:blue",
        alpha=0.06,
        zorder=0,
    )

    plt.axvline(
        float(test_horizon_years),
        linestyle="--",
        linewidth=1,
        color="black",
        zorder=3,
    )

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    ymins = []

    for i, group in enumerate(groups):
        mask = (df_plot["material_group"] == group).to_numpy()

        if not mask.any():
            continue

        color = palette[i % len(palette)]

        group_survival = survival_matrix[mask, :]
        median_survival = np.nanmedian(group_survival, axis=0)
        q25, q75 = np.nanpercentile(group_survival, [25, 75], axis=0)

        plt.fill_between(
            t_grid[idx_observed],
            q25[idx_observed],
            q75[idx_observed],
            color=color,
            alpha=0.15,
            linewidth=0,
            step="post",
        )

        plt.plot(
            t_grid,
            median_survival,
            linestyle="--",
            linewidth=2,
            color=color,
            drawstyle="steps-post",
            label=f"{group} — predicted median (n={int(mask.sum())})",
        )

        durations = df_plot.loc[mask, "duration"].to_numpy(float)
        events = df_plot.loc[mask, "event"].to_numpy(bool)

        km_curve = km_line_on_grid(
            durations=durations,
            events=events,
            grid=t_grid_observed,
        )

        km_full = np.full_like(t_grid, np.nan, dtype=float)
        km_full[idx_observed] = km_curve

        plt.plot(
            t_grid,
            km_full,
            linestyle="-",
            linewidth=2,
            color=color,
            drawstyle="steps-post",
            label=(
                f"{group} — KM "
                f"(0–{int(test_horizon_years)}y, "
                f"events={int(events.sum())})"
            ),
        )

        ymins.append(np.nanmin([median_survival[idx_observed].min(), km_curve.min()]))

    ymin = max(0.0, min(ymins) - 0.03) if ymins else 0.0

    plt.xlabel("Years since 2020", fontsize=14)
    plt.ylabel("Survival S(t)", fontsize=14)
    plt.title("Survival by material", fontsize=16)
    plt.ylim(ymin, 1.02)
    plt.xlim(0.0, float(t_pred_max))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_survival_by_installation_year_bins(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    df_eval: pd.DataFrame,
    out_path: Path,
    test_horizon_years: float = 5.0,
    t_pred_max: float = 10.0,
) -> None:
    """
    Installation year is reconstructed as:
        installation_year = start_year - age_start

    For the 2020-2024 evaluation file:
        start_year = 2020
        age_start = pipe age at 2020
    """
    t_grid = np.linspace(0.0, float(t_pred_max), 101)
    idx_observed = t_grid <= float(test_horizon_years)
    t_grid_observed = t_grid[idx_observed]

    survival_matrix = evaluate_survival_grid(pipe, X_eval, t_grid)

    df_plot = df_eval.copy()

    if "installation_year" not in df_plot.columns:
        if "start_year" not in df_plot.columns or "age_start" not in df_plot.columns:
            raise KeyError(
                "Need either installation_year, or both start_year and age_start."
            )

        df_plot["installation_year"] = (
            pd.to_numeric(df_plot["start_year"], errors="coerce")
            - pd.to_numeric(df_plot["age_start"], errors="coerce")
        )

    bins = [1940, 1960, 1980, 2000, 2021]
    labels = [
        "1940–1959",
        "1960–1979",
        "1980–1999",
        "2000–2020",
    ]

    df_plot["installation_bin"] = pd.cut(
        df_plot["installation_year"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    groups = [
        lab for lab in labels
        if (df_plot["installation_bin"] == lab).any()
    ]

    plt.figure(figsize=(8, 6))

    plt.axvspan(
        float(test_horizon_years),
        float(t_pred_max),
        color="tab:blue",
        alpha=0.06,
        zorder=0,
    )

    plt.axvline(
        float(test_horizon_years),
        linestyle="--",
        linewidth=1,
        color="black",
        zorder=3,
    )

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    ymins = []

    for i, group in enumerate(groups):
        mask = (df_plot["installation_bin"] == group).to_numpy()

        if not mask.any():
            continue

        color = palette[i % len(palette)]

        group_survival = survival_matrix[mask, :]
        median_survival = np.nanmedian(group_survival, axis=0)
        q25, q75 = np.nanpercentile(group_survival, [25, 75], axis=0)

        plt.fill_between(
            t_grid[idx_observed],
            q25[idx_observed],
            q75[idx_observed],
            color=color,
            alpha=0.15,
            linewidth=0,
            step="post",
        )

        plt.plot(
            t_grid,
            median_survival,
            linestyle="--",
            linewidth=2,
            color=color,
            drawstyle="steps-post",
            label=f"{group} — predicted median (n={int(mask.sum())})",
        )

        durations = df_plot.loc[mask, "duration"].to_numpy(float)
        events = df_plot.loc[mask, "event"].to_numpy(bool)

        km_curve = km_line_on_grid(
            durations=durations,
            events=events,
            grid=t_grid_observed,
        )

        km_full = np.full_like(t_grid, np.nan, dtype=float)
        km_full[idx_observed] = km_curve

        plt.plot(
            t_grid,
            km_full,
            linestyle="-",
            linewidth=2,
            color=color,
            drawstyle="steps-post",
            label=(
                f"{group} — KM "
                f"(0–{int(test_horizon_years)}y, "
                f"events={int(events.sum())})"
            ),
        )

        ymins.append(np.nanmin([median_survival[idx_observed].min(), km_curve.min()]))

    ymin = max(0.0, min(ymins) - 0.03) if ymins else 0.0

    plt.xlabel("Years since 2020", fontsize=14)
    plt.ylabel("Survival S(t)", fontsize=14)
    plt.title("Survival by installation year", fontsize=16)
    plt.ylim(ymin, 1.02)
    plt.xlim(0.0, float(t_pred_max))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# 7. Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    print("Training RSF using all synthetic temporal training intervals.")
    print(f"Training file:   {TRAIN_CSV}")
    print(f"Evaluation file: {EVAL_CSV}")
    print(f"Output folder:   {OUT_DIR}")

    df_train = read_input_csv(TRAIN_CSV)
    df_eval = read_input_csv(EVAL_CSV)

    required_cols = [
        ID_COL,
        "start_year",
        "end_year",
        "duration",
        "event",
        "age_start",
        "num_prev_failures",
        "Material",
    ]

    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Training file missing required column: {col}")
        if col not in df_eval.columns:
            raise ValueError(f"Evaluation file missing required column: {col}")

    df_train[ID_COL] = df_train[ID_COL].astype(str).str.strip()
    df_eval[ID_COL] = df_eval[ID_COL].astype(str).str.strip()

    for df in [df_train, df_eval]:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
        df["age_start"] = pd.to_numeric(df["age_start"], errors="coerce")
        df["num_prev_failures"] = pd.to_numeric(df["num_prev_failures"], errors="coerce")

    df_train = df_train.dropna(subset=["duration"]).copy()
    df_eval = df_eval.dropna(subset=["duration"]).copy()

    if (df_train["duration"] <= 0).any():
        raise ValueError("Training data contains non-positive duration values.")

    if (df_eval["duration"] <= 0).any():
        raise ValueError("Evaluation data contains non-positive duration values.")

    emb_cols = detect_embedding_columns(df_train)

    categorical_features = CATEGORICAL_FEATURES.copy()
    numeric_features = NUMERIC_BASE_FEATURES + GRAPH_SCALARS + emb_cols

    feature_cols = categorical_features + numeric_features

    missing_train = [c for c in feature_cols if c not in df_train.columns]
    missing_eval = [c for c in feature_cols if c not in df_eval.columns]

    if missing_train:
        raise ValueError(f"Training data missing required features: {missing_train}")

    if missing_eval:
        raise ValueError(f"Evaluation data missing required features: {missing_eval}")

    print("\nData summary:")
    print(f"Training rows: {len(df_train)}")
    print(f"Training pipes: {df_train[ID_COL].nunique()}")
    print(f"Training events: {int(df_train['event'].sum())}")
    print(f"Evaluation rows: {len(df_eval)}")
    print(f"Evaluation pipes: {df_eval[ID_COL].nunique()}")
    print(f"Evaluation events: {int(df_eval['event'].sum())}")

    print("\nFeature summary:")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numeric non-embedding features: {len(NUMERIC_BASE_FEATURES + GRAPH_SCALARS)}")
    print(f"Embedding features: {len(emb_cols)}")
    print(f"Total input variables before encoding/scaling: {len(feature_cols)}")

    X_train = df_train[feature_cols].copy()
    y_train = make_survival_object(df_train)
    weights = make_row_weights(df_train)

    X_eval = df_eval[feature_cols].copy()
    y_eval_event = df_eval["event"].astype(bool).to_numpy()
    y_eval_time = df_eval["duration"].astype(float).to_numpy()
    y_eval_binary = df_eval["event"].astype(int).to_numpy()

    if USE_HYPERPARAMETER_TUNING:
        best_params, tuning_log = tune_rsf_hyperparameters_harrell_c(
            df_train=df_train,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            feature_cols=feature_cols,
        )
    else:
        best_params = None

    pipe = build_rsf_pipeline(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        params=best_params,
    )

    print("\nFitting final RSF model on all 2007-2019 training intervals...")
    pipe.fit(X_train, y_train, rsf__sample_weight=weights)

    print("Predicting temporal 2020-2024 evaluation window...")
    risk_scores = cumulative_hazard_risk_scores(
        pipe=pipe,
        X_eval=X_eval,
        observed_times=y_eval_time,
    )

    failure_prob_5yr = failure_probability_at_horizon(
        pipe=pipe,
        X_eval=X_eval,
        horizon_years=TEST_HORIZON_YEARS,
    )

    harrell_c = harrell_c_index(
        y_event=y_eval_event,
        y_time=y_eval_time,
        risk_scores=risk_scores,
    )

    top20_recall, top20_k, top20_caught, total_failures = top20_recall_from_probability(
        y_true=y_eval_binary,
        failure_probability=failure_prob_5yr,
        top_fraction=TOP_FRACTION,
    )

    print("\nTemporal evaluation results:")
    print(f"Harrell C-index: {harrell_c:.4f}")
    print(
        f"Top 20% recall: {top20_recall:.4f} "
        f"({top20_caught}/{total_failures} failures captured in top {top20_k} pipes)"
    )

    df_out = df_eval.copy()
    df_out["risk_score_cumhaz"] = risk_scores
    df_out["failure_prob_5yr"] = failure_prob_5yr
    df_out["rank_by_failure_prob"] = (
        df_out["failure_prob_5yr"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    df_out["in_top20_percent"] = (
        df_out["rank_by_failure_prob"] <= top20_k
    ).astype(int)

    # Reconstruct installation year from start_year and age_start.
    df_out["installation_year"] = (
        pd.to_numeric(df_out["start_year"], errors="coerce")
        - pd.to_numeric(df_out["age_start"], errors="coerce")
    )

    df_out.to_csv(
        OUT_DIR / "temporal_predictions_all_pipes.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "temporal_train_period": f"{TRAIN_START}-{TRAIN_END}",
                "temporal_eval_period": f"{TEST_START}-{TEST_END}",
                "harrell_c_index": harrell_c,
                "top20_recall": top20_recall,
                "top20_k": top20_k,
                "top20_caught_failures": top20_caught,
                "total_failures": total_failures,
                "hyperparameter_tuning_used": USE_HYPERPARAMETER_TUNING,
                "best_rsf_params": json.dumps(best_params),
                "n_train_rows": len(df_train),
                "n_train_pipes": df_train[ID_COL].nunique(),
                "n_train_events": int(df_train["event"].sum()),
                "n_eval_rows": len(df_out),
                "n_eval_pipes": df_out[ID_COL].nunique(),
                "n_eval_events": int(df_out["event"].sum()),
                "n_embedding_features": len(emb_cols),
            }
        ]
    )

    summary.to_csv(
        OUT_DIR / "summary_metrics.csv",
        index=False,
    )

    with open(OUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary.iloc[0].to_dict(), f, indent=2)

    print("\nGenerating survival plots...")

    plot_survival_by_risk_groups(
        pipe=pipe,
        X_eval=X_eval,
        df_eval=df_out,
        out_path=OUT_DIR / "survival_by_risk_groups.png",
        test_horizon_years=TEST_HORIZON_YEARS,
        t_pred_max=T_PRED_MAX,
        n_groups=N_RISK_GROUPS,
    )

    plot_survival_by_material(
        pipe=pipe,
        X_eval=X_eval,
        df_eval=df_out,
        out_path=OUT_DIR / "survival_by_material.png",
        material_col="Material",
        test_horizon_years=TEST_HORIZON_YEARS,
        t_pred_max=T_PRED_MAX,
        max_materials=4,
    )

    plot_survival_by_installation_year_bins(
        pipe=pipe,
        X_eval=X_eval,
        df_eval=df_out,
        out_path=OUT_DIR / "survival_by_installation_year_bins.png",
        test_horizon_years=TEST_HORIZON_YEARS,
        t_pred_max=T_PRED_MAX,
    )

    print("\nFiles saved:")
    print(f"Predictions:       {OUT_DIR / 'temporal_predictions_all_pipes.csv'}")
    print(f"Summary metrics:   {OUT_DIR / 'summary_metrics.csv'}")
    print(f"Risk groups plot:  {OUT_DIR / 'survival_by_risk_groups.png'}")
    print(f"Material plot:     {OUT_DIR / 'survival_by_material.png'}")
    print(f"Install year plot: {OUT_DIR / 'survival_by_installation_year_bins.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
