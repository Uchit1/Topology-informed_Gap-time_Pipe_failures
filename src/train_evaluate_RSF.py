"""
Train and evaluate a combined Random Survival Forest using synthetic Net3
gap-time data with a pipe-level held-out test design.

The script trains on 2007–2019 recurrent gap-time intervals from trainval
pipes and evaluates 2020–2024 failure risk on held-out test pipes only.

Inputs:
    outputs/survival_data/gap_time_train_intervals_2007_2019_with_embeddings.csv
    outputs/survival_data/eval_window_2020_2024_with_embeddings.csv

Outputs:
    outputs/rsf_results_synthetic_minimal/
        test_predictions.csv
        summary_metrics.csv
        permutation_importance_harrell_c.csv
        permutation_importance_harrell_c.png

Metrics reported:
    - Harrell C-index
    - Top 20% recall
    - Permutation importance based on decrease in Harrell C-index

Model:
    Random Survival Forest using:
        baseline input variables
        khop_fail_sum
        emb_00 ... emb_63
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, train_test_split

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

OUT_DIR = ROOT_DIR / "outputs" / "rsf_results_synthetic_minimal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "pipe_id"

SEED = 42

# Pipe-level held-out test design
HOLDOUT_TEST_FRAC = 0.20
CV_STRATIFY_ON = "any_event_in_train"
SPLIT_OUT = OUT_DIR / f"pipe_split_seed{SEED}.json"

TEST_HORIZON_YEARS = 5.0
TOP_FRACTION = 0.20

# Number of permutation repeats for variable importance.
N_PERM_REPEATS = 30

# Event rows can be upweighted. Use 1.0 for no event weighting.
ALPHA_EVENT_WEIGHT = 1

N_CV_FOLDS = 5
N_ITER = 19
USE_HYPERPARAMETER_TUNING = True



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
# 3. Helpers
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

    emb_cols = sorted(
        emb_cols,
        key=lambda c: int(str(c).split("_")[1])
    )

    return emb_cols


def make_one_hot_encoder():
    """
    Make OneHotEncoder compatible with different scikit-learn versions.
    """
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
    params: dict | None = None,
) -> Pipeline:
    """
    Build RSF pipeline.

    If params is None, use stable default RSF parameters.
    If params is provided, use tuned RSF parameters.
    """
    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True,
        }

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


def sample_rsf_param_sets(n_iter: int, seed: int) -> List[dict]:
    """
    Randomly sample RSF hyperparameter combinations.

    The ranges are intentionally modest because the synthetic Net3 dataset
    is small and the purpose is a reproducible public example.
    """
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

    # Always include one stable default candidate.
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

def make_or_load_pipe_holdout_split(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    seed: int,
    holdout_frac: float,
) -> tuple[set[str], set[str]]:
    """
    Create a pipe-level trainval/test split.

    The split is based on unique pipe IDs. Stratification uses whether each
    pipe has at least one event in the 2007–2019 training interval table.
    The same split is then applied to both:
        - training intervals
        - 2020–2024 evaluation window
    """
    if SPLIT_OUT.exists():
        split = json.loads(SPLIT_OUT.read_text())
    
        if int(split.get("seed", seed)) != int(seed):
            raise RuntimeError(
                f"Existing split file was created with seed={split.get('seed')}, "
                f"but current SEED={seed}. Delete {SPLIT_OUT} or use the same seed."
            )
    
        if abs(float(split.get("holdout_test_frac", holdout_frac)) - float(holdout_frac)) > 1e-9:
            raise RuntimeError(
                f"Existing split file was created with holdout_test_frac="
                f"{split.get('holdout_test_frac')}, but current HOLDOUT_TEST_FRAC="
                f"{holdout_frac}. Delete {SPLIT_OUT} or use the same setting."
            )
    
        trainval_ids = set(map(str, split["trainval_ids"]))
        test_ids = set(map(str, split["test_ids"]))
    
        overlap = trainval_ids.intersection(test_ids)
        if overlap:
            raise RuntimeError(
                f"Stored split file contains train/test overlap: {len(overlap)} pipes."
            )
    
        return trainval_ids, test_ids

    train_ids = set(df_train[ID_COL].astype(str).str.strip())
    eval_ids = set(df_eval[ID_COL].astype(str).str.strip())

    eligible_ids = sorted(train_ids.intersection(eval_ids))

    if len(eligible_ids) < 10:
        raise ValueError(
            f"Too few common pipes for holdout splitting: {len(eligible_ids)}"
        )

    pipe_event = (
        df_train[df_train[ID_COL].astype(str).isin(eligible_ids)]
        .groupby(ID_COL)["event"]
        .max()
        .reindex(eligible_ids)
        .fillna(0)
        .astype(int)
    )

    y_strat = pipe_event.to_numpy()

    if len(np.unique(y_strat)) >= 2 and min(np.bincount(y_strat)) >= 2:
        stratify = y_strat
    else:
        stratify = None
        print(
            "Warning: stratified pipe split not possible because one class "
            "has too few pipes. Using unstratified pipe split."
        )

    trainval_ids, test_ids = train_test_split(
        np.asarray(eligible_ids),
        test_size=holdout_frac,
        random_state=seed,
        stratify=stratify,
    )

    split = {
        "seed": int(seed),
        "holdout_test_frac": float(holdout_frac),
        "stratify_on": CV_STRATIFY_ON,
        "trainval_ids": trainval_ids.astype(str).tolist(),
        "test_ids": test_ids.astype(str).tolist(),
        "stratified": stratify is not None,
    }

    SPLIT_OUT.write_text(json.dumps(split, indent=2))

    return set(split["trainval_ids"]), set(split["test_ids"])

def make_pipe_level_cv_splits(
    df_train: pd.DataFrame,
    n_folds: int,
    seed: int,
):
    """
    Create pipe-level CV folds.

    Returns row indices for df_train, but the splitting is performed on
    unique pipe IDs so all intervals from the same pipe remain in the
    same fold.
    """
    pipe_event = (
        df_train.groupby(ID_COL)["event"]
        .max()
        .astype(int)
    )

    pipe_ids = pipe_event.index.astype(str).to_numpy()
    y_pipe = pipe_event.to_numpy()
    groups = pipe_ids.copy()

    class_counts = pd.Series(y_pipe).value_counts()

    if len(class_counts) >= 2 and class_counts.min() >= 2:
        usable_folds = min(n_folds, int(class_counts.min()))
        usable_folds = max(2, usable_folds)

        splitter = StratifiedGroupKFold(
            n_splits=usable_folds,
            shuffle=True,
            random_state=seed,
        )

        pipe_splits = splitter.split(
            np.zeros((len(pipe_ids), 1)),
            y_pipe,
            groups=groups,
        )
    else:
        usable_folds = min(n_folds, len(pipe_ids))

        if usable_folds < 2:
            raise ValueError("At least two unique pipes are needed for CV tuning.")

        print(
            "Warning: StratifiedGroupKFold was not possible because one class "
            "has too few pipes. Using GroupKFold instead."
        )

        splitter = GroupKFold(n_splits=usable_folds)

        pipe_splits = splitter.split(
            np.zeros((len(pipe_ids), 1)),
            y_pipe,
            groups=groups,
        )

    row_splits = []

    for train_pipe_idx, val_pipe_idx in pipe_splits:
        train_pipe_ids = set(pipe_ids[train_pipe_idx])
        val_pipe_ids = set(pipe_ids[val_pipe_idx])

        train_rows = np.where(
            df_train[ID_COL].astype(str).isin(train_pipe_ids).to_numpy()
        )[0]

        val_rows = np.where(
            df_train[ID_COL].astype(str).isin(val_pipe_ids).to_numpy()
        )[0]

        row_splits.append((train_rows, val_rows))

    return row_splits

def tune_rsf_hyperparameters_harrell_c(
    df_train: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
    feature_cols: List[str],
) -> tuple[dict, pd.DataFrame]:
    """
    Tune RSF hyperparameters using mean Harrell C-index over pipe-level CV folds.
    """
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
                    f"Warning: tuning candidate {rank}, fold {fold_idx} failed: {error}"
                )
                fold_scores.append(float("nan"))

        valid_scores = [
            s for s in fold_scores
            if np.isfinite(s)
        ]

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
            f"mean Harrell C = {mean_c:.4f} | params = {params}"
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



def cumulative_hazard_risk_scores(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    observed_times: np.ndarray,
) -> np.ndarray:
    """
    Risk score for Harrell C-index.

    The score is the predicted cumulative hazard evaluated at each row's
    observed duration. Higher score means higher predicted failure risk.
    """
    import bisect

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
    """
    Gap-time failure probability:
        P(event within H years) = 1 - S(H)
    """
    import bisect

    preprocessor = pipe.named_steps["preproc"]
    rsf = pipe.named_steps["rsf"]

    X_transformed = preprocessor.transform(X_eval)
    survival_functions = rsf.predict_survival_function(
        X_transformed,
        return_array=False,
    )

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


def harrell_c_index(
    y_event: np.ndarray,
    y_time: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """
    Harrell C-index using higher risk score = higher failure risk.
    """
    result = concordance_index_censored(
        event_indicator=np.asarray(y_event, dtype=bool),
        event_time=np.asarray(y_time, dtype=float),
        estimate=np.asarray(risk_scores, dtype=float),
    )

    return float(result[0])


def top_20_recall(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    top_fraction: float = 0.20,
) -> Tuple[float, int, int, int]:
    """
    Recall among the highest-risk top fraction of pipes.
    """
    y_true = np.asarray(y_true, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    n = len(risk_scores)
    k = max(1, int(round(top_fraction * n)))

    top_idx = np.argsort(risk_scores)[::-1][:k]

    total_failures = int(y_true.sum())
    captured_failures = int(y_true[top_idx].sum())

    recall = (
        captured_failures / total_failures
        if total_failures > 0
        else 0.0
    )

    return float(recall), int(k), captured_failures, total_failures


# ---------------------------------------------------------------------
# 4. Permutation importance
# ---------------------------------------------------------------------

def grouped_permutation_importance_harrell_c(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    y_event: np.ndarray,
    y_time: np.ndarray,
    feature_names: List[str],
    emb_cols: List[str],
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Permutation importance based on decrease in Harrell C-index.

    Non-embedding variables are permuted individually.
    All emb_* variables are permuted jointly and reported as:
        Embeddings (group)
    """
    rng = np.random.default_rng(random_state)

    base_risk = cumulative_hazard_risk_scores(
        pipe=pipe,
        X_eval=X_eval,
        observed_times=y_time,
    )

    base_c = harrell_c_index(
        y_event=y_event,
        y_time=y_time,
        risk_scores=base_risk,
    )

    non_embedding_features = [
        f for f in feature_names
        if f not in emb_cols
    ]

    records = []

    def score_after_permutation(X_perm: pd.DataFrame) -> float:
        perm_risk = cumulative_hazard_risk_scores(
            pipe=pipe,
            X_eval=X_perm,
            observed_times=y_time,
        )

        return harrell_c_index(
            y_event=y_event,
            y_time=y_time,
            risk_scores=perm_risk,
        )

    # Individual non-embedding variables
    for feature in non_embedding_features:
        deltas = []

        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            perm_idx = rng.permutation(len(X_perm))

            X_perm[feature] = X_perm[feature].to_numpy()[perm_idx]

            perm_c = score_after_permutation(X_perm)
            deltas.append(base_c - perm_c)

        records.append(
            {
                "variable": feature,
                "group": "single",
                "base_harrell_c": base_c,
                "mean_delta_harrell_c": float(np.mean(deltas)),
                "std_delta_harrell_c": float(np.std(deltas, ddof=1)),
                "n_repeats": int(n_repeats),
            }
        )

    # Embeddings as one grouped block
    if emb_cols:
        deltas = []

        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            perm_idx = rng.permutation(len(X_perm))

            X_perm.loc[:, emb_cols] = X_perm[emb_cols].to_numpy()[perm_idx, :]

            perm_c = score_after_permutation(X_perm)
            deltas.append(base_c - perm_c)

        records.append(
            {
                "variable": "Embeddings (group)",
                "group": "embeddings",
                "base_harrell_c": base_c,
                "mean_delta_harrell_c": float(np.mean(deltas)),
                "std_delta_harrell_c": float(np.std(deltas, ddof=1)),
                "n_repeats": int(n_repeats),
            }
        )

    importance = pd.DataFrame(records)

    importance["abs_mean_delta_harrell_c"] = importance[
        "mean_delta_harrell_c"
    ].abs()

    importance = importance.sort_values(
        "abs_mean_delta_harrell_c",
        ascending=False,
    ).reset_index(drop=True)

    return importance


def save_permutation_importance_plot(
    importance: pd.DataFrame,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """
    Save a horizontal bar plot of permutation importance.
    """
    if importance.empty:
        print("Permutation importance table is empty. No plot saved.")
        return

    plot_df = importance.copy()

    # Keep the strongest variables by absolute effect.
    plot_df = plot_df.head(top_n).copy()

    # For readability, plot in ascending order.
    plot_df = plot_df.sort_values("mean_delta_harrell_c", ascending=True)

    pretty_names = {
        "age_start": "Age start",
        "num_prev_failures": "No. previous failures",
        "khop_fail_sum": "K-hop failure score",
        "Road Intersections": "Road intersections",
        "Near Railways": "Near railways",
        "Air Release Valve": "Air release valves",
    }

    plot_df["display_name"] = plot_df["variable"].replace(pretty_names)

    y_pos = np.arange(len(plot_df))

    ci = 1.96 * plot_df["std_delta_harrell_c"] / np.sqrt(
        plot_df["n_repeats"]
    )

    plt.figure(figsize=(9, max(5, 0.45 * len(plot_df))))
    plt.barh(
        y_pos,
        plot_df["mean_delta_harrell_c"],
        xerr=ci,
        align="center",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )

    plt.yticks(y_pos, plot_df["display_name"])
    plt.xlabel("Decrease in Harrell C-index after permutation")
    plt.title("Permutation-based variable importance")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# 5. Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    print("Training synthetic RSF model with minimal evaluation.")
    print(f"Training file:   {TRAIN_CSV}")
    print(f"Evaluation file: {EVAL_CSV}")
    print(f"Output folder:   {OUT_DIR}")

    df_train = read_input_csv(TRAIN_CSV)
    df_eval = read_input_csv(EVAL_CSV)

    required_cols = [
        ID_COL,
        "duration",
        "event",
    ]

    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Training file missing required column: {col}")
        if col not in df_eval.columns:
            raise ValueError(f"Evaluation file missing required column: {col}")

    df_train[ID_COL] = df_train[ID_COL].astype(str).str.strip()
    df_eval[ID_COL] = df_eval[ID_COL].astype(str).str.strip()

    df_train["duration"] = pd.to_numeric(
        df_train["duration"],
        errors="coerce",
    )
    df_eval["duration"] = pd.to_numeric(
        df_eval["duration"],
        errors="coerce",
    )

    df_train["event"] = pd.to_numeric(
        df_train["event"],
        errors="coerce",
    ).fillna(0).astype(int)

    df_eval["event"] = pd.to_numeric(
        df_eval["event"],
        errors="coerce",
    ).fillna(0).astype(int)

    df_train = df_train.dropna(subset=["duration"]).copy()
    df_eval = df_eval.dropna(subset=["duration"]).copy()

    if (df_train["duration"] <= 0).any():
        raise ValueError("Training data contains non-positive duration values.")

    if (df_eval["duration"] <= 0).any():
        raise ValueError("Evaluation data contains non-positive duration values.")

    trainval_ids, test_ids = make_or_load_pipe_holdout_split(
        df_train=df_train,
        df_eval=df_eval,
        seed=SEED,
        holdout_frac=HOLDOUT_TEST_FRAC,
    )
    
    df_train = df_train[df_train[ID_COL].astype(str).isin(trainval_ids)].copy()
    df_eval = df_eval[df_eval[ID_COL].astype(str).isin(test_ids)].copy()
    
    print("\nPipe-level holdout split:")
    print(f"Trainval pipes: {len(trainval_ids)}")
    print(f"Held-out test pipes: {len(test_ids)}")
    print(f"Training rows after split: {len(df_train)}")
    print(f"Evaluation rows after split: {len(df_eval)}")
    print(f"Evaluation events after split: {int(df_eval['event'].sum())}")
    
    overlap = set(df_train[ID_COL].astype(str)).intersection(
    set(df_eval[ID_COL].astype(str))
    )
    
    if overlap:
        raise RuntimeError(
            f"Train/evaluation pipe overlap detected: {len(overlap)} pipes."
    )

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
    
    print("\nFitting final RSF model on all training intervals...")
    pipe.fit(X_train, y_train, rsf__sample_weight=weights)

    print("Predicting evaluation risk...")
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

    top20_recall, top20_k, top20_caught, total_failures = top_20_recall(
        y_true=y_eval_binary,
        risk_scores=failure_prob_5yr,
        top_fraction=TOP_FRACTION,
    )

    print("\nEvaluation results:")
    print(f"Harrell C-index: {harrell_c:.4f}")
    print(
        f"Top 20% recall: {top20_recall:.4f} "
        f"({top20_caught}/{total_failures} failures captured in top {top20_k} pipes)"
    )

    predictions = df_eval[
        [
            ID_COL,
            "start_year",
            "end_year",
            "duration",
            "event",
        ]
    ].copy()

    predictions["risk_score_cumhaz"] = risk_scores
    predictions["failure_prob_5yr"] = failure_prob_5yr
    predictions["rank_by_failure_prob"] = predictions[
        "failure_prob_5yr"
    ].rank(ascending=False, method="first").astype(int)

    predictions["in_top20_percent"] = (
        predictions["rank_by_failure_prob"] <= top20_k
    ).astype(int)

    predictions.to_csv(
        OUT_DIR / "test_predictions.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
            "harrell_c_index": harrell_c,
            "top20_recall": top20_recall,
            "top20_k": top20_k,
            "top20_caught_failures": top20_caught,
            "total_failures": total_failures,
            "best_rsf_params": json.dumps(best_params),
            "hyperparameter_tuning_used": USE_HYPERPARAMETER_TUNING,
            "n_train_rows": len(df_train),
            "n_train_pipes": df_train[ID_COL].nunique(),
            "n_train_events": int(df_train["event"].sum()),
            "n_eval_rows": len(df_eval),
            "n_eval_pipes": df_eval[ID_COL].nunique(),
            "n_eval_events": int(df_eval["event"].sum()),
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

    print("\nComputing permutation importance...")
    importance = grouped_permutation_importance_harrell_c(
        pipe=pipe,
        X_eval=X_eval,
        y_event=y_eval_event,
        y_time=y_eval_time,
        feature_names=feature_cols,
        emb_cols=emb_cols,
        n_repeats=N_PERM_REPEATS,
        random_state=SEED,
    )

    importance.to_csv(
        OUT_DIR / "permutation_importance_harrell_c.csv",
        index=False,
    )

    save_permutation_importance_plot(
        importance=importance,
        out_path=OUT_DIR / "permutation_importance_harrell_c.png",
        top_n=15,
    )

    print("\nFiles saved:")
    print(f"Predictions:            {OUT_DIR / 'test_predictions.csv'}")
    print(f"Summary metrics:        {OUT_DIR / 'summary_metrics.csv'}")
    print(f"Permutation importance: {OUT_DIR / 'permutation_importance_harrell_c.csv'}")
    print(f"Importance plot:        {OUT_DIR / 'permutation_importance_harrell_c.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()

