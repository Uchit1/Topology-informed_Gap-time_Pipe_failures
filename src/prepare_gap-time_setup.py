"""
Prepare gap-time survival data for the synthetic Net3 dataset.

Inputs:
    data_synthetic/Synthetic_assets_and_failures_from_Net3.csv
    outputs/aligned_graph_features/snapYYYY_linegraph_node2vec_*.csv

Outputs:
    outputs/survival_data/gap_time_train_intervals_2007_2019_with_embeddings.csv
    outputs/survival_data/eval_window_2020_2024_with_embeddings.csv

Purpose:
    Convert the synthetic pipe-level yearly failure table into recurrent
    gap-time survival intervals for Random Survival Forest modelling.

Final RSF-ready output columns:
    pipe_id
    start_year
    end_year
    duration
    event
    Length
    Diameter
    Soil
    Material
    Elevation
    Landuse
    Road Intersections
    Near Railways
    GWT
    Service Connections
    Fire Hydrants
    Outlets
    Air Release Valve
    age_start
    num_prev_failures
    emb_00 ... emb_63
    khop_fail_sum
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data_synthetic"
EMB_DIR = ROOT_DIR / "outputs" / "aligned_graph_features"
OUT_DIR = ROOT_DIR / "outputs" / "survival_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_CANDIDATES = [
    DATA_DIR / "Synthetic_assets_and_failures_from_Net3.csv",
    DATA_DIR / "Synthetic_assets_and_failures_data_from_Net3.csv",
    DATA_DIR / "Synthetic_assets_and_failures_data_fromNet3.csv",
]

TRAIN_START = 2007
TRAIN_END = 2019

TEST_START = 2020
TEST_END = 2024

HISTORY_START_YEAR = TRAIN_START

MIN_INSTALL_YEAR = 1940

ID_CANDIDATES = ["EntityID", "pipe_id", "id"]
INSTALL_YEAR_COL = "Installation Year"

YEAR_COLUMNS = [str(y) for y in range(TRAIN_START, TEST_END + 1)]

STATIC_COLS = [
    "Length",
    "Diameter",
    "Soil",
    "Material",
    "Elevation",
    "Landuse",
    "Road Intersections",
    "Near Railways",
    "GWT",
    "Service Connections",
    "Fire Hydrants",
    "Outlets",
    "Air Release Valve",
]

TRAIN_OUT = OUT_DIR / "gap_time_train_intervals_2007_2019_with_embeddings.csv"
TEST_OUT = OUT_DIR / "eval_window_2020_2024_with_embeddings.csv"


# ---------------------------------------------------------------------
# 2. CSV and ID helpers
# ---------------------------------------------------------------------

def read_smart_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader for comma, semicolon, or tab-separated files.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        pass

    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="ISO-8859-1")
    except Exception:
        pass

    for enc in ("utf-8-sig", "ISO-8859-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, sep=sep, engine="python", encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue

    raise ValueError(f"Could not parse CSV: {path}")


def find_synthetic_csv() -> Path:
    """
    Find the synthetic asset/failure CSV.
    First tries known filenames, then searches data_synthetic for Net3 CSV files.
    """
    for path in CSV_CANDIDATES:
        if path.exists():
            return path

    candidates = sorted(DATA_DIR.glob("*Net3*.csv"))

    if candidates:
        print(f"Using detected synthetic CSV: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "Could not find the synthetic asset/failure CSV.\n\n"
        f"Searched folder:\n{DATA_DIR}\n\n"
        "Expected a CSV file with 'Net3' in the filename."
    )


def pick_id_col(df: pd.DataFrame) -> str:
    """
    Detect pipe identifier column.
    """
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col

    raise KeyError(
        f"No suitable pipe ID column found. Expected one of: {ID_CANDIDATES}"
    )


def normalize_id_series(series: pd.Series) -> pd.Series:
    """
    Normalize pipe IDs for matching CSV and embedding files.
    """
    out = series.copy()

    if pd.api.types.is_float_dtype(out):
        as_float = out.dropna().astype(float)
        if len(as_float) and np.isclose(as_float, np.floor(as_float)).all():
            out = out.astype("Int64")

    return (
        out.astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )


# ---------------------------------------------------------------------
# 3. Clean and validate data
# ---------------------------------------------------------------------

def clean_synthetic_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Clean synthetic pipe-level data and validate temporal consistency.

    This function raises an error if any pipe has a failure before its
    installation year.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    id_col = pick_id_col(df)

    df[id_col] = normalize_id_series(df[id_col])

    if df[id_col].duplicated().any():
        duplicated = df.loc[df[id_col].duplicated(), id_col].tolist()
        raise ValueError(
            f"Pipe IDs must be unique. Example duplicates: {duplicated[:10]}"
        )

    required_cols = [id_col, INSTALL_YEAR_COL] + STATIC_COLS + YEAR_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in synthetic CSV: {missing}")

    numeric_cols = [
        INSTALL_YEAR_COL,
        "Length",
        "Diameter",
        "Elevation",
        "GWT",
        "Service Connections",
        "Fire Hydrants",
        "Outlets",
        "Air Release Valve",
    ] + YEAR_COLUMNS

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    essential_cols = [
        INSTALL_YEAR_COL,
        "Length",
        "Diameter",
        "Soil",
        "Material",
    ]

    n_before = len(df)
    df = df.dropna(subset=essential_cols).copy()
    n_after = len(df)

    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows with missing essential values.")

    df = df[df[INSTALL_YEAR_COL].astype(int) >= int(MIN_INSTALL_YEAR)].copy()

    df["Year"] = df[INSTALL_YEAR_COL].astype(int)

    invalid_records = []

    for _, row in df.iterrows():
        install_year = int(row["Year"])

        for yc in YEAR_COLUMNS:
            failure_year = int(yc)
            value = row.get(yc, 0)

            if pd.notna(value) and int(value) > 0 and failure_year < install_year:
                invalid_records.append(
                    {
                        "pipe_id": row[id_col],
                        "installation_year": install_year,
                        "failure_year": failure_year,
                    }
                )

    if invalid_records:
        invalid_df = pd.DataFrame(invalid_records)
        raise ValueError(
            "Temporal inconsistency detected: at least one pipe has a failure "
            "before its installation year.\n\n"
            f"Examples:\n{invalid_df.head(20)}"
        )

    print("Temporal consistency check passed: no failures before installation year.")

    return df, id_col


# ---------------------------------------------------------------------
# 4. Gap-time interval builders
# ---------------------------------------------------------------------

def build_training_intervals(
    df: pd.DataFrame,
    start_min_year: int,
    end_year: int,
    id_col: str,
    static_cols: List[str],
) -> pd.DataFrame:
    """
    Build recurrent gap-time training intervals.

    Each pipe can contribute:
        - one interval ending at each observed failure
        - one final censored interval if it remains under observation
    """
    out = []

    for _, row in df.iterrows():
        install_year = int(row["Year"])
        pipe_id = str(row[id_col]).strip()

        prev_year = max(start_min_year, install_year)

        failure_count = 0
        last_failure_year = None

        for y in range(prev_year, end_year + 1):
            val = row.get(str(y), 0)
            failure = 1 if pd.notna(val) and int(val) > 0 else 0

            if failure:
                out.append(
                    {
                        id_col: pipe_id,
                        "start_year": prev_year,
                        "end_year": y,
                        "duration": (y - prev_year) + 1,
                        "event": 1,
                        "age_start": prev_year - install_year,
                        "num_prev_failures": failure_count,
                        **{c: row[c] for c in static_cols},
                    }
                )

                failure_count += 1
                last_failure_year = y
                prev_year = y + 1

        if prev_year <= end_year:
            out.append(
                {
                    id_col: pipe_id,
                    "start_year": prev_year,
                    "end_year": end_year,
                    "duration": (end_year - prev_year) + 1,
                    "event": 0,
                    "age_start": prev_year - install_year,
                    "num_prev_failures": failure_count,
                    **{c: row[c] for c in static_cols},
                }
            )

    intervals = pd.DataFrame(out)

    if intervals.empty:
        raise ValueError("No training intervals were created.")

    return intervals


def build_eval_window(
    df: pd.DataFrame,
    window_start: int,
    window_end: int,
    id_col: str,
    static_cols: List[str],
    history_start_year: int,
) -> pd.DataFrame:
    """
    Build one evaluation interval per eligible pipe for a fixed prediction window.

    event = 1 if the pipe has at least one failure during the window.
    duration ends at first failure year if event = 1, otherwise at window_end.
    """
    out = []

    for _, row in df.iterrows():
        install_year = int(row["Year"])
        pipe_id = str(row[id_col]).strip()

        if install_year > window_start:
            continue

        failure_count = 0

        hist_start = max(history_start_year, install_year)

        for y in range(hist_start, window_start):
            val = row.get(str(y), 0)

            if pd.notna(val) and int(val) > 0:
                failure_count += 1

        first_failure = None

        for y in range(window_start, window_end + 1):
            val = row.get(str(y), 0)

            if pd.notna(val) and int(val) > 0:
                first_failure = y
                break

        if first_failure is not None:
            end_year = first_failure
            event = 1
        else:
            end_year = window_end
            event = 0

        duration = (end_year - window_start) + 1

        if duration > 0:
            out.append(
                {
                    id_col: pipe_id,
                    "start_year": window_start,
                    "end_year": end_year,
                    "duration": duration,
                    "event": event,
                    "age_start": window_start - install_year,
                    "num_prev_failures": failure_count,
                    **{c: row[c] for c in static_cols},
                }
            )

    intervals = pd.DataFrame(out)

    if intervals.empty:
        raise ValueError("No evaluation-window intervals were created.")

    return intervals


# ---------------------------------------------------------------------
# 5. Embedding attachment helpers
# ---------------------------------------------------------------------

def embedding_path_for_year(year: int) -> Path:
    """
    Find aligned embedding file for a given snapshot year.
    """
    files = sorted(EMB_DIR.glob(f"snap{year}_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No aligned embedding file found for snapshot {year} in:\n{EMB_DIR}"
        )

    if len(files) > 1:
        print(
            f"Warning: multiple embedding files found for snapshot {year}. "
            f"Using: {files[0].name}"
        )

    return files[0]


def available_embedding_years() -> List[int]:
    """
    Detect available embedding years from aligned embedding folder.
    """
    if not EMB_DIR.exists():
        raise FileNotFoundError(f"Aligned embedding folder not found:\n{EMB_DIR}")

    years = []

    for path in EMB_DIR.glob("snap*.csv"):
        match = re.match(r"^snap(\d{4})_", path.name)
        if match:
            years.append(int(match.group(1)))

    years = sorted(set(years))

    if not years:
        raise FileNotFoundError(f"No snap*.csv files found in:\n{EMB_DIR}")

    return years


def choose_snapshot_year(start_year: int, emb_years: List[int]) -> int:
    """
    Time-correct embedding rule:
        use the latest embedding snapshot year <= interval start_year.
    """
    available = [y for y in emb_years if y <= int(start_year)]

    if available:
        return max(available)

    return min(emb_years)


def detect_embedding_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect embedding columns sorted as emb_00, emb_01, ...
    """
    emb_cols = [c for c in df.columns if re.match(r"^emb_\d+$", str(c))]

    if not emb_cols:
        raise ValueError("No embedding columns found. Expected emb_00, emb_01, ...")

    def emb_index(col):
        match = re.search(r"emb_(\d+)", str(col))
        return int(match.group(1)) if match else 999999

    return sorted(emb_cols, key=emb_index)


def load_embedding_features_for_year(
    year: int,
    id_col_main: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load aligned embedding file for one snapshot year.

    Keeps only:
        id_col_main
        khop_fail_sum
        emb_*

    This avoids duplicate ID columns when the embedding file contains both
    pipe_id and id.
    """
    path = embedding_path_for_year(year)
    emb = read_smart_csv(path)
    emb.columns = emb.columns.str.strip()

    if "pipe_id" in emb.columns:
        emb_key = "pipe_id"
    elif "EntityID" in emb.columns:
        emb_key = "EntityID"
    elif "id" in emb.columns:
        emb_key = "id"
    else:
        raise KeyError(
            f"No suitable ID column found in embedding file: {path.name}. "
            "Expected one of: pipe_id, EntityID, id."
        )

    emb[emb_key] = normalize_id_series(emb[emb_key])

    emb_cols = detect_embedding_columns(emb)

    if "khop_fail_sum" not in emb.columns:
        raise ValueError(f"'khop_fail_sum' missing from embedding file: {path.name}")

    emb_clean = emb[[emb_key, "khop_fail_sum"] + emb_cols].copy()
    emb_clean = emb_clean.rename(columns={emb_key: id_col_main})
    emb_clean = emb_clean.loc[:, ~emb_clean.columns.duplicated()].copy()

    emb_clean[id_col_main] = normalize_id_series(emb_clean[id_col_main])

    if emb_clean[id_col_main].duplicated().any():
        duplicated = emb_clean.loc[
            emb_clean[id_col_main].duplicated(),
            id_col_main
        ].tolist()

        raise ValueError(
            f"Duplicated pipe IDs in embedding file {path.name}. "
            f"Example duplicates: {duplicated[:10]}"
        )

    for c in ["khop_fail_sum"] + emb_cols:
        emb_clean[c] = pd.to_numeric(emb_clean[c], errors="coerce")

    return emb_clean, ["khop_fail_sum"] + emb_cols


def attach_embeddings_by_start_year(
    intervals: pd.DataFrame,
    id_col: str,
) -> Tuple[pd.DataFrame, Dict[int, float], List[str]]:
    """
    Attach aligned embeddings to interval rows using:
        embedding year = latest snapshot <= start_year
    """
    df = intervals.copy()
    df[id_col] = normalize_id_series(df[id_col])

    emb_years = available_embedding_years()

    df["embedding_snapshot_year"] = df["start_year"].astype(int).map(
        lambda y: choose_snapshot_year(y, emb_years)
    )

    cache: Dict[int, pd.DataFrame] = {}
    feature_union: List[str] = []
    match_rates: Dict[int, float] = {}

    parts = []

    for emb_year in sorted(df["embedding_snapshot_year"].unique()):
        sub = df[df["embedding_snapshot_year"] == emb_year].copy()

        if emb_year not in cache:
            emb_df, feature_cols = load_embedding_features_for_year(
                year=int(emb_year),
                id_col_main=id_col,
            )
            cache[int(emb_year)] = emb_df
            feature_union = sorted(set(feature_union).union(feature_cols))

        emb_df = cache[int(emb_year)]

        sub = sub.merge(
            emb_df,
            on=id_col,
            how="left",
            indicator=True,
            validate="many_to_one",
        )

        match_rates[int(emb_year)] = float((sub["_merge"] == "both").mean())
        sub = sub.drop(columns=["_merge"])

        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)

    emb_cols = [c for c in out.columns if re.match(r"^emb_\d+$", str(c))]

    missing_rows = out[emb_cols].isna().all(axis=1).sum()

    if missing_rows > 0:
        raise ValueError(
            f"{missing_rows} interval rows did not match any embedding row. "
            "Check pipe IDs between the synthetic CSV and aligned embeddings."
        )

    return out, match_rates, feature_union


# ---------------------------------------------------------------------
# 6. Final output column selection
# ---------------------------------------------------------------------

def make_final_rsf_table(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Keep only the columns required for RSF training/evaluation.
    """
    out = df.copy()

    if id_col != "pipe_id":
        out = out.rename(columns={id_col: "pipe_id"})

    emb_cols = [c for c in out.columns if re.match(r"^emb_\d+$", str(c))]
    emb_cols = sorted(emb_cols, key=lambda c: int(str(c).split("_")[1]))

    required_output_cols = [
        "pipe_id",
        "start_year",
        "end_year",
        "duration",
        "event",
        "Length",
        "Diameter",
        "Soil",
        "Material",
        "Elevation",
        "Landuse",
        "Road Intersections",
        "Near Railways",
        "GWT",
        "Service Connections",
        "Fire Hydrants",
        "Outlets",
        "Air Release Valve",
        "age_start",
        "num_prev_failures",
    ] + emb_cols + [
        "khop_fail_sum",
    ]

    missing_cols = [c for c in required_output_cols if c not in out.columns]

    if missing_cols:
        raise ValueError(f"Missing required RSF output columns: {missing_cols}")

    return out[required_output_cols].copy()


# ---------------------------------------------------------------------
# 7. Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    csv_path = find_synthetic_csv()

    print("Preparing synthetic gap-time survival data.")
    print(f"Synthetic CSV: {csv_path}")
    print(f"Aligned embeddings: {EMB_DIR}")
    print(f"Output folder: {OUT_DIR}")

    df = read_smart_csv(csv_path)
    df, id_col = clean_synthetic_data(df)

    print(f"Using ID column: {id_col}")
    print(f"Number of cleaned pipes: {len(df)}")

    train_intervals = build_training_intervals(
        df=df,
        start_min_year=TRAIN_START,
        end_year=TRAIN_END,
        id_col=id_col,
        static_cols=STATIC_COLS,
    )

    test_intervals = build_eval_window(
        df=df,
        window_start=TEST_START,
        window_end=TEST_END,
        id_col=id_col,
        static_cols=STATIC_COLS,
        history_start_year=HISTORY_START_YEAR,
    )

    train_with_emb, train_match_rates, _ = attach_embeddings_by_start_year(
        intervals=train_intervals,
        id_col=id_col,
    )

    test_with_emb, test_match_rates, _ = attach_embeddings_by_start_year(
        intervals=test_intervals,
        id_col=id_col,
    )

    train_final = make_final_rsf_table(train_with_emb, id_col=id_col)
    test_final = make_final_rsf_table(test_with_emb, id_col=id_col)

    train_final.to_csv(TRAIN_OUT, index=False)
    test_final.to_csv(TEST_OUT, index=False)

    emb_cols = [c for c in train_final.columns if re.match(r"^emb_\d+$", str(c))]

    print("\nGap-time survival data created successfully.")
    print(f"Training file: {TRAIN_OUT}")
    print(f"Evaluation file: {TEST_OUT}")

    print("\nTraining intervals:")
    print(f"Rows: {len(train_final)}")
    print(f"Events: {int(train_final['event'].sum())}")
    print(f"Censored intervals: {int((train_final['event'] == 0).sum())}")
    print(f"Unique pipes: {train_final['pipe_id'].nunique()}")
    print(f"Embedding match rates: {train_match_rates}")

    print("\nEvaluation window:")
    print(f"Rows: {len(test_final)}")
    print(f"Events: {int(test_final['event'].sum())}")
    print(f"Censored intervals: {int((test_final['event'] == 0).sum())}")
    print(f"Unique pipes: {test_final['pipe_id'].nunique()}")
    print(f"Embedding match rates: {test_match_rates}")

    print("\nFinal RSF-ready columns:")
    print(f"Total columns: {len(train_final.columns)}")
    print(f"Embedding columns: {len(emb_cols)}")
    print("Graph scalar: khop_fail_sum")

    preview_cols = [
        "pipe_id",
        "start_year",
        "end_year",
        "duration",
        "event",
        "Length",
        "Diameter",
        "Soil",
        "Material",
        "age_start",
        "num_prev_failures",
        "khop_fail_sum",
    ] + emb_cols[:5]

    print("\nPreview:")
    print(train_final[preview_cols].head())


if __name__ == "__main__":
    main()