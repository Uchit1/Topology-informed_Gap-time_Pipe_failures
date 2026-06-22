"""
Align yearly graph embeddings for the synthetic Net3 dataset.

Input:
    outputs/graph_features/
        snap2007_linegraph_node2vec_....csv
        snap2008_linegraph_node2vec_....csv
        ...
        snap2019_linegraph_node2vec_....csv

Output:
    outputs/aligned_graph_features/
        aligned versions of the same files

Purpose:
    Yearly node2vec/Word2Vec embeddings can differ by arbitrary rotation or
    reflection between snapshot years. This script aligns all yearly embedding
    files to the 2019 reference embedding space using orthogonal Procrustes
    alignment.

Notes:
    - Only emb_* columns are transformed.
    - pipe_id, id, and khop_fail_sum are preserved.
    - The 2019 reference file is copied unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT_DIR / "outputs" / "graph_features"
OUT_DIR = ROOT_DIR / "outputs" / "aligned_graph_features"

REF_YEAR = 2019

# Net3 is a small example network, so not using a very high minimum anchor threshold.
MIN_ANCHORS = 20

# Centering to improve Procrustes stability.
CENTER = True

# False allows rotation/reflection.
ENFORCE_PROPER_ROTATION = False


# ---------------------------------------------------------------------
# 2. Utility functions
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


def detect_id_col(df: pd.DataFrame) -> str:
    """
    Detect pipe identifier column.
    """
    df.columns = df.columns.str.strip()

    for col in ("pipe_id", "EntityID", "id", "OBJECTID"):
        if col in df.columns:
            return col

    raise ValueError(
        "No pipe ID column found. Expected one of: pipe_id, EntityID, id, OBJECTID."
    )


def detect_emb_cols(df: pd.DataFrame) -> list[str]:
    """
    Detect embedding columns and sort them numerically.
    """
    emb_cols = [col for col in df.columns if str(col).startswith("emb_")]

    if not emb_cols:
        raise ValueError("No embedding columns found. Expected columns starting with 'emb_'.")

    def emb_index(col):
        match = re.search(r"emb_(\d+)", str(col))
        return int(match.group(1)) if match else 999999

    return sorted(emb_cols, key=emb_index)


def filename_year(path: Path) -> int | None:
    """
    Extract snapshot year from filename.

    Example:
        snap2007_linegraph_node2vec_p1_q2_...csv -> 2007
    """
    match = re.match(r"^snap(\d{4})_", path.stem)

    if match:
        return int(match.group(1))

    return None


def filename_tag(path: Path) -> str:
    """
    Group key for alignment.

    This removes the leading snapYYYY_ part so files with the same embedding
    parameters are grouped together.

    Example:
        snap2007_linegraph_node2vec_p1_q2_... -> linegraph_node2vec_p1_q2_...
    """
    match = re.match(r"^snap(\d{4})_(.+)$", path.stem)

    if match:
        return match.group(2)

    return path.stem


def cosine_sim_rows(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise cosine similarity.
    """
    numerator = np.sum(A * B, axis=1)
    denominator = (np.linalg.norm(A, axis=1) + eps) * (np.linalg.norm(B, axis=1) + eps)

    return numerator / denominator


def orthogonal_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    enforce_proper_rotation: bool = False
) -> np.ndarray:
    """
    Find orthogonal matrix R minimizing ||X R - Y||_F.

    X:
        Source embedding matrix for anchor pipes.
    Y:
        Reference embedding matrix for the same anchor pipes.
    """
    matrix = X.T @ Y
    U, _, Vt = np.linalg.svd(matrix, full_matrices=False)
    R = U @ Vt

    if enforce_proper_rotation and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def align_one_to_reference(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    id_col: str,
    emb_cols: list[str],
    center: bool = True,
    enforce_proper_rotation: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Align one snapshot embedding file to the reference-year embedding file.

    Returns:
        aligned dataframe
        alignment statistics
    """
    source = source_df.copy()
    reference = reference_df.copy()

    source.columns = source.columns.str.strip()
    reference.columns = reference.columns.str.strip()

    source[id_col] = source[id_col].astype(str).str.strip()
    reference[id_col] = reference[id_col].astype(str).str.strip()

    # Convert embeddings to numeric.
    for col in emb_cols:
        source[col] = pd.to_numeric(source[col], errors="coerce")
        reference[col] = pd.to_numeric(reference[col], errors="coerce")

    # Fill missing embedding values with column means.
    source[emb_cols] = source[emb_cols].fillna(source[emb_cols].mean(axis=0))
    reference[emb_cols] = reference[emb_cols].fillna(reference[emb_cols].mean(axis=0))

    # Anchor pipes are pipes present in both source and reference snapshots.
    merged = source[[id_col] + emb_cols].merge(
        reference[[id_col] + emb_cols],
        on=id_col,
        how="inner",
        suffixes=("_src", "_ref")
    )

    n_anchors = len(merged)

    if n_anchors < MIN_ANCHORS:
        raise ValueError(
            f"Too few common anchor pipes for alignment: {n_anchors}. "
            f"Minimum required: {MIN_ANCHORS}."
        )

    X = merged[[col + "_src" for col in emb_cols]].to_numpy(float)
    Y = merged[[col + "_ref" for col in emb_cols]].to_numpy(float)

    if center:
        X_mean = X.mean(axis=0, keepdims=True)
        Y_mean = Y.mean(axis=0, keepdims=True)

        X_centered = X - X_mean
        Y_centered = Y - Y_mean
    else:
        X_mean = np.zeros((1, X.shape[1]))
        Y_mean = np.zeros((1, Y.shape[1]))

        X_centered = X
        Y_centered = Y

    cos_before = float(np.mean(cosine_sim_rows(X, Y)))

    R = orthogonal_procrustes(
        X=X_centered,
        Y=Y_centered,
        enforce_proper_rotation=enforce_proper_rotation
    )

    # Apply alignment to all source rows, not only anchors.
    E = source[emb_cols].to_numpy(float)

    if center:
        E_aligned = (E - X_mean) @ R + Y_mean
        X_aligned = (X - X_mean) @ R + Y_mean
    else:
        E_aligned = E @ R
        X_aligned = X @ R

    source.loc[:, emb_cols] = E_aligned

    cos_after = float(np.mean(cosine_sim_rows(X_aligned, Y)))

    stats = {
        "n_anchors": int(n_anchors),
        "cos_before": cos_before,
        "cos_after": cos_after,
    }

    return source, stats


# ---------------------------------------------------------------------
# 3. Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Input embedding folder not found: {SRC_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC_DIR.glob("snap*.csv"))

    if not files:
        raise FileNotFoundError(f"No snap*.csv files found in: {SRC_DIR}")

    # Group by parameter tag, excluding the snapshot year.
    groups: dict[str, list[Path]] = {}

    for file in files:
        year = filename_year(file)

        if year is None:
            continue

        tag = filename_tag(file)
        groups.setdefault(tag, []).append(file)

    print("Aligning yearly graph embeddings.")
    print(f"Input folder:  {SRC_DIR}")
    print(f"Output folder: {OUT_DIR}")
    print(f"Reference year: {REF_YEAR}")
    print(f"Found files: {len(files)}")
    print(f"Found parameter groups: {len(groups)}")

    written_files = []

    for tag, file_list in groups.items():
        file_list = sorted(file_list, key=lambda p: filename_year(p) or 0)

        reference_file = None

        for file in file_list:
            if filename_year(file) == REF_YEAR:
                reference_file = file
                break

        if reference_file is None:
            print(f"[skip] No snap{REF_YEAR} reference file for parameter group: {tag}")
            continue

        reference_df = read_smart_csv(reference_file)
        reference_df.columns = reference_df.columns.str.strip()

        id_col = detect_id_col(reference_df)
        emb_cols = detect_emb_cols(reference_df)

        reference_df[id_col] = reference_df[id_col].astype(str).str.strip()

        if reference_df[id_col].duplicated().any():
            raise ValueError(f"Reference file contains duplicated pipe IDs: {reference_file.name}")

        print(f"\n[group] {tag}")
        print(f"Reference file: {reference_file.name}")
        print(f"Embedding dimension: {len(emb_cols)}")

        for file in file_list:
            out_path = OUT_DIR / file.name

            source_df = read_smart_csv(file)
            source_df.columns = source_df.columns.str.strip()

            source_id_col = detect_id_col(source_df)

            if source_id_col != id_col:
                source_df = source_df.rename(columns={source_id_col: id_col})

            missing_emb_cols = [col for col in emb_cols if col not in source_df.columns]

            if missing_emb_cols:
                raise ValueError(
                    f"{file.name} is missing embedding columns. "
                    f"First missing columns: {missing_emb_cols[:5]}"
                )

            if filename_year(file) == REF_YEAR:
                # Reference file is copied without transformation.
                source_df.to_csv(out_path, index=False)
                written_files.append(out_path)
                print(f"  [copy]  {file.name}")
                continue

            aligned_df, stats = align_one_to_reference(
                source_df=source_df,
                reference_df=reference_df,
                id_col=id_col,
                emb_cols=emb_cols,
                center=CENTER,
                enforce_proper_rotation=ENFORCE_PROPER_ROTATION
            )

            aligned_df.to_csv(out_path, index=False)
            written_files.append(out_path)

            print(
                f"  [align] {file.name} | "
                f"anchors={stats['n_anchors']} | "
                f"cos={stats['cos_before']:.4f} -> {stats['cos_after']:.4f}"
            )

    print("\nFinished alignment.")
    print(f"Number of written files: {len(written_files)}")
    print(f"Aligned files saved in: {OUT_DIR}")

    if len(written_files) == 0:
        raise RuntimeError("No aligned files were written. Check filenames and reference year.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        raise