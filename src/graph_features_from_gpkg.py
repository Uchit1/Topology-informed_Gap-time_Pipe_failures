"""
Build yearly topology-informed pipe embeddings from the synthetic Net3 GIS dataset.

Input:
    data_synthetic/Net3.gpkg
        layer: "pipes"

Expected pipe layer columns:
    EntityID
    geometry
    ConstructionYear
    InnerDiameter
    PipeMaterial
    yearly failure columns: 2007, 2008, ..., 2024

Outputs:
    outputs/graph_features/snap2007_....csv
    outputs/graph_features/snap2008_....csv
    ...
    outputs/graph_features/snap2019_....csv

Each output contains only:
    pipe_id
    id
    emb_00 ... emb_63
    khop_fail_sum

This script is designed for the synthetic Net3-based example dataset.
"""

from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from gensim.models import Word2Vec


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_GPKG = ROOT_DIR / "data_synthetic" / "Net3.gpkg"
LAYER_NAME = "pipes"

OUT_DIR = ROOT_DIR / "outputs" / "graph_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Snapshot years required by the RSF survival workflow
SNAPSHOT_YEARS = list(range(2007, 2020))  

# Geometry processing
PROJECT_TO_EPSG = 3006
ROUND_NDP = 1

# Embedding parameters
SEED = 42
WALKS_PER_NODE = 40
WALK_LEN = 10
P_RET = 1.0
Q_INOUT = 2.0
EMB_DIM = 64
WINDOW = 5
EPOCHS = 5
NEGATIVE = 5
WORKERS = 1
ALPHA_ATTR = 0.0


# Column names in the synthetic GPKG
ID_COL = "EntityID"
YEAR_COL = "ConstructionYear"
DIAM_COL = "InnerDiameter"
MAT_COL = "PipeMaterial"

# k-hop failure density
KHOP_RADIUS = 2
KHOP_SECOND_HOP_WEIGHT = 0.5


# ---------------------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def endpoints(geom):
    """Return start and end points of a LineString or longest MultiLineString."""
    if geom is None or geom.is_empty:
        return None, None

    if isinstance(geom, LineString):
        line = geom
    elif isinstance(geom, MultiLineString):
        line = max(list(geom.geoms), key=lambda x: x.length)
    else:
        return None, None

    return Point(line.coords[0]), Point(line.coords[-1])


def find_failure_year_columns(gdf: gpd.GeoDataFrame, max_year: int):
    """
    Detect yearly failure columns.

    Accepts:
        2007, 2008, ...
    or:
        yearlyfailure_2007, yearlyfailure_2008, ...
    """
    year_map = {}

    for col in gdf.columns:
        col_str = str(col).strip()

        if col_str.isdigit() and len(col_str) == 4:
            year = int(col_str)
            if year <= max_year:
                year_map[col] = year

        elif col_str.startswith("yearlyfailure_"):
            match = re.search(r"(\d{4})$", col_str)
            if match:
                year = int(match.group(1))
                if year <= max_year:
                    year_map[col] = year

    return year_map


def build_failure_table_from_wide(gdf: gpd.GeoDataFrame, max_year: int) -> pd.DataFrame:
    """
    Convert wide yearly failure columns to long format.

    Returns:
        pipe_id, year, count
    """
    year_map = find_failure_year_columns(gdf, max_year=max_year)

    if not year_map:
        return pd.DataFrame(columns=["pipe_id", "year", "count"])

    sub = gdf[list(year_map.keys())].copy()
    sub["pipe_id"] = gdf.index.astype(str)

    long = sub.melt(
        id_vars=["pipe_id"],
        value_vars=list(year_map.keys()),
        var_name="year_col",
        value_name="count"
    )

    long["year"] = long["year_col"].map(year_map).astype(int)
    long["count"] = pd.to_numeric(long["count"], errors="coerce").fillna(0).clip(lower=0)

    out = (
        long.groupby(["pipe_id", "year"], as_index=False)["count"]
        .sum()
        .sort_values(["pipe_id", "year"])
    )

    return out


def w_edge(graph, u, v, default=1.0):
    """Return edge weight between two nodes."""
    data = graph.get_edge_data(u, v)

    if not data:
        return default

    if graph.is_multigraph():
        return max(edge_data.get("w", default) for edge_data in data.values())

    return data.get("w", default)


def biased_walks(graph, num_walks, walk_length, p, q, seed=42):
    """
    Generate node2vec-style biased random walks.

    Nodes are pipe IDs in the line graph.
    """
    random.seed(seed)

    walks = []
    nodes = list(graph.nodes())

    for _ in range(num_walks):
        random.shuffle(nodes)

        for start in nodes:
            walk = [start]
            previous = None

            for _ in range(walk_length - 1):
                current = walk[-1]
                neighbors = list(graph.neighbors(current))

                if not neighbors:
                    break

                if previous is None:
                    weights = [w_edge(graph, current, nb, 1.0) for nb in neighbors]
                else:
                    weights = []

                    for nb in neighbors:
                        return_bias = (1.0 / p) if nb == previous else 1.0
                        inout_bias = 1.0 if graph.has_edge(nb, previous) else (1.0 / q)
                        edge_weight = w_edge(graph, current, nb, 1.0)
                        weights.append(return_bias * inout_bias * edge_weight)

                total_weight = sum(weights)

                if total_weight <= 0:
                    next_node = random.choice(neighbors)
                else:
                    r = random.random() * total_weight
                    cumulative = 0.0
                    next_node = neighbors[-1]

                    for nb, weight in zip(neighbors, weights):
                        cumulative += weight
                        if cumulative >= r:
                            next_node = nb
                            break

                walk.append(next_node)
                previous = current

            walks.append([str(x) for x in walk])

    return walks


def make_output_filename(snapshot_year: int) -> Path:
    """
    Create a compact filename using only the embedding parameters
    """
    tag = (
        f"snap{snapshot_year}_"
        f"linegraph_"
        f"node2vec_"
        f"p{P_RET:g}_"
        f"q{Q_INOUT:g}_"
        f"W{WALKS_PER_NODE}_"
        f"L{WALK_LEN}_"
        f"d{EMB_DIM}_"
        f"win{WINDOW}_"
        f"ep{EPOCHS}_"
        f"neg{NEGATIVE}_"
        f"a{int(ALPHA_ATTR * 100)}_"
        f"seed{SEED}.csv"
    )

    return OUT_DIR / tag


# ---------------------------------------------------------------------
# 3. Load and prepare GPKG once
# ---------------------------------------------------------------------

def load_base_pipe_layer() -> gpd.GeoDataFrame:
    if not INPUT_GPKG.exists():
        raise FileNotFoundError(f"Input GeoPackage not found: {INPUT_GPKG}")

    pipes = gpd.read_file(INPUT_GPKG, layer=LAYER_NAME)
    pipes.columns = pipes.columns.str.strip()

    if ID_COL not in pipes.columns:
        raise ValueError(f"Required ID column not found: {ID_COL}")

    if pipes[ID_COL].duplicated().any():
        duplicated = pipes.loc[pipes[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Pipe IDs must be unique. Example duplicates: {duplicated[:5]}")

    if not pipes.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
        bad_types = pipes.geometry.geom_type.value_counts()
        raise ValueError(f"Pipe layer contains non-line geometries:\n{bad_types}")

    if PROJECT_TO_EPSG is not None:
        if pipes.crs is None:
            raise ValueError("Input pipe layer has no CRS. Set a CRS or disable reprojection.")
        pipes = pipes.to_crs(PROJECT_TO_EPSG)

    pipes = pipes.rename(columns={ID_COL: "pipe_id"})
    pipes["pipe_id"] = pipes["pipe_id"].astype(str).str.strip()
    pipes = pipes.set_index("pipe_id", drop=False)

    return pipes


# ---------------------------------------------------------------------
# 4. Build embeddings for one snapshot year
# ---------------------------------------------------------------------

def build_snapshot_features(base_pipes: gpd.GeoDataFrame, snapshot_year: int) -> Path:
    set_seeds(SEED)

    output_csv = make_output_filename(snapshot_year)

    if output_csv.exists():
        print(f"[exists] {output_csv}")
        return output_csv

    pipes = base_pipes.copy()

    # Snapshot-honest construction-year filtering
    if YEAR_COL in pipes.columns:
        yy = pd.to_numeric(pipes[YEAR_COL], errors="coerce")
        pipes = pipes[(yy.isna()) | (yy <= snapshot_year)].copy()

    if pipes.empty:
        raise ValueError(f"No pipes remain for snapshot year {snapshot_year}.")

    # -----------------------------------------------------------------
    # 4.1 Build junction graph and line graph
    # -----------------------------------------------------------------

    pipes["u_pt"], pipes["v_pt"] = zip(*pipes.geometry.apply(endpoints))
    pipes = pipes.dropna(subset=["u_pt", "v_pt"]).copy()

    def point_key(point):
        return (round(point.x, ROUND_NDP), round(point.y, ROUND_NDP))

    endpoints_df = pd.concat(
        [
            pipes[["u_pt"]].rename(columns={"u_pt": "pt"}),
            pipes[["v_pt"]].rename(columns={"v_pt": "pt"}),
        ],
        ignore_index=True
    )

    endpoints_df["key"] = endpoints_df["pt"].apply(point_key)

    unique_endpoints = endpoints_df.drop_duplicates("key").reset_index(drop=True)
    endpoint_lookup = dict(zip(unique_endpoints["key"], unique_endpoints.index))

    pipes["u_id"] = pipes["u_pt"].apply(lambda p: endpoint_lookup[point_key(p)])
    pipes["v_id"] = pipes["v_pt"].apply(lambda p: endpoint_lookup[point_key(p)])

    junction_graph = nx.MultiGraph()

    for pipe_id, row in pipes.iterrows():
        junction_graph.add_edge(
            f"n{row.u_id}",
            f"n{row.v_id}",
            key=pipe_id,
            pipe_id=pipe_id
        )

    line_graph = nx.line_graph(junction_graph)
    line_graph = nx.relabel_nodes(line_graph, {edge: edge[2] for edge in line_graph.nodes()})

    if line_graph.number_of_nodes() == 0:
        raise ValueError(f"Empty line graph for snapshot year {snapshot_year}.")

    print(
        f"[snapshot {snapshot_year}] "
        f"pipes={len(pipes)}, "
        f"line_graph_nodes={line_graph.number_of_nodes()}, "
        f"line_graph_edges={line_graph.number_of_edges()}"
    )

    # -----------------------------------------------------------------
    # 4.2 Snapshot-honest k-hop failure feature
    # -----------------------------------------------------------------

    failures_df = build_failure_table_from_wide(
        pipes,
        max_year=snapshot_year
    )

    if not failures_df.empty:
        if failures_df["year"].max() > snapshot_year:
            raise ValueError(
                f"Failure table contains years after snapshot year {snapshot_year}."
            )

        failures_df["count"] = pd.to_numeric(
            failures_df["count"],
            errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    else:
        failures_df = pd.DataFrame(columns=["pipe_id", "year", "count"])

    fail_flag = pd.Series(0.0, index=pipes.index, dtype=float)

    if not failures_df.empty:
        any_failure = (failures_df.groupby("pipe_id")["count"].sum() > 0).astype(float)
        common = fail_flag.index.intersection(any_failure.index)
        fail_flag.loc[common] = any_failure.reindex(common).fillna(0.0)

    def khop_fail_sum(pipe_id, radius=2, second_hop_weight=0.5):
        total = 0.0

        try:
            first_neighbors = list(line_graph.neighbors(pipe_id))
        except Exception:
            first_neighbors = []

        for nb1 in first_neighbors:
            total += float(fail_flag.get(nb1, 0.0))

        if radius >= 2 and first_neighbors:
            second_neighbors = set()

            for nb1 in first_neighbors:
                try:
                    second_neighbors.update(line_graph.neighbors(nb1))
                except Exception:
                    pass

            second_neighbors.discard(pipe_id)
            second_neighbors.difference_update(first_neighbors)

            for nb2 in second_neighbors:
                total += second_hop_weight * float(fail_flag.get(nb2, 0.0))

        return float(total)

    pipes["khop_fail_sum"] = pd.Series(
        {
            pipe_id: khop_fail_sum(
                pipe_id,
                radius=KHOP_RADIUS,
                second_hop_weight=KHOP_SECOND_HOP_WEIGHT
            )
            for pipe_id in line_graph.nodes()
        },
        dtype=float
    ).reindex(pipes.index).fillna(0.0)

    # -----------------------------------------------------------------
    # 4.3 Node2vec-style walks and Word2Vec embeddings
    # -----------------------------------------------------------------

    nx.set_edge_attributes(line_graph, 1.0, name="w")

    walks = biased_walks(
        graph=line_graph,
        num_walks=WALKS_PER_NODE,
        walk_length=WALK_LEN,
        p=P_RET,
        q=Q_INOUT,
        seed=SEED
    )

    model = Word2Vec(
        sentences=walks,
        vector_size=EMB_DIM,
        window=WINDOW,
        min_count=1,
        sg=1,
        negative=NEGATIVE,
        workers=max(1, WORKERS),
        epochs=EPOCHS,
        seed=SEED
    )

    pipe_ids = sorted(line_graph.nodes(), key=lambda x: str(x))
    emb_matrix = np.vstack([model.wv[str(pipe_id)] for pipe_id in pipe_ids])
    emb_cols = [f"emb_{i:02d}" for i in range(EMB_DIM)]

    emb_df = pd.DataFrame(
        emb_matrix,
        index=pipe_ids,
        columns=emb_cols
    )

    pipes = pipes.drop(columns=[c for c in pipes.columns if c.startswith("emb_")], errors="ignore")
    pipes = pipes.join(emb_df, how="left")

    for col in emb_cols:
        if pipes[col].isna().any():
            pipes[col] = pipes[col].fillna(pipes[col].mean())

    # -----------------------------------------------------------------
    # 4.4 Export only required columns
    # -----------------------------------------------------------------

    out = pipes.reset_index(drop=True).copy()
    out["id"] = out["pipe_id"].astype(str)

    export_cols = [
        "pipe_id",
        "id",
    ] + emb_cols + [
        "khop_fail_sum",
    ]

    out[export_cols].to_csv(output_csv, index=False)

    print(f"[saved] {output_csv}")
    print(
        f"[snapshot {snapshot_year}] exported pipes={len(out)}, "
        f"embedding_dim={EMB_DIM}, "
        f"khop_min={out['khop_fail_sum'].min():.3f}, "
        f"khop_mean={out['khop_fail_sum'].mean():.3f}, "
        f"khop_max={out['khop_fail_sum'].max():.3f}"
    )

    return output_csv


# ---------------------------------------------------------------------
# 5. Run all yearly snapshots
# ---------------------------------------------------------------------

def main():
    print("Building yearly graph embeddings for synthetic Net3 dataset.")
    print(f"Input GPKG: {INPUT_GPKG}")
    print(f"Output directory: {OUT_DIR}")

    base_pipes = load_base_pipe_layer()

    created_files = []

    for snapshot_year in SNAPSHOT_YEARS:
        output_csv = build_snapshot_features(
            base_pipes=base_pipes,
            snapshot_year=snapshot_year
        )
        created_files.append(output_csv)

    print("\nFinished building yearly embeddings.")
    print(f"Number of output files: {len(created_files)}")
    print("Files:")
    for path in created_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()