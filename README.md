# Code and synthetic data for topology-informed recurrent survival modelling of water pipe failures

This repository contains code and synthetic Net3-based demonstration data for reproducing the computational workflow of the topology informed gap-time modelling.

The workflow demonstrates:

* topology-based graph-feature generation from a pipe-network GIS file;
* yearly node2vec-style embedding generation using pipe-line-graph topology;
* yearly embedding alignment;
* recurrent gap-time survival-data preparation;
* Random Survival Forest modelling and holdout temporal evaluation;
* temporal prediction for a 2020–2024 evaluation window;
* Harrell C-index, top-20% recall, permutation importance, and survival-curve visualization.

The synthetic dataset do not contain any confidential water utility data. They are provided only to demonstrate the structure and computational workflow used. Numerical results are produced from synthetic data and are intended only to demonstrate the workflow and do not represent real-world model performance.

## Running versus inspecting the workflow

To fully reproduce the synthetic workflow, a Python environment with the required libraries is needed. The recommended installation method is provided through `environment.yml`.

Users who do not wish to install Python packages can still inspect the source code, synthetic input data, and any selected pre-generated outputs included in the `outputs/` folder. Full reproducibility requires running the workflow from the synthetic input data.

## Repository structure

```text
Topology-informed_Gap-time_Pipe_failures/
├── README.md
├── requirements.txt
├── LICENSE.txt
├── environment.yml
├── run_all_synthetic_workflow.py
├── data_synthetic/
│   ├── Net3.inp
│   ├── Net3.gpkg
│   └── Synthetic_assets_and_failures_from_Net3.csv
├── src/
│   ├── graph_features_from_gpkg.py
│   ├── aligned_graph_features.py
│   ├── prepare_gap-time_setup.py
│   ├── train_evaluate_RSF.py
│   └── practical_results_RSF.py
└── outputs/
    ├── graph_features/
    ├── aligned_graph_features/
    ├── survival_data/
    ├── rsf_results_synthetic_minimal/
    └── rsf_temporal_survival_plots_synthetic/
```

## Data

The `data_synthetic/` folder contains a synthetic public dataset derived from the EPANET Net3 network. The file `Synthetic_assets_and_failures_from_Net3.csv` contains pipe-level static input variables and yearly synthetic failure indicators.

The main columns are:

```text
id
Installation Year
Length
Material
Diameter
Service Connections
Air Release Valve
Fire Hydrants
Outlets
Road Intersections
Elevation
Near Railways
Soil
Landuse
GWT
2007
2008
...
2024
```

The GIS file `Net3.gpkg` contains the pipe geometries used for graph construction. Some compatibility columns are retained for the graph-feature workflow, including `EntityID`, `ConstructionYear`, `InnerDiameter`, and `PipeMaterial`.

## Software requirements

The workflow can be run from a standard terminal, Anaconda Prompt, VS Code terminal, PyCharm terminal, or any Python environment. Spyder is not required.

Python 3.11 is recommended.

## Installation

Two installation options are provided.

### Option 1: recommended installation using conda

Because this workflow uses geospatial and survival-analysis libraries with compiled dependencies, installation through `conda-forge` is recommended.

From the repository root, run:

```bash
conda env create -f environment.yml
conda activate rsf_synthetic
```

### Option 2: installation using standard Python and pip

If Python is already installed, the workflow can also be run without Anaconda or Miniconda.

On Windows, open Command Prompt or PowerShell, go to the repository folder, and run:

```bat
cd "C:\path\to\Code-Model-Submission"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all_synthetic_workflow.py
```

On macOS/Linux, open a terminal, go to the repository folder, and run:

```bash
cd /path/to/Code-Model-Submission
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all_synthetic_workflow.py
```

If installation with pip fails for `geopandas` or `scikit-survival`, use the recommended conda installation method with `environment.yml`.

## Running the full workflow

After installing the required libraries, run the full workflow from the repository root:

```bash
python run_all_synthetic_workflow.py
```

This executes all scripts in order:

```text
src/graph_features_from_gpkg.py
src/aligned_graph_features.py
src/prepare_gap-time_setup.py
src/Train_evaluate_RSF.py
src/Train_RSF_temporal_survival_plots.py
```

All paths are resolved relative to the repository root, so no manual working-directory setup is required.

## Step-by-step execution

The scripts can also be run one by one:

```bash
python src/graph_features_from_gpkg.py
python src/aligned_graph_features.py
python src/prepare_gap-time_setup.py
python src/Train_evaluate_RSF.py
python src/Train_RSF_temporal_survival_plots.py
```

## Expected outputs

After running the workflow, the following folders are created or updated.

### 1. Graph features and yearly embeddings

```text
outputs/graph_features/
```

This folder contains yearly pipe-line-graph embedding files for 2007–2019, including node2vec-style embedding columns and graph/failure-informed scalar features such as `khop_fail_sum`.

### 2. Aligned yearly embeddings

```text
outputs/aligned_graph_features/
```

This folder contains yearly embedding files aligned to a common reference embedding space.

### 3. Recurrent gap-time survival data

```text
outputs/survival_data/
├── gap_time_train_intervals_2007_2019_with_embeddings.csv
└── eval_window_2020_2024_with_embeddings.csv
```

The training file contains recurrent gap-time survival intervals for 2007–2019. The evaluation file contains one fixed 2020–2024 prediction-window row per eligible pipe.

### 4. Minimal RSF evaluation

```text
outputs/rsf_results_synthetic_minimal/
├── test_predictions.csv
├── summary_metrics.csv
├── summary_metrics.json
├── permutation_importance_harrell_c.csv
└── permutation_importance_harrell_c.png
```

This script reports Harrell C-index, top-20% recall, and permutation importance based on Harrell C-index.

### 5. Temporal prediction and survival plots

```text
outputs/rsf_temporal_survival_plots_synthetic/
├── temporal_predictions_all_pipes.csv
├── summary_metrics.csv
├── summary_metrics.json
├── survival_by_risk_groups.png
├── survival_by_material.png
└── survival_by_installation_year_bins.png
```

This script trains the Random Survival Forest using gap-time training intervals and generates temporal predictions for all eligible pipes in the 2020–2024 evaluation window. It also produces survival plots by predicted risk group, material, and installation-year bin.

## Interpretation of the temporal predictions

The temporal evaluation is formulated as a fixed 2020–2024 prediction window. For each pipe, the model estimates the probability of experiencing at least one failure within five years.

The evaluation outcome is represented using:

```text
start_year
end_year
duration
event
```

where `event = 1` indicates that the pipe had at least one failure within 2020–2024, and `duration` is the time from 2020 to the first failure or to right censoring at the end of 2024.

Therefore, the model produces pipe-level five-year failure risks. Survival curves show cumulative survival over gap time.

## Reproducibility note

The synthetic Net3 dataset is provided to demonstrate the computational workflow. It follows the same data structure and modelling workflow as the original utility-based application, but the synthetic dataset and resulting numerical outputs are provided only for demonstration and reproducibility purposes.

## Code and data availability statement

The code and synthetic demonstration data are provided to reproduce the computational workflow. Users may adapt the workflow to their own pipe asset and failure data by replacing the synthetic input files with datasets that follow the required structure.

## License

This repository is released under the BSD 3-Clause License. See the `LICENSE` file for details.