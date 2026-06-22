"""
Run the full synthetic Net3 RSF workflow.

This script executes all workflow steps in order:
1. Generate graph features and node2vec-style embeddings
2. Align yearly embeddings
3. Prepare recurrent gap-time survival data
4. Train/evaluate RSF with Harrell C-index, top-20% recall, permutation importance
5. Train temporal RSF and generate survival plots
"""

from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "src/graph_features_from_gpkg.py",
    "src/aligned_graph_features.py",
    "src/prepare_gap-time_setup.py",
    "src/train_evaluate_RSF.py",
    "src/practical_results_RSF.py",
]


def run_script(script_relative_path: str) -> None:
    script_path = ROOT_DIR / script_relative_path

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 80)
    print(f"Running: {script_relative_path}")
    print("=" * 80)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT_DIR,
        check=True,
    )


def main() -> None:
    print("Running full synthetic Net3 RSF workflow.")
    print(f"Repository root: {ROOT_DIR}")

    for script in SCRIPTS:
        run_script(script)

    print("\n" + "=" * 80)
    print("Synthetic workflow completed successfully.")
    print("=" * 80)

    print("\nExpected output folders:")
    print("outputs/graph_features/")
    print("outputs/aligned_graph_features/")
    print("outputs/survival_data/")
    print("outputs/rsf_results_synthetic_minimal/")
    print("outputs/rsf_temporal_survival_plots_synthetic/")


if __name__ == "__main__":
    main()