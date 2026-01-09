"""
FCLGA GraphTransformer - Main Pipeline Script

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
Paper: "Graph Neural Networks with Hybrid Local-Global Attention for
        Effective Prediction of Mechanical Response in Structures"
Journal: Computer Methods in Applied Mechanics and Engineering

This script orchestrates the complete pipeline from geometry generation to model testing.
Designed for reproducibility and publication.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Project root is two levels up from this file (src/preprocessing/fclga_run_pipeline.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_subprocess(cmd, description, cwd=None):
    """Run a subprocess command with error handling and logging."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd}\n")

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, cwd=cwd, capture_output=False, text=True
        )
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}")
        print(f"Exit code: {e.returncode}\n")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in {description}: {str(e)}\n")
        return False


def check_abaqus_available():
    """Check if Abaqus is available in the system."""
    try:
        result = subprocess.run(
            "abaqus information=system", shell=True, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except:
        return False


def run_preprocessing(args, project_root, material_type):
    """Run the complete preprocessing pipeline."""
    print("\n" + "=" * 60)
    print(f"PREPROCESSING PIPELINE ({material_type})")
    print("=" * 60)

    # Check for Abaqus
    if not check_abaqus_available():
        print("\n⚠ WARNING: Abaqus not detected in system PATH")
        print("Abaqus is required for steps 1, 2, and 4")
        print("Please ensure Abaqus is installed and accessible\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return False

    # Define organized paths with material_type
    src_preprocessing = os.path.join(project_root, "src", "preprocessing", material_type)
    geometry_dir = os.path.join(project_root, "data", "raw", material_type, "geometry")
    simulations_dir = os.path.join(project_root, "data", "raw", material_type, "simulations")
    strains_dir = os.path.join(project_root, "data", "interim", material_type, "strains")
    processed_dir = os.path.join(project_root, "data", "processed", material_type)

    # Step 1: Generate Geometry (Abaqus script)
    print("\n" + "=" * 60)
    print("Step 1/6: Generating Geometry")
    print("=" * 60)
    print("This creates .inp files and geometry tensor")
    print(f"Expected output: {geometry_dir}/*.inp, {processed_dir}/plate_geometry_data.pt\n")

    geometry_script = os.path.join(src_preprocessing, "fclga_generate_geometry.py")
    cmd = f"abaqus cae nogui={geometry_script}"
    if not run_subprocess(cmd, "Geometry Generation", cwd=project_root):
        print("⚠ Geometry generation failed. Cannot continue.")
        return False

    # Check output
    if not os.path.exists(geometry_dir):
        print(f"✗ Geometry directory not created: {geometry_dir}")
        return False
    inp_files = list(Path(geometry_dir).glob("*.inp"))
    print(f"✓ Generated {len(inp_files)} .inp files in {geometry_dir}\n")

    # Step 2: Run FEA Simulations (Python script calling Abaqus)
    print("\n" + "=" * 60)
    print("Step 2/6: Running FEA Simulations")
    print("=" * 60)
    print(f"Parallel simulations: {args.num_cpus} CPUs")
    print(f"Expected output: {simulations_dir}/*/[model].odb")
    print("⚠ This step takes considerable time (hours for 500 models)\n")

    simulation_script = os.path.join(src_preprocessing, "fclga_run_simulations.py")
    cmd = f"python {simulation_script}"

    if not run_subprocess(cmd, "FEA Simulations", cwd=project_root):
        print("⚠ FEA simulations failed. Cannot continue.")
        return False

    # Check output
    if not os.path.exists(simulations_dir):
        print(f"✗ Simulations directory not created: {simulations_dir}")
        return False
    odb_files = list(Path(simulations_dir).rglob("*.odb"))
    print(f"✓ Generated {len(odb_files)} .odb files in {simulations_dir}\n")

    # Step 3: Extract Features (Python script)
    print("\n" + "=" * 60)
    print("Step 3/6: Extracting Features from Mesh")
    print("=" * 60)
    print("This parses .inp files to create graph structure")
    print(f"Expected output: {processed_dir}/node_gnn_data.pt, triangulation_data.pkl\n")

    features_script = os.path.join(src_preprocessing, "fclga_extract_features.py")
    cmd = f"python {features_script}"
    if not run_subprocess(cmd, "Feature Extraction", cwd=project_root):
        print("⚠ Feature extraction failed. Cannot continue.")
        return False

    # Check output
    node_data = os.path.join(processed_dir, "node_gnn_data.pt")
    if not os.path.exists(node_data):
        print(f"✗ {node_data} not created!")
        return False
    print(f"✓ Created node_gnn_data.pt and triangulation_data.pkl in {processed_dir}\n")

    # Step 4: Extract Results (Abaqus script)
    print("\n" + "=" * 60)
    print("Step 4/6: Extracting Results from ODBs")
    print("=" * 60)
    print("This reads .odb files to extract strain values")
    print(f"Expected output: {strains_dir}/E11_*.txt\n")

    results_script = os.path.join(src_preprocessing, "fclga_extract_results.py")
    cmd = f"abaqus cae nogui={results_script}"
    if not run_subprocess(cmd, "Result Extraction", cwd=project_root):
        print("⚠ Result extraction failed. Cannot continue.")
        return False

    # Check output
    if not os.path.exists(strains_dir):
        print(f"✗ Strains directory not created: {strains_dir}")
        return False
    strain_files = list(Path(strains_dir).glob("E11_*.txt"))
    print(f"✓ Generated {len(strain_files)} strain files in {strains_dir}\n")

    # Step 5: Build Dataset (Python script)
    print("\n" + "=" * 60)
    print("Step 5/6: Building Strain Tensor")
    print("=" * 60)
    print("This combines strain data into tensor format")
    print(f"Expected output: {processed_dir}/strains.pt\n")

    dataset_script = os.path.join(src_preprocessing, "fclga_build_dataset.py")
    cmd = f"python {dataset_script}"
    if not run_subprocess(cmd, "Strain Tensor Building", cwd=project_root):
        print("⚠ Strain tensor building failed.")
        return False

    # Check output
    strains_tensor = os.path.join(processed_dir, "strains.pt")
    if not os.path.exists(strains_tensor):
        print(f"✗ {strains_tensor} not created!")
        return False
    print(f"✓ Created strains.pt in {processed_dir}\n")

    # Step 6: Prepare Training Data (Python script)
    print("\n" + "=" * 60)
    print("Step 6/6: Preparing Final Training Dataset")
    print("=" * 60)
    print("This combines all data into PyTorch Geometric Data objects")
    print(f"Expected output: {processed_dir}/datasets/processed_data.pt\n")

    prepare_script = os.path.join(src_preprocessing, "fclga_prepare_training_data.py")
    cmd = f"python {prepare_script}"
    if not run_subprocess(cmd, "Training Data Preparation", cwd=project_root):
        print("⚠ Training data preparation failed.")
        return False

    # Check output
    datasets_dir = processed_dir
    processed_data = os.path.join(datasets_dir, "processed_data.pt")
    if not os.path.exists(processed_data):
        print(f"✗ {processed_data} not created!")
        return False
    print(f"✓ Created processed_data.pt in {datasets_dir}\n")

    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nOrganized Data Structure:")
    print(f"  📁 {geometry_dir}")
    print(f"  📁 {simulations_dir}")
    print(f"  📁 {strains_dir}")
    print(f"  📁 {processed_dir}")
    print(f"  📄 {processed_data} - Ready for training!")
    print("\nYou can now run: python -m src.training.fclga_train_model\n")

    return True


def run_training(args, project_root):
    """Run the training pipeline."""
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    training_script = os.path.join(project_root, "src", "training", "fclga_train_model.py")

    if not os.path.exists(training_script):
        print(f"\n✗ Training script not found: {training_script}")
        print("Please implement the training module first.\n")
        return False

    cmd = "python -m src.training.fclga_train_model"
    return run_subprocess(cmd, "Model Training", cwd=project_root)


def run_testing(args, project_root):
    """Run the testing pipeline."""
    print("\n" + "=" * 60)
    print("TESTING PIPELINE")
    print("=" * 60)

    testing_script = os.path.join(project_root, "src", "evaluation", "fclga_test_model.py")

    if not os.path.exists(testing_script):
        print(f"\n✗ Testing script not found: {testing_script}")
        print("Please implement the evaluation module first.\n")
        return False

    cmd = "python -m src.evaluation.fclga_test_model"
    return run_subprocess(cmd, "Model Testing", cwd=project_root)


def main():
    parser = argparse.ArgumentParser(
        description="FCLGA GraphTransformer Complete Reproducible Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (nonlinear, default)
  python -m src.preprocessing.fclga_run_pipeline --stage all
  
  # Run preprocessing for linear material
  python -m src.preprocessing.fclga_run_pipeline --stage preprocess --material_type linear --num_cpus 8
  
  # Run only training
  python -m src.preprocessing.fclga_run_pipeline --stage train
  
  # Run only testing
  python -m src.preprocessing.fclga_run_pipeline --stage test
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "train", "test"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--material_type",
        choices=["linear", "nonlinear"],
        default="nonlinear",
        help="Material type (default: nonlinear)",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="Number of CPUs for parallel simulation (default: auto-detect)",
    )
    parser.add_argument(
        "--output_variables",
        nargs="+",
        default=["E11"],
        help="Variables to extract from FEA results (currently: E11)",
    )

    args = parser.parse_args()

    # Auto-detect CPUs if not specified
    if args.num_cpus is None:
        import multiprocessing

        args.num_cpus = max(1, multiprocessing.cpu_count() // 2)

    # Get project root (two levels up: src/preprocessing/ -> src/ -> root/)
    project_root = str(PROJECT_ROOT)

    print(f"\n{'=' * 60}")
    print("FCLGA GraphTransformer Pipeline")
    print(f"{'=' * 60}")
    print(f"Project root: {project_root}")
    print(f"Material type: {args.material_type}")
    print(f"Stage: {args.stage}")
    print(f"CPUs: {args.num_cpus}")
    print(f"{'=' * 60}\n")

    success = True

    # Run requested stages
    if args.stage in ["all", "preprocess"]:
        success = run_preprocessing(args, project_root, args.material_type)
        if not success and args.stage == "all":
            print("\n✗ Preprocessing failed. Stopping pipeline.")
            sys.exit(1)

    if success and args.stage in ["all", "train"]:
        success = run_training(args, project_root)
        if not success and args.stage == "all":
            print("\n✗ Training failed. Stopping pipeline.")
            sys.exit(1)

    if success and args.stage in ["all", "test"]:
        success = run_testing(args, project_root)

    if success:
        print(f"\n{'=' * 60}")
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n{'=' * 60}")
        print("✗ PIPELINE FAILED")
        print(f"{'=' * 60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
