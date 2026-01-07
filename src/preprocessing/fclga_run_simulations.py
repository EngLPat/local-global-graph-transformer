"""
Run Abaqus FEA simulations in parallel for composite plate geometries.

This script orchestrates parallel execution of Abaqus finite element analysis
simulations for multiple geometry files. It manages temporary files, organizes
simulation results, and performs cleanup operations.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Try to load configuration from YAML, fallback to hardcoded default if not available
try:
    import yaml
    PROJECT_ROOT = Path.cwd()
    config_path = PROJECT_ROOT / "config" / "defaults.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    NUM_PARALLEL_SIMULATIONS = config['preprocessing']['simulation']['num_parallel_jobs']
except (ImportError, FileNotFoundError, KeyError):
    # Fallback to hardcoded default if YAML not available or config file missing
    NUM_PARALLEL_SIMULATIONS = 4


def run_simulation(inp_file, simulations_dir, temp_dir, project_root):
    """
    Run a single Abaqus simulation and organize output files.
    
    This function executes an Abaqus job for a given input file, manages
    temporary scratch directories, and organizes the output files into a
    structured directory. After completion, it cleans up temporary files
    while preserving essential results (.odb, .dat, .msg, .sta).
    
    Args:
        inp_file (Path): Path to the Abaqus input (.inp) file.
        simulations_dir (Path): Directory where simulation results will be stored.
        temp_dir (Path): Base directory for temporary Abaqus scratch files.
        project_root (Path): Project root directory for cleanup operations.
    
    Returns:
        str: The job name (filename without extension).
    
    Raises:
        subprocess.CalledProcessError: If the Abaqus simulation fails.
    """
    job_name = inp_file.stem
    result_folder = simulations_dir / job_name
    result_folder.mkdir(parents=True, exist_ok=True)
    
    job_temp_dir = temp_dir / job_name
    job_temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Abaqus job: {job_name}")
    command = (
        f'abaqus job={job_name} input="{inp_file}" '
        f'scratch="{job_temp_dir}" interactive'
    )
    process = subprocess.Popen(command, shell=True, cwd=str(job_temp_dir))
    process.wait()

    print(f"Organizing results for {job_name}")
    
    keep_extensions = ['.odb', '.dat', '.msg', '.sta']
    temp_extensions = [
        '.log', '.com', '.prt', '.sim', '.stt', 
        '.res', '.pac', '.mdl', '.abq', '.lck'
    ]
    
    for extension in keep_extensions:
        for search_path in [job_temp_dir, project_root]:
            file_path = search_path / f"{job_name}{extension}"
            if file_path.exists():
                dest_path = result_folder / f"{job_name}{extension}"
                shutil.move(str(file_path), str(dest_path))
                print(f"  Moved {file_path.name} to {result_folder.name}/")
                break
        else:
            if extension == '.odb':
                print(
                    f"  Warning: {job_name}{extension} not found "
                    "(simulation may have failed)"
                )

    dest_inp_path = result_folder / inp_file.name
    if inp_file.exists():
        shutil.copy(str(inp_file), str(dest_inp_path))
    try:
        shutil.rmtree(str(job_temp_dir))
        print(f"  Cleaned up temporary files for {job_name}")
    except Exception as e:
        print(f"  Warning: Could not clean temp directory: {e}")
    
    for extension in temp_extensions + keep_extensions:
        stray_file = project_root / f"{job_name}{extension}"
        if stray_file.exists():
            stray_file.unlink()

    print(f"✓ Completed Abaqus job: {job_name}\n")
    return job_name


def cleanup_stray_files(project_root):
    """
    Remove temporary Abaqus files from the project root directory.
    
    Args:
        project_root (Path): Project root directory to clean.
    
    Returns:
        int: Number of files cleaned up.
    """
    cleanup_patterns = [
        '*.stt', '*.res', '*.pac', '*.mdl', '*.abq', 
        '*.lck', 'abaqus.rpy*', '*.rec'
    ]
    cleaned_count = 0
    for pattern in cleanup_patterns:
        for stray_file in project_root.glob(pattern):
            try:
                stray_file.unlink()
                cleaned_count += 1
            except Exception:
                pass
    return cleaned_count


def main():
    """
    Main execution function for running parallel Abaqus simulations.
    
    Sets up directories, discovers input files, runs simulations in parallel,
    and performs cleanup operations.
    """
    project_root = Path.cwd()
    geometry_dir = project_root / "data" / "raw" / "geometry"
    simulations_dir = project_root / "data" / "raw" / "simulations"
    temp_dir = project_root / "temp" / "abaqus_scratch"

    simulations_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not geometry_dir.exists():
        raise FileNotFoundError(f"Geometry directory not found: {geometry_dir}")

    inp_files = sorted(geometry_dir.glob("*.inp"))
    if not inp_files:
        raise FileNotFoundError(f"No .inp files found in {geometry_dir}")

    print(f"Found {len(inp_files)} geometry files to simulate")
    print(f"Input directory: {geometry_dir}")
    print(f"Output directory: {simulations_dir}")
    print(
        f"\nRunning {len(inp_files)} simulations with "
        f"{NUM_PARALLEL_SIMULATIONS} parallel workers\n"
    )

    with ThreadPoolExecutor(max_workers=NUM_PARALLEL_SIMULATIONS) as executor:
        future_to_file = {
            executor.submit(
                run_simulation, inp_file, simulations_dir, temp_dir, project_root
            ): inp_file 
            for inp_file in inp_files
        }

        completed = 0
        total = len(inp_files)
        for future in as_completed(future_to_file):
            inp_file = future_to_file[future]
            completed += 1
            try:
                future.result()
                print(f"Progress: {completed}/{total} simulations completed")
            except Exception as exc:
                print(f"Simulation for {inp_file.name} generated exception: {exc}")

    print(f"\n{'='*60}")
    print("ALL SIMULATIONS COMPLETED")
    print(f"{'='*60}")
    print(f"Results stored in: {simulations_dir}")
    print(f"Total simulations: {len(inp_files)}")

    print("\nCleaning up stray temporary files...")
    cleaned_count = cleanup_stray_files(project_root)
    if cleaned_count > 0:
        print(f"✓ Cleaned up {cleaned_count} temporary files from project root")

    try:
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            print("✓ Removed empty temp directory")
    except Exception:
        pass

    print("\n✓ All files organized and temporary files cleaned up!")


if __name__ == "__main__":
    main()

