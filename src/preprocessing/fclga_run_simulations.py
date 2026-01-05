import os
import subprocess
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up directories - using relative paths to current working directory
inp_directory = os.path.join(".", "INPs")  # Input folder in current directory
results_directory = os.path.join(".", "ODBs")  # Output folder in current directory

# Verify directories exist
if not os.path.exists(inp_directory):
    raise FileNotFoundError(f"The input directory '{inp_directory}' was not found.")
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Get list of all inp files in the directory
inp_files = [f for f in os.listdir(inp_directory) if f.endswith('.inp')]
inp_files.sort()  # Ensure order if needed

# Number of parallel simulations to run
num_parallel_simulations = 4  # Reduced for small test

def run_simulation(inp_file):
    job_name = os.path.splitext(inp_file)[0]
    result_folder = os.path.join(results_directory, job_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    inp_path = os.path.join(inp_directory, inp_file)

    # Submit the Abaqus job
    print(f"Starting Abaqus job: {job_name}")
    # Use platform-independent path handling
    command = f'abaqus job={job_name} input="{inp_path}" interactive'
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for the simulation to complete

    # After completion, move result files and copy .inp to the specific folder
    print(f"Organizing results for {job_name} in {result_folder}")
    for extension in ['.odb', '.dat', '.msg', '.sta', '.log', '.com', '.prt', '.sim']:
        # Look for output files in the current directory where Abaqus creates them
        file_path = os.path.join(".", job_name + extension)  
        dest_path = os.path.join(result_folder, job_name + extension)

        # Move the result files to the result folder
        if os.path.exists(file_path):
            shutil.move(file_path, dest_path)
            print(f"Moved {file_path} to {dest_path}")
        else:
            print(f"File not found: {file_path}")

    # Always copy the .inp file
    inp_file_path = os.path.join(inp_directory, job_name + '.inp')
    dest_inp_path = os.path.join(result_folder, job_name + '.inp')
    if os.path.exists(inp_file_path):
        shutil.copy(inp_file_path, dest_inp_path)

    print(f"Completed Abaqus job: {job_name}")

# Use a thread pool to manage parallel jobs
with ThreadPoolExecutor(max_workers=num_parallel_simulations) as executor:
    # Submit all simulations to the executor
    future_to_file = {executor.submit(run_simulation, inp_file): inp_file for inp_file in inp_files}

    # Monitor completion of batches
    batch_number = 0
    for future in as_completed(future_to_file):
        inp_file = future_to_file[future]
        batch_number += 1
        try:
            future.result()  # Get the result (or exception) from the simulation
            print(f"Batch {batch_number}: Simulation completed for {inp_file}")
        except Exception as exc:
            print(f"Batch {batch_number}: Simulation for {inp_file} generated an exception: {exc}")

print("All simulations have been completed and stored in respective folders.")