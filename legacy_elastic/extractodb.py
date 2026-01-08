import os
import shutil

def copy_odb_files(source_directory, target_directory):
    # Check if the target directory exists, if not, create it
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Iterate over all folders in the source directory
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            # Check if the file has the .odb extension
            if file.endswith(".odb"):
                # Full path to the source file
                source_file = os.path.join(root, file)

                # Destination path in the target directory
                target_file = os.path.join(target_directory, file)

                # If the file already exists in the target directory, append a number to the filename
                counter = 1
                while os.path.exists(target_file):
                    name, ext = os.path.splitext(file)
                    target_file = os.path.join(target_directory, f"{name}_{counter}{ext}")
                    counter += 1

                # Copy the .odb file to the target directory
                shutil.copy(source_file, target_file)
                print(f"Copied {file} to {target_file}")

# Define the source and target directories
# source_directory = "C:/Users/lpatrign/Desktop/ODBs"  # Change to the parent directory containing your subfolders
# target_directory = "C:/Users/lpatrign/Desktop/ODBsONLY"   # Change to your main folder where you want to gather all the .inp files

source_directory = os.path.join(".", "ODBs")
target_directory = os.path.join(".", "ODBsONLY")

# Call the function
copy_odb_files(source_directory, target_directory)
