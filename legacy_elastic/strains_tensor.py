import os
import torch
import re  # Regular expression for extracting numbers from filenames

def create_strains_tensor(input_folder):
    all_strains = []
    element_ids = []

    # Get all files that start with "STRAIN" and sort them numerically
    files = [f for f in os.listdir(input_folder) if f.startswith("E11")]
    
    # Sort files numerically based on the model number extracted from the filename
    files.sort(key=lambda x: int(re.search(r'E11_Plate_(\d+)\.txt', x).group(1)))

    # Collect all strain tensors from the sorted files
    for file_name in files:
        print(f"Processing sample {files.index(file_name) + 1}/{len(files)}: {file_name}")
        file_path = os.path.join(input_folder, file_name)

        # Read the strain data from the text file
        strains = []
        with open(file_path, 'r') as f:
            for line in f:
                element_label, strain = line.split(", ")
                strains.append(float(strain.strip()))
                element_ids.append(int(element_label.strip()))  # Collect element IDs

        # Convert to a tensor and add it to the list
        strain_tensor = torch.tensor(strains).unsqueeze(-1)  # Shape [n, 1]
        all_strains.append(strain_tensor)

    # Find the maximum size of all tensors (to use for padding)
    max_size = max(strain.size(0) for strain in all_strains)

    # Pad all tensors to the same size and preserve the element order
    
    padded_strains = []
    for strain_tensor in all_strains:
        padding_size = max_size - strain_tensor.size(0) + 1  # ADDED one more for debugging.
        if padding_size > 0:
            # Pad the tensor with zeros
            padded_tensor = torch.cat([strain_tensor, torch.zeros(padding_size, 1)], dim=0)
        else:
            padded_tensor = strain_tensor
        padded_strains.append(padded_tensor)

    # Stack all the padded tensors into a single tensor
    strains_tensor = torch.stack(padded_strains, dim=0)  # Shape [num_samples, max_size, 1]

    return strains_tensor

# Usage
input_folder = os.path.join(".", "strains")

strains_tensor = create_strains_tensor(input_folder)

# Save the resulting stress tensor without scaling
torch.save(strains_tensor, "strains.pt")
print("Saved the strain tensor to strains.pt without normalization.")