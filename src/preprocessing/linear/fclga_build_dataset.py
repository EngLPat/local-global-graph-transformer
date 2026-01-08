"""Build final strain dataset for FCLGA GraphTransformer training.

Combines extracted strain values from Abaqus .odb files with node ordering
from .inp geometry files to create properly aligned training data. Handles
variable mesh sizes through padding.

ELASTIC CASE: Static/Implicit solver with fabric composite material.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import re
from pathlib import Path

import torch

# Setup organized paths
PROJECT_ROOT = Path.cwd()
STRAINS_DIR = PROJECT_ROOT / "data" / "interim" / "linear" / "strains"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "linear"

# Create output directory
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def create_strains_tensor(input_folder, node_gnn_data_path):
    """Create strain tensor from text files aligned with node IDs.

    Reads strain values from Abaqus output files and aligns them with
    the node ordering from geometry files. Handles missing strain data
    (e.g., boundary nodes) by padding with zeros.

    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Strains directory not found: {input_path}")

    # Load node GNN data to get node IDs for each sample
    # Note: weights_only=False required for PyTorch Geometric Data objects
    node_gnn_data = torch.load(node_gnn_data_path, weights_only=False)
    print(f"Loaded {len(node_gnn_data)} samples from node GNN data")

    # Get all strain files and sort by sample number
    # Regex matches the number at the end: E11_Plate_0.txt -> 0
    files = sorted(
        input_path.glob("E11_*.txt"),
        key=lambda x: int(re.search(r"_(\d+)\.txt$", x.name).group(1))
        if re.search(r"_(\d+)\.txt$", x.name)
        else 0,
    )

    if not files:
        raise FileNotFoundError(f"No E11_*.txt files found in {input_path}")

    print(f"\nProcessing {len(files)} strain files...")

    all_strains = []

    # Process each strain file
    for i, file_path in enumerate(files, 1):
        if i % 100 == 0 or i == 1:
            print(f"  Processing sample {i}/{len(files)}: {file_path.name}")

        # Read strain data: format is "node_label, strain_value"
        strain_dict = {}
        with open(file_path) as f:
            for line in f:
                node_label, strain = line.split(", ")
                strain_dict[int(node_label.strip())] = float(strain.strip())

        # Get node IDs for this sample from the GNN data
        sample_idx = i - 1
        if sample_idx >= len(node_gnn_data):
            raise ValueError(f"Sample {sample_idx} not found in node GNN data")

        sample_data = node_gnn_data[sample_idx]
        if hasattr(sample_data, "node_ids"):
            node_ids = sample_data.node_ids.numpy()
        else:
            # Fallback: assume node IDs are 1-indexed sequential
            if i == 1:
                print("    WARNING: No node_ids found, assuming sequential 1-indexed")
            node_ids = list(range(1, sample_data.x.shape[0] + 1))

        # Align strains with node order from GNN data
        aligned_strains = []
        missing_count = 0
        for node_id in node_ids:
            if node_id in strain_dict:
                aligned_strains.append(strain_dict[node_id])
            else:
                # Node missing in strain data - use 0
                aligned_strains.append(0.0)
                missing_count += 1

        if missing_count > 0 and i <= 3:
            print(
                f"    INFO: {missing_count}/{len(node_ids)} nodes have no strain data (expected for boundary nodes)"
            )

        # Convert to tensor
        strain_tensor = torch.tensor(aligned_strains, dtype=torch.float32).unsqueeze(-1)
        all_strains.append(strain_tensor)

    # Pad all tensors to uniform size (required for batching)
    max_size = max(strain.size(0) for strain in all_strains)

    padded_strains = []
    for strain_tensor in all_strains:
        padding_size = max_size - strain_tensor.size(0)
        if padding_size > 0:
            padded_tensor = torch.cat([strain_tensor, torch.zeros(padding_size, 1)], dim=0)
        else:
            padded_tensor = strain_tensor
        padded_strains.append(padded_tensor)

    # Stack into single tensor
    strains_tensor = torch.stack(padded_strains, dim=0)

    return strains_tensor


if __name__ == "__main__":
    print("=" * 70)
    print("ELASTIC CASE: BUILDING STRAIN DATASET")
    print("=" * 70)
    print(f"Reading strain files from: {STRAINS_DIR}")
    print(f"Reading node data from: {DATA_PROCESSED}")
    print(f"Output will be saved to: {DATA_PROCESSED}")

    # Build strain tensor
    node_gnn_data_path = DATA_PROCESSED / "node_gnn_data.pt"
    strains_tensor = create_strains_tensor(STRAINS_DIR, node_gnn_data_path)

    # Save to organized directory
    output_path = DATA_PROCESSED / "strains.pt"
    torch.save(strains_tensor, str(output_path))

    print(f"\n{'=' * 70}")
    print("DATASET BUILDING COMPLETE")
    print(f"{'=' * 70}")
    print(f"✓ Saved strain tensor to: {output_path}")
    print(f"Tensor shape: {strains_tensor.shape}")
    print(f"[num_samples, max_nodes, features] = {list(strains_tensor.shape)}")
