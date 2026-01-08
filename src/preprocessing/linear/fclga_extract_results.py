"""Extract strain results from Abaqus ODB files for elastic case.

This script processes Abaqus output database (.odb) files to extract
E11 strain values at integration points, averages them to nodes, and
saves the results to text files.

ELASTIC CASE: Extracts from Static (implicit) step.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import traceback
from pathlib import Path

# Abaqus Python API uses star imports by design - this is how Abaqus scripts work
# noqa:F405 suppresses linter warnings about undefined names from star imports
from abaqus import *  # noqa: F403
from abaqusConstants import *  # noqa: F403
import odbAccess


PROJECT_ROOT = Path.cwd()
SIMULATIONS_DIR = PROJECT_ROOT / "data" / "raw" / "linear" / "simulations"
STRAINS_DIR = PROJECT_ROOT / "data" / "interim" / "linear" / "strains"

def extract_e11_strains(odb_file, field_name='E', component_index=0):
    """Extract E11 strains from an Abaqus ODB file.
    
    Reads strain field data from the last frame of the last step,
    transforms to global coordinates, and averages integration point
    values to nodes.
    
    ELASTIC CASE: Uses ELEMENT_NODAL position (Static/Implicit solver).
    NONLINEAR CASE: Uses INTEGRATION_POINT position (Dynamic/Explicit solver).
    
    Args:
        odb_file: Path to the Abaqus .odb file
        field_name: Name of the field output to extract (default: 'E')
        component_index: Index of the strain component (default: 0 for E11)
        
    Returns:
        List of tuples (node_label, average_strain) sorted by node label,
        or empty list if extraction fails
    """
    try:
        odb = odbAccess.openOdb(odb_file)
        step = odb.steps.values()[-1]
        frame = step.frames[-1]
        
        if field_name not in frame.fieldOutputs:
            print(f"  ERROR: Field '{field_name}' not found!")
            odb.close()
            return []
        
        strain_field = frame.fieldOutputs[field_name]
        
        scratchOdb = session.ScratchOdb(odb)  # noqa: F405
        scratchOdb.rootAssembly.DatumCsysByThreePoints(
            name='CSYS-1',
            coordSysType=CARTESIAN,  # noqa: F405
            origin=(0.0, 0.0, 0.0),
            point1=(1.0, 0.0, 0.0),
            point2=(0.0, 1.0, 0.0)
        )
        
        datumCsys = scratchOdb.rootAssembly.datumCsyses['CSYS-1']
        transformed_field = strain_field.getTransformedField(datumCsys=datumCsys)
        
        # Elastic (Static/Implicit) uses ELEMENT_NODAL position
        # Nonlinear (Dynamic/Explicit) uses INTEGRATION_POINT position
        try:
            ip_field = transformed_field.getSubset(position=ELEMENT_NODAL)  # noqa: F405
        except:
            # Fallback to integration points if ELEMENT_NODAL not available
            ip_field = transformed_field.getSubset(position=INTEGRATION_POINT)  # noqa: F405
        
        node_strains = {}
        assembly = odb.rootAssembly
        
        for instance_name in assembly.instances.keys():
            instance = assembly.instances[instance_name]
            
            try:
                instance_field = ip_field.getSubset(region=instance)
            except Exception:
                continue
                
            for value in instance_field.values:
                element_label = value.elementLabel
                strain_value = value.data[component_index]
                
                try:
                    element = instance.elements[element_label - 1]
                    element_nodes = element.connectivity
                    
                    for node_label in element_nodes:
                        if node_label not in node_strains:
                            node_strains[node_label] = []
                        node_strains[node_label].append(strain_value)
                except Exception:
                    continue
        
        e11_strains = {
            node_label: sum(strain_list) / len(strain_list)
            for node_label, strain_list in node_strains.items()
        }

        odb.close()
        return sorted(e11_strains.items())

    except Exception as e:
        print(f"Error processing {odb_file}: {e}")
        traceback.print_exc()
        return []

def save_strains_to_file(strains, output_file):
    """Save strain data to a text file.
    
    Args:
        strains: List of tuples (node_label, strain_value)
        output_file: Path to output text file
    """
    with open(output_file, 'w') as f:
        for node_label, strain in strains:
            f.write(f"{node_label}, {strain}\n")

def process_odb_files(input_folder, output_folder):
    """Process all ODB files in a directory and extract E11 strains.
    
    Args:
        input_folder: Directory containing .odb files (searches recursively)
        output_folder: Directory where strain text files will be saved
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    odb_files = sorted(input_path.rglob("*.odb"))
    
    if not odb_files:
        print(f"ERROR: No .odb files found in {input_path}")
        return
    
    print(f"\nFound {len(odb_files)} ODB files to process\n")
    print("=" * 80)
    
    for i, file_path in enumerate(odb_files, 1):
        print(f"[{i}/{len(odb_files)}] Processing {file_path.name}...")

        e11_strains = extract_e11_strains(str(file_path))

        base_name = file_path.stem
        output_file = output_path / f"E11_{base_name}.txt"

        if e11_strains:
            save_strains_to_file(e11_strains, str(output_file))
            print(f"  ✓ Saved {len(e11_strains)} nodal strain values")
        else:
            print(f"  ⚠ Warning: No strains extracted from {file_path.name}")
    
    print("=" * 80)
    print("ELASTIC CASE: STRAIN EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Processed {len(odb_files)} files")
    print(f"Output directory: {output_path}")

def main():
    """Main execution function."""
    print("=" * 80)
    print("ELASTIC CASE: EXTRACTING STRAINS FROM ODB FILES")
    print("=" * 80)
    
    STRAINS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading ODB files from: {SIMULATIONS_DIR}")
    print(f"Output will be saved to: {STRAINS_DIR}")
    print()
    process_odb_files(SIMULATIONS_DIR, STRAINS_DIR)


if __name__ == '__main__':
    main()
