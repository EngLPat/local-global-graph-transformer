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
    
    IMPORTANT: This function transforms strains to a global Cartesian
    coordinate system with origin at (0,0,0). If you use a custom Abaqus
    macro that already outputs in global coordinates, you may need to
    skip the transformation step (remove the getTransformedField call).
    
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
        
        # Debug: Print step information
        step_names = odb.steps.keys()
        print(f"  DEBUG: Available steps: {step_names}")
        
        step = odb.steps.values()[-1]
        print(f"  DEBUG: Using step: {step.name}, frames: {len(step.frames)}")
        
        frame = step.frames[-1]
        
        # Debug: Print available field outputs
        available_fields = frame.fieldOutputs.keys()
        print(f"  DEBUG: Available field outputs: {available_fields}")
        
        if field_name not in frame.fieldOutputs:
            print(f"  ERROR: Field '{field_name}' not found!")
            odb.close()
            return []
        
        strain_field = frame.fieldOutputs[field_name]
        print(f"  DEBUG: Strain field has {len(strain_field.values)} values")
        
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
        
        # Debug: Check available positions
        print(f"  DEBUG: Checking available positions...")
        print(f"  DEBUG: Total transformed field values: {len(transformed_field.values)}")
        
        # Try different positions - elastic uses ELEMENT_NODAL, not INTEGRATION_POINT
        try:
            ip_field = transformed_field.getSubset(position=INTEGRATION_POINT)  # noqa: F405
            print(f"  DEBUG: INTEGRATION_POINT field has {len(ip_field.values)} values")
        except:
            ip_field = None
            print(f"  DEBUG: INTEGRATION_POINT not available")
        
        try:
            nodal_field = transformed_field.getSubset(position=ELEMENT_NODAL)  # noqa: F405
            print(f"  DEBUG: ELEMENT_NODAL field has {len(nodal_field.values)} values")
            if nodal_field and len(nodal_field.values) > 0:
                ip_field = nodal_field  # Use nodal values for elastic case
        except:
            print(f"  DEBUG: ELEMENT_NODAL not available")
        
        if not ip_field or len(ip_field.values) == 0:
            print(f"  ERROR: No valid field position found!")
            odb.close()
            return []
        
        node_strains = {}
        assembly = odb.rootAssembly
        
        print(f"  DEBUG: Processing {len(assembly.instances.keys())} instances")
        
        for instance_name in assembly.instances.keys():
            instance = assembly.instances[instance_name]
            print(f"  DEBUG: Instance '{instance_name}' has {len(instance.elements)} elements")
            
            try:
                instance_field = ip_field.getSubset(region=instance)
                print(f"  DEBUG: Instance field has {len(instance_field.values)} values")
            except Exception as e:
                print(f"  DEBUG: Failed to get instance field: {e}")
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
                except Exception as e:
                    print(f"  DEBUG: Element processing error: {e}")
                    continue
        
        print(f"  DEBUG: Collected strains for {len(node_strains)} nodes")
        
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
    # Create log file to capture output
    log_file = PROJECT_ROOT / "extraction_linear.log"
    
    import sys
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(str(log_file))
    
    print("=" * 80)
    print("ELASTIC CASE: EXTRACTING STRAINS FROM ODB FILES")
    print("=" * 80)
    print(f"Working directory: {PROJECT_ROOT}")
    print(f"SIMULATIONS_DIR: {SIMULATIONS_DIR}")
    print(f"SIMULATIONS_DIR exists: {SIMULATIONS_DIR.exists()}")
    print()
    
    STRAINS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading ODB files from: {SIMULATIONS_DIR}")
    print(f"Output will be saved to: {STRAINS_DIR}")
    print()
    process_odb_files(SIMULATIONS_DIR, STRAINS_DIR)


if __name__ == '__main__':
    main()
