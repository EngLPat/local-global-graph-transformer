import os
from abaqus import *
from abaqusConstants import *
import odbAccess

def extract_e11_strains(odb_file):
    """Extract transformed E11 strains from the specified ODB file."""
    try:
        # Open the ODB file
        odb = odbAccess.openOdb(odb_file)
        step = odb.steps.values()[-1]  # Get the last step
        frame = step.frames[-1]  # Get the last frame
        strain_field = frame.fieldOutputs['E']  # Get the strain field
        
        # Create scratch ODB for transformation
        scratchOdb = session.ScratchOdb(odb)
        
        # Create datum coordinate system (global Cartesian)
        scratchOdb.rootAssembly.DatumCsysByThreePoints(
            name='CSYS-1', 
            coordSysType=CARTESIAN, 
            origin=(0.0, 0.0, 0.0), 
            point1=(1.0, 0.0, 0.0), 
            point2=(0.0, 1.0, 0.0)
        )
        
        # Get the datum coordinate system
        datumCsys = scratchOdb.rootAssembly.datumCsyses['CSYS-1']
        
        # Transform the strain field to the global coordinate system
        transformed_field = strain_field.getTransformedField(datumCsys=datumCsys)
        
        # Get values at integration points
        ip_field = transformed_field.getSubset(position=INTEGRATION_POINT)
        
        # Dictionary to store node-to-element connectivity and strains
        node_strains = {}  # {node_label: [list of e11 values]}
        
        # Get the assembly
        assembly = odb.rootAssembly
        
        # Iterate over all instances
        for instance_name in assembly.instances.keys():
            instance = assembly.instances[instance_name]
            
            # Get the subset of the field for this instance
            try:
                instance_field = ip_field.getSubset(region=instance)
            except:
                continue
                
            # Iterate over the integration point values
            for value in instance_field.values:
                element_label = value.elementLabel
                e11_strain = value.data[0]  # E11 is the first component
                
                # Get the element and its nodes
                try:
                    element = instance.elements[element_label - 1]  # 0-based indexing
                    element_nodes = element.connectivity
                    
                    # Add this strain value to all nodes of the element
                    for node_label in element_nodes:
                        if node_label not in node_strains:
                            node_strains[node_label] = []
                        node_strains[node_label].append(e11_strain)
                except:
                    continue
        
        # Average the strains for each node
        e11_strains = {}
        for node_label, strain_list in node_strains.items():
            e11_strains[node_label] = sum(strain_list) / len(strain_list)

        # Close the ODB file
        odb.close()

        # Convert dictionary to list of tuples and sort by node label
        sorted_strains = sorted(e11_strains.items())

        return sorted_strains

    except Exception as e:
        print(f"Error processing {odb_file}: {e}")
        import traceback
        traceback.print_exc()
        return []
    
def save_strains_to_file(strains, output_file):
    """Save the strains vector to a text file."""
    with open(output_file, 'w') as f:
        for node_label, strain in strains:
            f.write(f"{node_label}, {strain}\n")
    print(f"Saved strains to {output_file}")

def process_odb_files(input_folder, output_folder):
    """Process all ODB files in the specified folder and extract E11 strains."""
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
        
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".odb"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_path}...")

            # Extract E11 strains from the ODB file
            e11_strains = extract_e11_strains(file_path)

            # Generate the output file name for the strains
            base_name = os.path.splitext(file_name)[0]
            output_file = os.path.join(output_folder, f"E11_{base_name}.txt")

            # Save the strains to a text file
            save_strains_to_file(e11_strains, output_file)

# Usage
input_folder = os.path.join(".", "ODBsONLY")
output_folder = os.path.join(".", "strains")

process_odb_files(input_folder, output_folder)