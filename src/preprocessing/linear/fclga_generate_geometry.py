"""Generate Abaqus geometry for elastic fabric composite plates with open holes.

This script generates parametric plate geometries with:
- Random plate dimensions (L, W)
- Random hole radius and position
- Elastic fabric composite material (orthotropic properties)
- Four-ply symmetric layup [0/45/90/-45]s
- Static implicit analysis
- Displacement-controlled loading

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import sys
import os

# --- Abaqus Python Environment Setup ---
print("=" * 80)
print("ELASTIC FABRIC COMPOSITE GEOMETRY GENERATION")
print("=" * 80)
print(f"Python executable: {sys.executable}")

# Custom path for PyTorch in Abaqus environment
custom_install_path = '/home/lpatrign/abaqus_python_packages'

if not os.path.isdir(custom_install_path):
    print(f"WARNING: Custom installation path not found: {custom_install_path}")
else:
    print(f"✓ Custom installation path exists: {custom_install_path}")
    sys.path.append(custom_install_path)

# Setup organized paths
from pathlib import Path
PROJECT_ROOT = Path.cwd()
GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "linear" / "geometry"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "linear" / "geometry_images"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "linear"

# Create output directories
GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Geometry output: {GEOMETRY_DIR}")
print(f"✓ Images output: {IMAGES_DIR}")

# Import PyTorch
try:
    import torch
    print(f"✓ PyTorch imported (version {torch.__version__})")
except ImportError as e:
    print(f"✗ Could NOT import torch: {e}")
    sys.exit(1)

# Import Abaqus modules
from abaqus import *
from abaqusConstants import *
from caeModules import *
import numpy as np
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import shutil
from math import pi

print("✓ Abaqus modules imported")
print("=" * 80)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Plate geometry bounds
MIN_LENGTH = 100.0  # mm
MAX_LENGTH = 200.0  # mm

# Hole parameters
MIN_HOLE_RADIUS = 10.0  # mm
MAX_HOLE_RADIUS = 20.0  # mm
RADIUS_TOLERANCE = 30.0  # Minimum distance from hole to plate edges (mm)

# Sampling configuration
N_HOLE_CONFIGS = 25  # Number of unique hole position configurations
N_DISPLACEMENTS = 20  # Number of displacement values per configuration
# Total samples: 25 × 20 = 500

# Displacement range (elastic regime)
MIN_DISPLACEMENT = 1.0  # mm
MAX_DISPLACEMENT = 2.0  # mm

# Mesh density
EXPECTED_NUM_ELEMENTS = 1000

# Random seed for reproducibility
GEOMETRY_SEED = 42

# Elastic fabric composite material properties (orthotropic)
FABRIC_E1 = 150000.0  # MPa (fiber direction)
FABRIC_E2 = 9000.0    # MPa (transverse direction)
FABRIC_E3 = 9000.0    # MPa (through-thickness)
FABRIC_NU12 = 0.34    # Poisson's ratio 12
FABRIC_NU13 = 0.34    # Poisson's ratio 13
FABRIC_NU23 = 0.4     # Poisson's ratio 23
FABRIC_G12 = 5000.0   # MPa (in-plane shear)
FABRIC_G13 = 5000.0   # MPa (shear 13)
FABRIC_G23 = 5000.0   # MPa (shear 23)

# Layup configuration
PLY_THICKNESS = 0.25  # mm per ply
PLY_ORIENTATIONS = [0.0, 45.0, 90.0, -45.0]  # degrees

print("Configuration:")
print(f"  Plate dimensions: {MIN_LENGTH}-{MAX_LENGTH} mm")
print(f"  Hole radius: {MIN_HOLE_RADIUS}-{MAX_HOLE_RADIUS} mm")
print(f"  Displacement range: {MIN_DISPLACEMENT}-{MAX_DISPLACEMENT} mm")
print(f"  Hole configurations: {N_HOLE_CONFIGS}")
print(f"  Displacements per config: {N_DISPLACEMENTS}")
print(f"  Total samples: {N_HOLE_CONFIGS * N_DISPLACEMENTS}")
print(f"  Random seed: {GEOMETRY_SEED}")
print(f"  Material: Elastic fabric composite (orthotropic)")
print(f"  Layup: [{'/'.join(map(str, map(int, PLY_ORIENTATIONS)))}]s (symmetric)")
print("=" * 80)

# =============================================================================
# FUNCTIONS
# =============================================================================

def generate_configurations(N, min_pos=0.3, max_pos=0.7):
    """Generate hole position configurations.
    
    Creates N configurations with hole center positions varying across
    the plate dimensions using linspace.
    
    Args:
        N: Number of configurations
        min_pos: Minimum position factor (fraction of plate dimension)
        max_pos: Maximum position factor (fraction of plate dimension)
    
    Returns:
        List of configuration dictionaries
    """
    configurations = []
    hole_x_factors = np.linspace(min_pos, max_pos, num=N)
    hole_y_factors = np.linspace(min_pos, max_pos, num=N)
    
    for x, y in zip(hole_x_factors, hole_y_factors):
        configurations.append({
            "hole_x_factor": x,
            "hole_y_factor": y
        })
    
    return configurations


def generate_plate_geometry(config):
    """Generate random plate geometry parameters.
    
    Args:
        config: Configuration dictionary with hole position factors
    
    Returns:
        Tuple: (L, W, hole_radius, hole_x, hole_y)
    """
    L = np.random.uniform(MIN_LENGTH, MAX_LENGTH)
    W = np.random.uniform(MIN_LENGTH, MAX_LENGTH)
    hole_radius = np.random.uniform(MIN_HOLE_RADIUS, MAX_HOLE_RADIUS)
    hole_x = L * config["hole_x_factor"]
    hole_y = W * config["hole_y_factor"]
    
    return L, W, hole_radius, hole_x, hole_y


def calculate_seed_size(L, W, hole_radius, expected_num_elements):
    """Calculate mesh seed size for target element count.
    
    Args:
        L: Plate length
        W: Plate width
        hole_radius: Hole radius
        expected_num_elements: Target number of elements
    
    Returns:
        float: Seed size for meshing
    """
    total_area = L * W
    hole_area = pi * (hole_radius ** 2)
    effective_area = total_area - hole_area
    target_element_area = effective_area / expected_num_elements
    seed_size = target_element_area ** 0.5  # Assuming square elements
    return seed_size


def create_plate_model(L, W, hole_radius, hole_x, hole_y, displacement, sample_id, jobname):
    """Create complete Abaqus model for a plate with hole.
    
    Generates geometry, material, layup, boundary conditions, mesh, and job.
    
    Args:
        L: Plate length
        W: Plate width
        hole_radius: Hole radius
        hole_x: Hole center x-coordinate
        hole_y: Hole center y-coordinate
        displacement: Applied displacement
        sample_id: Sample number
        jobname: Job name for output files
    """
    mymodel = mdb.models['Model-1']
    
    # =========================================================================
    # GEOMETRY
    # =========================================================================
    
    # Create sketch
    s1 = mymodel.ConstrainedSketch(name='__profile__', sheetSize=500.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.rectangle(point1=(0, 0), point2=(L, W))
    s1.CircleByCenterPerimeter(center=(hole_x, hole_y), point1=(hole_x, hole_y + hole_radius))
    
    # Create part
    p = mymodel.Part(name='PLATE', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p = mymodel.parts['PLATE']
    p.BaseShell(sketch=s1)
    s1.unsetPrimaryObject()
    
    # Partitioning for structured mesh
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mymodel.sketches['__profile__']
    
    f1, e, d1 = p.faces, p.edges, p.datums
    t = p.MakeSketchTransform(sketchPlane=f1[0], sketchUpEdge=e[3],
                              sketchPlaneSide=SIDE1, sketchOrientation=TOP,
                              origin=(0.0, 0.0, 0.0))
    s1 = mymodel.ConstrainedSketch(name='__profile__', sheetSize=432.66,
                                   gridSpacing=10.81, transform=t)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    
    # Create partition lines around hole
    s1.Line(point1=(hole_x, W), point2=(hole_x, hole_y + hole_radius))
    s1.VerticalConstraint(entity=g[7], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[5], entity2=g[7], addUndoState=False)
    
    s1.Line(point1=(L, hole_y), point2=(hole_x + hole_radius, hole_y))
    s1.HorizontalConstraint(entity=g[8], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[4], entity2=g[8], addUndoState=False)
    
    s1.Line(point1=(hole_x, 0), point2=(hole_x, hole_y - hole_radius))
    s1.VerticalConstraint(entity=g[9], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[3], entity2=g[9], addUndoState=False)
    
    s1.Line(point1=(0, hole_y), point2=(hole_x - hole_radius, hole_y))
    s1.HorizontalConstraint(entity=g[10], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[6], entity2=g[10], addUndoState=False)
    
    # Add dimensions
    s1.DistanceDimension(entity1=g[10], entity2=g[3], 
                        textPoint=(-49.18, -43.75), value=hole_y)
    s1.DistanceDimension(entity1=g[8], entity2=g[3],
                        textPoint=(20.29, -53.39), value=hole_y)
    s1.DistanceDimension(entity1=g[7], entity2=g[6],
                        textPoint=(80.0, 0.0), value=hole_x)
    s1.DistanceDimension(entity1=g[9], entity2=g[6],
                        textPoint=(-79.09, -87.33), value=hole_x)
    
    # Apply partitions
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#1 ]',),)
    e1, d2 = p.edges, p.datums
    p.PartitionFaceBySketch(sketchUpEdge=e1[3], faces=pickedFaces,
                           sketchOrientation=TOP, sketch=s1)
    s1.unsetPrimaryObject()
    
    # =========================================================================
    # MATERIAL AND LAYUP
    # =========================================================================
    
    # Define elastic fabric composite material
    mymodel.Material(name='CFRP')
    mymodel.materials['CFRP'].Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((FABRIC_E1, FABRIC_E2, FABRIC_E3,
                FABRIC_NU12, FABRIC_NU13, FABRIC_NU23,
                FABRIC_G12, FABRIC_G13, FABRIC_G23),)
    )
    
    # Define regions for each face
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#f ]',),)
    region1 = regionToolset.Region(faces=faces)
    region2 = regionToolset.Region(faces=faces)
    region3 = regionToolset.Region(faces=faces)
    region4 = regionToolset.Region(faces=faces)
    
    # Create composite layup [0/45/90/-45]s (symmetric)
    compositeLayup = p.CompositeLayup(
        name='CompositeLayup-1',
        description='Symmetric fabric layup',
        elementType=SHELL,
        offsetType=MIDDLE_SURFACE,
        symmetric=True,
        thicknessAssignment=FROM_SECTION
    )
    
    compositeLayup.Section(
        preIntegrate=OFF,
        integrationRule=SIMPSON,
        thicknessType=UNIFORM,
        poissonDefinition=DEFAULT,
        temperature=GRADIENT,
        useDensity=OFF
    )
    
    compositeLayup.ReferenceOrientation(
        orientationType=GLOBAL,
        localCsys=None,
        fieldName='',
        additionalRotationType=ROTATION_NONE,
        angle=0.0,
        axis=AXIS_3
    )
    
    compositeLayup.suppress()
    
    # Add plies
    compositeLayup.CompositePly(
        suppressed=False, plyName='Ply-1', region=region1,
        material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=PLY_THICKNESS,
        orientationType=SPECIFY_ORIENT, orientationValue=PLY_ORIENTATIONS[0],
        additionalRotationType=ROTATION_NONE, additionalRotationField='',
        axis=AXIS_3, angle=0.0, numIntPoints=3
    )
    compositeLayup.CompositePly(
        suppressed=False, plyName='Ply-2', region=region2,
        material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=PLY_THICKNESS,
        orientationType=SPECIFY_ORIENT, orientationValue=PLY_ORIENTATIONS[1],
        additionalRotationType=ROTATION_NONE, additionalRotationField='',
        axis=AXIS_3, angle=0.0, numIntPoints=3
    )
    compositeLayup.CompositePly(
        suppressed=False, plyName='Ply-3', region=region3,
        material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=PLY_THICKNESS,
        orientationType=SPECIFY_ORIENT, orientationValue=PLY_ORIENTATIONS[2],
        additionalRotationType=ROTATION_NONE, additionalRotationField='',
        axis=AXIS_3, angle=0.0, numIntPoints=3
    )
    compositeLayup.CompositePly(
        suppressed=False, plyName='Ply-4', region=region4,
        material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=PLY_THICKNESS,
        orientationType=SPECIFY_ORIENT, orientationValue=PLY_ORIENTATIONS[3],
        additionalRotationType=ROTATION_NONE, additionalRotationField='',
        axis=AXIS_3, angle=0.0, numIntPoints=3
    )
    
    compositeLayup.resume()
    
    # =========================================================================
    # ASSEMBLY
    # =========================================================================
    
    a = mymodel.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    a.Instance(name='Plate-1', part=p, dependent=ON)
    
    # =========================================================================
    # STEP (Static implicit analysis)
    # =========================================================================
    
    mymodel.StaticStep(name='Step-1', previous='Initial')
    
    # =========================================================================
    # BOUNDARY CONDITIONS
    # =========================================================================
    
    # Fixed edge (x = 0)
    e1 = a.instances['Plate-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#4040 ]',),)
    region = a.Set(edges=edges1, name='Set-1')
    mymodel.EncastreBC(name='BC-1', createStepName='Step-1',
                      region=region, localCsys=None)
    
    # Displacement-controlled edge (x = L)
    edges1 = e1.getSequenceFromMask(mask=('[#808 ]',),)
    region = a.Set(edges=edges1, name='Set-2')
    mymodel.DisplacementBC(
        name='DISPLACEMENT',
        createStepName='Step-1',
        region=region,
        u1=displacement,
        u2=UNSET,
        u3=UNSET,
        ur1=UNSET,
        ur2=UNSET,
        ur3=UNSET,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None
    )
    
    # =========================================================================
    # MESHING
    # =========================================================================
    
    # Element types
    elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD,
                              secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
    
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#f ]',),)
    pickedRegions = (faces,)
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    p.setMeshControls(regions=faces, elemShape=TRI)
    
    # Calculate seed size for target element count
    seed_size = calculate_seed_size(L, W, hole_radius, EXPECTED_NUM_ELEMENTS)
    p.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()
    
    # Save mesh visualization
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.printToFile(
        fileName=str(IMAGES_DIR / jobname),
        format=PNG,
        canvasObjects=(session.viewports['Viewport: 1'],)
    )
    
    # =========================================================================
    # JOB DEFINITION
    # =========================================================================
    
    # Request nodal strain output (E11 at nodes)
    mymodel.FieldOutputRequest(
        name='F-Output-1',
        createStepName='Step-1',
        variables=('E',),
        position=NODES
    )
    
    # Create job
    mdb.Job(
        name=jobname,
        model='Model-1',
        description='Elastic fabric composite plate with hole',
        type=ANALYSIS,
        atTime=None,
        waitMinutes=0,
        waitHours=0,
        queue=None,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        userSubroutine='',
        scratch='',
        resultsFormat=ODB,
        numThreadsPerMpiProcess=1,
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numGPUs=0
    )
    
    # Write input file
    mdb.jobs[jobname].writeInput(consistencyChecking=OFF)
    
    # Move .inp file to INPs directory
    inp_filename = jobname + '.inp'
    if os.path.exists(inp_filename):
        shutil.move(inp_filename, str(GEOMETRY_DIR / inp_filename))
        print(f"  ✓ {jobname}.inp created")
    else:
        print(f"  ✗ Warning: {inp_filename} not found")


def generate_samples():
    """Generate all sample geometries."""
    print("\nGenerating samples...")
    print("-" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(GEOMETRY_SEED)
    print(f"Using random seed: {GEOMETRY_SEED} for reproducible geometry generation")
    print("Note: To match legacy dataset, use legacy .inp files instead of regenerating")
    print("-" * 80)
    
    configurations = generate_configurations(N_HOLE_CONFIGS)
    displacement_values = np.linspace(MIN_DISPLACEMENT, MAX_DISPLACEMENT, num=N_DISPLACEMENTS)
    geometry_data = []
    
    sample_id = 0
    for config_idx, config in enumerate(configurations):
        L, W, hole_radius, hole_x, hole_y = generate_plate_geometry(config)
        
        for disp_idx, displacement in enumerate(displacement_values):
            jobname = f"Plate_{sample_id}"
            
            if sample_id % 50 == 0 or sample_id == 0:
                print(f"\nSample {sample_id + 1}/{N_HOLE_CONFIGS * N_DISPLACEMENTS}:")
                print(f"  Config {config_idx + 1}/{N_HOLE_CONFIGS}, Disp {disp_idx + 1}/{N_DISPLACEMENTS}")
                print(f"  L={L:.2f}, W={W:.2f}, r={hole_radius:.2f}, disp={displacement:.3f}")
            
            geometry_data.append([L, W, hole_radius, hole_x, hole_y, displacement])
            create_plate_model(L, W, hole_radius, hole_x, hole_y, displacement, sample_id, jobname)
            
            sample_id += 1
    
    print(f"\n✓ Total models created: {sample_id}")
    
    # Convert to PyTorch tensor and save to processed directory
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    geometry_tensor = torch.tensor(geometry_data, dtype=torch.float32)
    geometry_data_path = DATA_PROCESSED / "plate_geometry_data.pt"
    torch.save(geometry_tensor, str(geometry_data_path))
    print(f"✓ Geometry data saved to {geometry_data_path}")
    print("=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    generate_samples()
    print("\nELASTIC GEOMETRY GENERATION COMPLETE")
    print("=" * 80)
