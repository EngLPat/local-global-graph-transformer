"""Generate parametric plate geometries with holes for FEA analysis.

Creates Abaqus input files (.inp) for plates with circular holes with various
sizes, positions and under different loading conditions.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import os
import pickle
import shutil
from math import pi
from pathlib import Path

import numpy as np

# Setup paths
PROJECT_ROOT = Path.cwd()
GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "nonlinear" / "geometry"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "nonlinear" / "geometry_images"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "nonlinear"

# Create directories
for directory in [GEOMETRY_DIR, IMAGES_DIR, DATA_PROCESSED]:
    directory.mkdir(parents=True, exist_ok=True)


torch_available = False
try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False

# Import Abaqus modules
import mesh  # noqa: F401, E402
import regionToolset  # noqa: F401, E402
from abaqus import *  # noqa: F403, E402
from abaqusConstants import *  # noqa: F403, E402
from caeModules import *  # noqa: F403, E402

# ============================================================================
# Configuration Parameters
# ============================================================================
# These parameters control geometry generation and material properties.
# Default values match config/defaults.yaml for consistency.
# Edit these values directly if you need different configurations.
# (Cannot load YAML here because Abaqus uses its own Python interpreter)

# Geometry generation parameters
MIN_LENGTH = 100.0  # Plate minimum length (mm)
MAX_LENGTH = 200.0  # Plate maximum length (mm)
MIN_HOLE_RADIUS = 10.0  # Minimum hole radius (mm)
MAX_HOLE_RADIUS = 20.0  # Maximum hole radius (mm)
RADIUS_TOLERANCE = 30.0  # Minimum distance from hole to edges (mm)
EXPECTED_ELEMENTS = 1000  # Target mesh element count

# Dataset size parameters
N_HOLE_CONFIGS = 25  # Number of different hole positions
N_DISPLACEMENTS = 20  # Number of different load cases

# Material properties for composite laminate
MATERIAL_DENSITY = 1.594e-09  # Material density (tonne/mm^3)
LAMINATE_LAYUP = [45.0, -45.0, -45.0, 45.0]  # Ply angles (degrees)
PLY_THICKNESS = 0.25  # Single ply thickness (mm)

# Mechanical constants for user-defined material (UMAT)
MATERIAL_CONSTANTS = (
    27500.0,
    27500.0,
    0.11,
    2900.0,
    27500.0,
    27500.0,
    0.11,
    0.0,
    604.0,
    291.0,
    604.0,
    291.0,
    75.0,
    0.0,
    0.0,
    0.0,
    20.438,
    4.6257,
    20.438,
    4.6257,
    0.3221,
    1.0,
    0.0,
    0.0,
    75.0,
    0.0008,
    0.552,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1000000000.0,
    1000000000.0,
    -1000000000.0,
    0.0,
    0.0,
    0.0,
)

# Legacy structure for compatibility with existing code
MATERIAL_PROPERTIES = {"density": MATERIAL_DENSITY, "mechanical_constants": MATERIAL_CONSTANTS}


def generate_configurations(n, min_pos=0.3, max_pos=0.7):
    """Generate hole position configurations.

    Args:
        n: Number of configurations to generate
        min_pos: Minimum position factor (0-1)
        max_pos: Maximum position factor (0-1)

    Returns:
        List of configuration dictionaries with hole position factors
    """
    configurations = []
    hole_x_factors = np.linspace(min_pos, max_pos, num=n)
    hole_y_factors = np.linspace(min_pos, max_pos, num=n)
    for x, y in zip(hole_x_factors, hole_y_factors):
        configurations.append({"hole_x_factor": x, "hole_y_factor": y, "hole_radius_factor": 1.0})
    return configurations


def generate_plate_geometry(config):
    """Generate randomized plate geometry parameters.

    Args:
        config: Configuration dictionary with hole position factors

    Returns:
        Tuple of (L, W, hole_radius, hole_x, hole_y)
    """
    L = np.random.uniform(MIN_LENGTH, MAX_LENGTH)
    W = np.random.uniform(MIN_LENGTH, MAX_LENGTH)
    hole_radius = np.random.uniform(MIN_HOLE_RADIUS, MAX_HOLE_RADIUS)
    hole_x = L * config["hole_x_factor"]
    hole_y = W * config["hole_y_factor"]

    return L, W, hole_radius, hole_x, hole_y


def calculate_seed_size(L, W, hole_radius, n_elements):
    """Calculate mesh seed size based on target element count.

    Args:
        L: Plate length
        W: Plate width
        hole_radius: Radius of the hole
        n_elements: Target number of elements

    Returns:
        Mesh seed size
    """
    total_area = L * W
    hole_area = pi * (hole_radius**2)
    effective_area = total_area - hole_area
    target_element_area = effective_area / n_elements
    return target_element_area**0.5


def create_plate_model(L, W, hole_radius, hole_x, hole_y, displacement, jobname):
    """Create complete Abaqus model for a plate with a hole.

    Generates geometry, applies material properties, defines composite layup,
    sets boundary conditions, creates mesh, and writes input file.

    Args:
        L: Plate length
        W: Plate width
        hole_radius: Hole radius
        hole_x: Hole center x-coordinate
        hole_y: Hole center y-coordinate
        displacement: Applied displacement magnitude
        jobname: Job name for the model
    """
    mymodel = mdb.models["Model-1"]  # noqa: F405

    # Create sketch for rectangular plate with circular hole
    s1 = mymodel.ConstrainedSketch(name="__profile__", sheetSize=500.0)
    g = s1.geometry
    s1.setPrimaryObject(option=STANDALONE)  # noqa: F405
    s1.rectangle(point1=(0, 0), point2=(L, W))
    s1.CircleByCenterPerimeter(center=(hole_x, hole_y), point1=(hole_x, hole_y + hole_radius))

    p = mymodel.Part(name="PLATE", dimensionality=THREE_D, type=DEFORMABLE_BODY)  # noqa: F405
    p = mymodel.parts["PLATE"]
    p.BaseShell(sketch=s1)
    s1.unsetPrimaryObject()

    # Partition face with cross-lines through hole center
    p = mymodel.parts["PLATE"]
    session.viewports["Viewport: 1"].setValues(displayedObject=p)  # noqa: F405
    del mymodel.sketches["__profile__"]
    p = mymodel.parts["PLATE"]
    f1, e, _ = p.faces, p.edges, p.datums
    t = p.MakeSketchTransform(
        sketchPlane=f1[0],
        sketchUpEdge=e[3],
        sketchPlaneSide=SIDE1,
        sketchOrientation=TOP,
        origin=(0.0, 0.0, 0.0),
    )  # noqa: F405
    s1 = mymodel.ConstrainedSketch(
        name="__profile__", sheetSize=432.66, gridSpacing=10.81, transform=t
    )
    g = s1.geometry
    s1.setPrimaryObject(option=SUPERIMPOSE)  # noqa: F405
    p = mymodel.parts["PLATE"]
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)  # noqa: F405

    # Define partition lines (vertical and horizontal through hole center)
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

    s1.DistanceDimension(
        entity1=g[10], entity2=g[3], textPoint=(-49.1842883876953, -43.7521054384766), value=hole_y
    )
    s1.DistanceDimension(
        entity1=g[8], entity2=g[3], textPoint=(20.285758609375, -53.3854977724609), value=hole_y
    )
    s1.DistanceDimension(entity1=g[7], entity2=g[6], textPoint=(80.0, 0.0), value=hole_x)
    s1.DistanceDimension(
        entity1=g[9], entity2=g[6], textPoint=(-79.0886005214844, -87.3317410585937), value=hole_x
    )

    p = mymodel.parts["PLATE"]
    f = p.faces
    pickedFaces = f.getSequenceFromMask(
        mask=("[#1 ]",),
    )
    e1 = p.edges
    p.PartitionFaceBySketch(sketchUpEdge=e1[3], faces=pickedFaces, sketchOrientation=TOP, sketch=s1)  # noqa: F405
    s1.unsetPrimaryObject()

    # Define material properties
    mymodel.Material(name="ABQ_PLY_FABRIC")
    mymodel.materials["ABQ_PLY_FABRIC"].Density(table=((1.594e-09,),))
    mymodel.materials["ABQ_PLY_FABRIC"].Depvar(deleteVar=16, n=16)
    mymodel.materials["ABQ_PLY_FABRIC"].UserMaterial(
        mechanicalConstants=(
            27500.0,
            27500.0,
            0.11,
            2900.0,
            27500.0,
            27500.0,
            0.11,
            0.0,
            604.0,
            291.0,
            604.0,
            291.0,
            75.0,
            0.0,
            0.0,
            0.0,
            20.438,
            4.6257,
            20.438,
            4.6257,
            0.3221,
            1.0,
            0.0,
            0.0,
            75.0,
            0.0008,
            0.552,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1000000000.0,
            1000000000.0,
            -1000000000.0,
            0.0,
            0.0,
            0.0,
        )
    )

    # Define composite layup with 4 plies
    p = mymodel.parts["PLATE"]
    f = p.faces
    faces = f.getSequenceFromMask(
        mask=("[#f ]",),
    )
    region1 = regionToolset.Region(faces=faces)  # noqa: F405
    region2 = regionToolset.Region(faces=faces)  # noqa: F405
    region3 = regionToolset.Region(faces=faces)  # noqa: F405
    region4 = regionToolset.Region(faces=faces)  # noqa: F405

    compositeLayup = p.CompositeLayup(
        name="CompositeLayup-1",
        description="",
        elementType=SHELL,  # noqa: F405
        offsetType=MIDDLE_SURFACE,
        symmetric=False,  # noqa: F405
        thicknessAssignment=FROM_SECTION,
    )  # noqa: F405
    compositeLayup.Section(
        preIntegrate=OFF,
        integrationRule=SIMPSON,  # noqa: F405
        thicknessType=UNIFORM,
        poissonDefinition=DEFAULT,
        temperature=GRADIENT,  # noqa: F405
        useDensity=OFF,
    )  # noqa: F405
    compositeLayup.ReferenceOrientation(
        orientationType=GLOBAL,
        localCsys=None,  # noqa: F405
        fieldName="",
        additionalRotationType=ROTATION_NONE,
        angle=0.0,  # noqa: F405
        axis=AXIS_3,
    )  # noqa: F405
    compositeLayup.suppress()
    compositeLayup.CompositePly(
        suppressed=False,
        plyName="Ply-1",
        region=region1,
        material="ABQ_PLY_FABRIC",
        thicknessType=SPECIFY_THICKNESS,  # noqa: F405
        thickness=0.25,
        orientationType=SPECIFY_ORIENT,
        orientationValue=45.0,  # noqa: F405
        additionalRotationType=ROTATION_NONE,
        additionalRotationField="",  # noqa: F405
        axis=AXIS_3,
        angle=0.0,
        numIntPoints=1,
    )  # noqa: F405
    compositeLayup.CompositePly(
        suppressed=False,
        plyName="Ply-2",
        region=region2,
        material="ABQ_PLY_FABRIC",
        thicknessType=SPECIFY_THICKNESS,  # noqa: F405
        thickness=0.25,
        orientationType=SPECIFY_ORIENT,
        orientationValue=-45.0,  # noqa: F405
        additionalRotationType=ROTATION_NONE,
        additionalRotationField="",  # noqa: F405
        axis=AXIS_3,
        angle=0.0,
        numIntPoints=1,
    )  # noqa: F405
    compositeLayup.CompositePly(
        suppressed=False,
        plyName="Ply-3",
        region=region3,
        material="ABQ_PLY_FABRIC",
        thicknessType=SPECIFY_THICKNESS,  # noqa: F405
        thickness=0.25,
        orientationType=SPECIFY_ORIENT,
        orientationValue=-45.0,  # noqa: F405
        additionalRotationType=ROTATION_NONE,
        additionalRotationField="",  # noqa: F405
        axis=AXIS_3,
        angle=0.0,
        numIntPoints=1,
    )  # noqa: F405
    compositeLayup.CompositePly(
        suppressed=False,
        plyName="Ply-4",
        region=region4,
        material="ABQ_PLY_FABRIC",
        thicknessType=SPECIFY_THICKNESS,  # noqa: F405
        thickness=0.25,
        orientationType=SPECIFY_ORIENT,
        orientationValue=45.0,  # noqa: F405
        additionalRotationType=ROTATION_NONE,
        additionalRotationField="",  # noqa: F405
        axis=AXIS_3,
        angle=0.0,
        numIntPoints=1,
    )  # noqa: F405
    compositeLayup.resume()

    # Create assembly
    a = mymodel.rootAssembly
    a1 = mymodel.rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)  # noqa: F405
    a1.Instance(name="Plate-1", part=p, dependent=ON)  # noqa: F405

    # Define explicit dynamics step
    mymodel.ExplicitDynamicsStep(
        name="Step-1", previous="Initial", timePeriod=0.01, nlgeom=OFF, improvedDtMethod=ON
    )  # noqa: F405

    # Apply boundary conditions
    e1 = a.instances["Plate-1"].edges
    edges1 = e1.getSequenceFromMask(
        mask=("[#4040 ]",),
    )
    region = a.Set(edges=edges1, name="Set-1")
    mymodel.EncastreBC(name="BC-1", createStepName="Step-1", region=region, localCsys=None)
    e1 = a.instances["Plate-1"].edges
    e1 = a.instances["Plate-1"].edges
    edges1 = e1.getSequenceFromMask(
        mask=("[#808 ]",),
    )
    region = a.Set(edges=edges1, name="Set-2")

    mymodel.TabularAmplitude(
        name="Ramp",
        timeSpan=TOTAL,  # noqa: F405
        smooth=SOLVER_DEFAULT,
        data=((0.0, 0.0), (0.01, 1.0)),
    )  # noqa: F405

    mymodel.DisplacementBC(
        name="BC-2",
        createStepName="Step-1",
        region=region,
        u1=displacement,
        u2=UNSET,
        u3=UNSET,
        ur1=UNSET,
        ur2=UNSET,  # noqa: F405
        ur3=UNSET,
        amplitude="Ramp",
        fixed=OFF,
        distributionType=UNIFORM,  # noqa: F405
        fieldName="",
        localCsys=None,
    )

    # Define mesh properties
    elemType1 = mesh.ElemType(
        elemCode=S4R,
        elemLibrary=STANDARD,  # noqa: F405
        secondOrderAccuracy=OFF,
        hourglassControl=DEFAULT,
    )  # noqa: F405
    elemType2 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)  # noqa: F405
    f = p.faces
    faces = f.getSequenceFromMask(
        mask=("[#f ]",),
    )
    pickedRegions = (faces,)
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    pickedRegions = f.getSequenceFromMask(
        mask=("[#f ]",),
    )
    p.setMeshControls(regions=pickedRegions, elemShape=TRI)  # noqa: F405

    # Calculate seed size and generate mesh
    seed_size = calculate_seed_size(L, W, hole_radius, EXPECTED_ELEMENTS)
    p.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()

    # Save mesh visualization (optional)
    try:
        session.viewports["Viewport: 1"].setValues(displayedObject=p)  # noqa: F405
        image_path = str(IMAGES_DIR / jobname)
        session.printToFile(  # noqa: F405
            fileName=image_path,
            format=PNG,  # noqa: F405
            canvasObjects=(session.viewports["Viewport: 1"],),  # noqa: F405
        )
        print(f"  Saved mesh image: {image_path}.png")
    except Exception as e:
        print(f"  Warning: Could not save mesh image: {e}")

    # Configure field output and create job
    mymodel.fieldOutputRequests["F-Output-1"].setValues(
        variables=("PEEQ", "LE", "SDV"), numIntervals=1, position=NODES
    )  # noqa: F405

    mdb.Job(
        name=jobname,
        model="Model-1",
        description="",  # noqa: F405
        type=ANALYSIS,
        atTime=None,
        waitMinutes=0,
        waitHours=0,
        queue=None,  # noqa: F405
        memory=90,
        memoryUnits=PERCENTAGE,
        explicitPrecision=SINGLE,  # noqa: F405
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,  # noqa: F405
        contactPrint=OFF,
        historyPrint=OFF,
        userSubroutine="",
        scratch="",  # noqa: F405
        resultsFormat=ODB,
        numDomains=1,
        activateLoadBalancing=False,  # noqa: F405
        numThreadsPerMpiProcess=1,
        multiprocessingMode=DEFAULT,
        numCpus=1,
    )  # noqa: F405

    # Write input file and move to organized directory
    mdb.jobs[jobname].writeInput(consistencyChecking=OFF)  # noqa: F405

    inp_filename = jobname + ".inp"
    if os.path.exists(inp_filename):
        target_path = GEOMETRY_DIR / inp_filename
        shutil.move(inp_filename, str(target_path))
        print(f"  Saved: {target_path}")
    else:
        print(f"  Warning: {inp_filename} not found after writing")


def generate_samples():
    """Generate geometry and displacement data for all configurations.

    Iterates through hole position configurations and displacement values,
    creating an Abaqus model for each combination.
    """
    sample_id = 0
    for config in configurations:
        L, W, hole_radius, hole_x, hole_y = generate_plate_geometry(config)
        for displacement in displacement_values:
            jobname = f"Plate_nonlinear_{sample_id}"
            geometry_data.append([L, W, hole_radius, hole_x, hole_y, displacement])
            create_plate_model(L, W, hole_radius, hole_x, hole_y, displacement, jobname)
            sample_id += 1
    print(f"Total models created: {sample_id}")


if __name__ == "__main__":
    # Set random seed for reproducible geometry generation
    GEOMETRY_SEED = 42
    np.random.seed(GEOMETRY_SEED)
    print(f"Using random seed: {GEOMETRY_SEED} for reproducible geometry generation")
    print("Note: To match legacy dataset, use legacy .inp files instead of regenerating")

    # Initialize configuration and displacement arrays
    configurations = generate_configurations(N_HOLE_CONFIGS)
    displacement_values = np.linspace(1, 3, num=N_DISPLACEMENTS)
    geometry_data = []

    # Generate all samples
    generate_samples()

    # Save results
    geometry_tensor_path = DATA_PROCESSED / "plate_geometry_data.pt"
    geometry_pickle_path = DATA_PROCESSED / "plate_geometry_data.pkl"

    if torch_available:
        geometry_tensor = torch.tensor(geometry_data, dtype=torch.float32)
        torch.save(geometry_tensor, str(geometry_tensor_path))
        print(f"\n✓ Saved geometry tensor to {geometry_tensor_path}")
    else:
        print("\n⚠ Skipping geometry tensor save (torch not available)")
        print("Note: You can manually create this tensor later from the geometry_data if needed")

    # Always save pickle backup
    with open(str(geometry_pickle_path), "wb") as f:
        pickle.dump(geometry_data, f)
    print(f"✓ Saved geometry data backup to {geometry_pickle_path}")

    print(f"\n{'=' * 60}")
    print("GEOMETRY GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {GEOMETRY_DIR}")
    print(f"Total files created: {len(geometry_data)}")
