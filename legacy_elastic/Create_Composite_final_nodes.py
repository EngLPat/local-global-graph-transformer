import sys

# --- Start Diagnostic Code ---
print("--- Diagnostic Info from Abaqus Script ---")
print(f"Python executable being used: {sys.executable}")
print(f"sys.path BEFORE adding custom path:")
for p in sys.path:
    print(f"  {p}")

# Your custom path for installed packages
custom_install_path = '/home/lpatrign/abaqus_python_packages'

# Check if the custom path exists
import os
if not os.path.isdir(custom_install_path):
    print(f"WARNING: Custom installation path DOES NOT EXIST: {custom_install_path}")
else:
    print(f"Custom installation path exists: {custom_install_path}")

# Create INPs directory if it doesn't exist
if not os.path.exists('./INPs'):
    os.makedirs('./INPs')
    print("Created ./INPs directory")
    
sys.path.append(custom_install_path)

print(f"sys.path AFTER adding custom path:")
for p in sys.path:
    print(f"  {p}")

try:
    import torch
    print(f"SUCCESS: torch imported from Abaqus script!")
    print(f"Torch version: {torch.__version__}")
except ImportError as e:
    print(f"FAILURE: Could NOT import torch after adding path. Error: {e}")
    print(f"Current sys.path (where Python searched for torch):")
    for p in sys.path:
        print(f"  {p}")
    sys.exit(1) # Exit the script if torch import fails

print("--- End Diagnostic Info ---")

# Import all required modules in Abaqus
from abaqus import *
from abaqusConstants import *
from caeModules import *
import numpy as np
import torch
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

# Plate parameters
min_length = 100  # Minimum length of the plate (m)
max_length = 200  # Maximum length of the plate (m)

# Hole parameters (radius and location)
min_hole_radius = 10  # Minimum hole radius (m)
max_hole_radius = 20  # Maximum hole radius (m)
radius_tolerance = 30   # Ensures that the hole has a minimum distance ot the edges.

# Adjust these parameters to get 500 total samples
N = 25  # 25 different hole position configurations
M = 20  # 20 different displacement values

# Generate configurations using np.linspace
def generate_configurations(N, min_pos=0.3, max_pos=0.7):
    configurations = []
    hole_x_factors = np.linspace(min_pos, max_pos, num=N)
    hole_y_factors = np.linspace(min_pos, max_pos, num=N)
    for x, y in zip(hole_x_factors, hole_y_factors):
        configurations.append({
            "hole_x_factor": x, 
            "hole_y_factor": y, 
            "hole_radius_factor": 1  # Assuming constant factor for simplicity
        })
    return configurations

configurations = generate_configurations(N)

# Function to generate plate geometry based on configurations
def generate_plate_geometry(config):

    L = np.random.uniform(min_length, max_length)
    W = np.random.uniform(min_length, max_length)
    
    hole_radius = np.random.uniform(min_hole_radius, max_hole_radius)
    hole_x = L * config["hole_x_factor"]
    hole_y = W * config["hole_y_factor"]

    # Debug prints to check the values
    print(f"L: {L}, W: {W}, hole_radius: {hole_radius}, hole_x: {hole_x}, hole_y: {hole_y}")

    return L, W, hole_radius, hole_x, hole_y

displacement_values = np.linspace(1, 2, num=M)  # 10 displacement loads

geometry_data = []

# def Create_Plates_rec(L, W, hole_radius, hole_x, hole_y, RHO, E, poisson, SY, SU, epsU, seed_size, i,displacement):
def Create_Plates_rec(L, W, hole_radius, hole_x, hole_y, displacement, i, jobname):
        
        # New model
        mymodel = mdb.models['Model-1']

        #sketching
        s1 = mymodel.ConstrainedSketch(name='__profile__', 
            sheetSize=500.0)
        g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
        s1.setPrimaryObject(option=STANDALONE)
        s1.rectangle(point1=(0, 0), point2=(L, W))
        s1.CircleByCenterPerimeter(center=(hole_x, hole_y), point1=(hole_x, hole_y+hole_radius))

        p = mymodel.Part(name='PLATE', dimensionality=THREE_D, 
            type=DEFORMABLE_BODY)
        p = mymodel.parts['PLATE']
        print(f"L: {L}, W: {W}, hole_radius: {hole_radius}, hole_x: {hole_x}, hole_y: {hole_y}")
        print(f"Bounds for hole_x: [{hole_radius + radius_tolerance}, {L - hole_radius - radius_tolerance}]")
        print(f"Bounds for hole_y: [{hole_radius + radius_tolerance}, {W - hole_radius - radius_tolerance}]")

        p.BaseShell(sketch=s1)
        s1.unsetPrimaryObject()

        # Partitioning            
        p = mymodel.parts['PLATE']
        session.viewports['Viewport: 1'].setValues(displayedObject=p)
        del mymodel.sketches['__profile__']
        p = mymodel.parts['PLATE']
        f1, e, d1 = p.faces, p.edges, p.datums
        t = p.MakeSketchTransform(sketchPlane=f1[0], sketchUpEdge=e[3], 
            sketchPlaneSide=SIDE1, sketchOrientation=TOP, origin=(0.0, 0.0, 0.0))
        # t = p.MakeSketchTransform(sketchPlane=f1[0], sketchUpEdge=e[3], 
        #     sketchPlaneSide=SIDE1, sketchOrientation=TOP, origin=(90.147591, 
        #     60.295181, 0.0))
        s1 = mymodel.ConstrainedSketch(name='__profile__', 
            sheetSize=432.66, gridSpacing=10.81, transform=t)
        g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
        s1.setPrimaryObject(option=SUPERIMPOSE)
        p = mymodel.parts['PLATE']
        p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)

        s1.Line(point1=(hole_x, W), point2=(hole_x, hole_y+hole_radius))
        s1.VerticalConstraint(entity=g[7], addUndoState=False)
        s1.PerpendicularConstraint(entity1=g[5], entity2=g[7], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[6], entity2=g[5], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[7], entity2=g[2], addUndoState=False)

        s1.Line(point1=(L, hole_y), point2=(hole_x+hole_radius, hole_y))
        s1.HorizontalConstraint(entity=g[8], addUndoState=False)
        s1.PerpendicularConstraint(entity1=g[4], entity2=g[8], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[8], entity2=g[4], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[9], entity2=g[2], addUndoState=False)

        s1.Line(point1=(hole_x, 0), point2=(hole_x, hole_y-hole_radius))
        s1.VerticalConstraint(entity=g[9], addUndoState=False)
        s1.PerpendicularConstraint(entity1=g[3], entity2=g[9], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[10], entity2=g[3], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[11], entity2=g[2], addUndoState=False)

        s1.Line(point1=(0, hole_y), point2=(hole_x-hole_radius, hole_y))
        s1.HorizontalConstraint(entity=g[10], addUndoState=False)
        s1.PerpendicularConstraint(entity1=g[6], entity2=g[10], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[12], entity2=g[6], addUndoState=False)
        # s1.CoincidentConstraint(entity1=v[13], entity2=g[2], addUndoState=False)
        s1.DistanceDimension(entity1=g[10], entity2=g[3], textPoint=(-49.1842883876953, 
            -43.7521054384766), value=hole_y)
        s1.DistanceDimension(entity1=g[8], entity2=g[3], textPoint=(20.285758609375, 
            -53.3854977724609), value=hole_y)
        s1.DistanceDimension(entity1=g[7], entity2=g[6], textPoint=(80.0, 0.0), 
            value=hole_x)
        s1.DistanceDimension(entity1=g[9], entity2=g[6], textPoint=(-79.0886005214844, 
            -87.3317410585937), value=hole_x)
        p = mymodel.parts['PLATE']
        f = p.faces
        pickedFaces = f.getSequenceFromMask(mask=('[#1 ]', ), )
        e1, d2 = p.edges, p.datums
        p.PartitionFaceBySketch(sketchUpEdge=e1[3], faces=pickedFaces, 
            sketchOrientation=TOP, sketch=s1)
        s1.unsetPrimaryObject()

        #material
        mymodel.Material(name='CFRP')
        mymodel.materials['CFRP'].Elastic(type=ENGINEERING_CONSTANTS, 
            table=((150000.0, 9000.0, 9000.0, 0.34, 0.34, 0.4, 5000.0, 5000.0, 5000.0), ))
        # layupOrientation = None
        # f = p.faces
        # faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
        # region1 = regionToolset.Region(faces=faces)
        # f = p.faces
        # faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
        # region2 = regionToolset.Region(faces=faces)
        # f = p.faces
        # faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
        # region3 = regionToolset.Region(faces=faces)

        # f = p.faces
        # faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
        # region4 = regionToolset.Region(faces=faces)

        layupOrientation = None
        p = mdb.models['Model-1'].parts['PLATE']
        f = p.faces
        faces = f.getSequenceFromMask(mask=('[#f ]', ), )
        region1 = regionToolset.Region(faces=faces)
        p = mdb.models['Model-1'].parts['PLATE']
        f = p.faces
        faces = f.getSequenceFromMask(mask=('[#f ]', ), )
        region2 = regionToolset.Region(faces=faces)
        p = mdb.models['Model-1'].parts['PLATE']
        f = p.faces
        faces = f.getSequenceFromMask(mask=('[#f ]', ), )
        region3 = regionToolset.Region(faces=faces)
        p = mdb.models['Model-1'].parts['PLATE']
        f = p.faces
        faces = f.getSequenceFromMask(mask=('[#f ]', ), )
        region4 = regionToolset.Region(faces=faces)

        #composite layup
        compositeLayup = mymodel.parts['PLATE'].CompositeLayup(
            name='CompositeLayup-1', description='', elementType=SHELL, 
            offsetType=MIDDLE_SURFACE, symmetric=True, 
            thicknessAssignment=FROM_SECTION)
        compositeLayup.Section(preIntegrate=OFF, integrationRule=SIMPSON, 
            thicknessType=UNIFORM, poissonDefinition=DEFAULT, temperature=GRADIENT, 
            useDensity=OFF)
        compositeLayup.ReferenceOrientation(orientationType=GLOBAL, localCsys=None, 
            fieldName='', additionalRotationType=ROTATION_NONE, angle=0.0, 
            axis=AXIS_3)
        compositeLayup.suppress()
        compositeLayup.CompositePly(suppressed=False, plyName='Ply-1', region=region1, 
            material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=0.25, 
            orientationType=SPECIFY_ORIENT, orientationValue=0.0, 
            additionalRotationType=ROTATION_NONE, additionalRotationField='', 
            axis=AXIS_3, angle=0.0, numIntPoints=3)
        compositeLayup.CompositePly(suppressed=False, plyName='Ply-2', region=region2, 
            material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=0.25, 
            orientationType=SPECIFY_ORIENT, orientationValue=45.0, 
            additionalRotationType=ROTATION_NONE, additionalRotationField='', 
            axis=AXIS_3, angle=0.0, numIntPoints=3)
        compositeLayup.CompositePly(suppressed=False, plyName='Ply-3', region=region3, 
            material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=0.25, 
            orientationType=SPECIFY_ORIENT, orientationValue=90.0, 
            additionalRotationType=ROTATION_NONE, additionalRotationField='', 
            axis=AXIS_3, angle=0.0, numIntPoints=3)
        compositeLayup.CompositePly(suppressed=False, plyName='Ply-4', region=region4, 
            material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=0.25, 
            orientationType=SPECIFY_ORIENT, orientationValue=-45.0, 
            additionalRotationType=ROTATION_NONE, additionalRotationField='', 
            axis=AXIS_3, angle=0.0, numIntPoints=3)
        compositeLayup.resume()

        a = mymodel.rootAssembly
        a1 = mymodel.rootAssembly
        a1.DatumCsysByDefault(CARTESIAN)

        a1.Instance(name='Plate-1', part=p, dependent=ON)
        mymodel.StaticStep(name='Step-1', previous='Initial')
        e1 = a.instances['Plate-1'].edges
        edges1 = e1.getSequenceFromMask(mask=('[#4040 ]', ), )
        region = a.Set(edges=edges1, name='Set-1')

        #BCs
        mymodel.EncastreBC(name='BC-1', createStepName='Step-1', 
            region=region, localCsys=None)
        e1 = a.instances['Plate-1'].edges
        edges1 = e1.getSequenceFromMask(mask=('[#808 ]', ), )
        region = a.Set(edges=edges1, name='Set-2')
        mymodel.DisplacementBC(name='DISPLACEMENT', 
            createStepName='Step-1', region=region, u1=displacement, u2=UNSET, u3=UNSET, 
            ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
            distributionType=UNIFORM, fieldName='', localCsys=None)

        #Meshing
        elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD, 
            secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
        elemType2 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
        f = p.faces
        faces = f.getSequenceFromMask(mask=('[#f ]', ), )
        pickedRegions =(faces, )
        p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
        pickedRegions = f.getSequenceFromMask(mask=('[#f ]', ), )
        p.setMeshControls(regions=pickedRegions, elemShape=TRI)
        # p.seedPart(size=4.0, deviationFactor=0.1, minSizeFactor=0.1)
        ### mine startfrom math import pi

        expected_number_of_elements = 1000

        def calculate_seed_size(L, W, hole_radius, expected_number_of_elements):
            total_area = L * W
            hole_area = pi * (hole_radius ** 2)
            effective_area = total_area - hole_area
            target_element_area = effective_area / expected_number_of_elements
            seed_size = (target_element_area)**0.5  # Approximating that the element is square
            return seed_size
        
        seed_size = calculate_seed_size(L, W, hole_radius, expected_number_of_elements)

        p.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
        # e = p.edges
        # pickedEdges = e.getSequenceFromMask(mask=('[#1e ]', ), )
        # p.seedEdgeByNumber(edges=pickedEdges, number=20, constraint=FINER)
        # e = p.edges
        # pickedEdges = e.getSequenceFromMask(mask=('[#1 ]', ), )
        # p.seedEdgeByNumber(edges=pickedEdges, number=20, constraint=FINER)
        ### mine start
        p.generateMesh()

        # Cheese :) (taking image)
        session.viewports['Viewport: 1'].setValues(displayedObject=p)
        session.printToFile(
        fileName='./IMGs/'+jobname, 
        format=PNG, canvasObjects=(session.viewports['Viewport: 1'],))

        #job
        mymodel.FieldOutputRequest(name='F-Output-1', 
        createStepName='Step-1', variables=('E', ), position=NODES)
        mdb.Job(name=jobname, model='Model-1', description='', type=ANALYSIS, 
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
            scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
            multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
        
        # Write the Inp File
        mdb.jobs[jobname].writeInput(consistencyChecking=OFF)

        inp_filename = jobname + '.inp'
        if os.path.exists(inp_filename):
            # Make sure the target directory exists
            if not os.path.exists('./INPs'):
                os.makedirs('./INPs')
            
            # Move the file
            shutil.move(inp_filename, os.path.join('./INPs', inp_filename))
            print(f"Moved {inp_filename} to ./INPs directory")
        else:
            print(f"Warning: {inp_filename} not found after writing")
# Generate samples
def generate_samples():
    """Generates geometry and displacement data."""
    sample_id = 0
    for config in configurations:
        L, W, hole_radius, hole_x, hole_y = generate_plate_geometry(config)
        for displacement in displacement_values:
            jobname = f"Plate_{sample_id}"
            geometry_data.append([L, W, hole_radius, hole_x, hole_y, displacement])
            Create_Plates_rec(L, W, hole_radius, hole_x, hole_y, displacement, sample_id, jobname)
            sample_id += 1
    print(f"Total models created: {sample_id}")

generate_samples()

# Convert to PyTorch tensor
geometry_tensor = torch.tensor(geometry_data, dtype=torch.float32)

# Save the tensor to a file
torch.save(geometry_tensor, "plate_geometry_data.pt")