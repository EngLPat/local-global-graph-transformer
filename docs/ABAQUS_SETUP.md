# Abaqus Integration Guide

This document provides detailed information about the Abaqus FEA setup used to generate training data for the FCLGA GraphTransformer model.

## Overview

The project simulates composite laminate plates with circular holes under displacement loading. The model uses **nonlinear material behavior** through a User Material (UMAT) subroutine with explicit dynamics.

---

## Material Properties

### Composite Material: `ABQ_PLY_FABRIC`

The material is a **carbon fiber/epoxy fabric composite** defined using a User Material (UMAT) subroutine with 40 mechanical constants for progressive damage modeling.

#### Density
```python
Density: 1.594e-09 tonne/mm³
```

#### User Material Constants (40 parameters)

**Elastic Properties (indices 1-8):**
```python
E11 = 27500.0 MPa      # In-plane modulus (warp direction)
E22 = 27500.0 MPa      # In-plane modulus (fill direction)
nu12 = 0.11            # Poisson's ratio
G12 = 2900.0 MPa       # In-plane shear modulus
E11 = 27500.0 MPa      # Repeated for damage model
E22 = 27500.0 MPa      # Repeated for damage model
nu12 = 0.11            # Repeated for damage model
(unused) = 0.0         # Reserved parameter
```

**Strength Properties (indices 9-16):**
```python
X_T = 604.0 MPa        # Longitudinal tensile strength
X_C = 291.0 MPa        # Longitudinal compressive strength
Y_T = 604.0 MPa        # Transverse tensile strength
Y_C = 291.0 MPa        # Transverse compressive strength
S_12 = 75.0 MPa        # In-plane shear strength
(unused) = 0.0         # Reserved parameters (3x)
```

**Damage Evolution Parameters (indices 17-28):**
```python
G_Ift = 20.438 N/mm    # Mode I fracture toughness (tension)
G_Ifc = 4.6257 N/mm    # Mode I fracture toughness (compression)
G_IIft = 20.438 N/mm   # Mode II fracture toughness (tension)
G_IIfc = 4.6257 N/mm   # Mode II fracture toughness (compression)
eta_BK = 0.3221        # Benzeggagh-Kenane coefficient
n_BK = 1.0             # Benzeggagh-Kenane exponent
(unused) = 0.0         # Reserved parameters (2x)
S_12 = 75.0 MPa        # Shear strength (repeated)
epsilon_f = 0.0008     # Failure strain
alpha_d = 0.552        # Damage coefficient
(unused) = 0.0         # Reserved parameters (5x)
```

**Numerical Parameters (indices 29-40):**
```python
(unused) = 1.0                  # Reserved
viscosity_penalty = 1e9         # Viscous regularization
viscosity_penalty = 1e9         # Viscous regularization (duplicate)
penalty = -1e9                  # Numerical penalty
(unused) = 0.0                  # Reserved parameters (8x)
```

**Full UMAT Call (from code):**
```python
mymodel.materials['ABQ_PLY_FABRIC'].UserMaterial(
    mechanicalConstants=(
        27500.0, 27500.0, 0.11, 2900.0, 27500.0, 27500.0, 
        0.11, 0.0, 604.0, 291.0, 604.0, 291.0, 75.0, 0.0, 0.0, 0.0, 20.438, 
        4.6257, 20.438, 4.6257, 0.3221, 1.0, 0.0, 0.0, 75.0, 0.0008, 0.552, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1000000000.0, 1000000000.0, 
        -1000000000.0, 0.0, 0.0, 0.0
    )
)
```

#### State Variables (SDV)
```python
Number of solution-dependent state variables: 16
# Used for tracking damage variables in progressive failure model
```

---

## Composite Laminate Configuration

### Layup Definition

**4-ply symmetric quasi-isotropic laminate:**

| Ply | Thickness | Orientation | Material |
|-----|-----------|-------------|----------|
| 1   | 0.25 mm   | +45°        | ABQ_PLY_FABRIC |
| 2   | 0.25 mm   | -45°        | ABQ_PLY_FABRIC |
| 3   | 0.25 mm   | -45°        | ABQ_PLY_FABRIC |
| 4   | 0.25 mm   | +45°        | ABQ_PLY_FABRIC |

**Total thickness:** 1.0 mm  
**Layup:** [+45/-45/-45/+45]  
**Offset:** Middle surface

**Code snippet:**
```python
compositeLayup.CompositePly(
    plyName='Ply-1', region=region1, material='ABQ_PLY_FABRIC',
    thicknessType=SPECIFY_THICKNESS, thickness=0.25,
    orientationType=SPECIFY_ORIENT, orientationValue=45.0,
    numIntPoints=1
)
# ... (repeated for Ply-2, Ply-3, Ply-4)
```

---

## Geometry Parameters

### Parametric Plate Dimensions

```python
Length (L):           100-200 mm (uniform random)
Width (W):            100-200 mm (uniform random)
Hole Radius:          10-20 mm (constrained to avoid edges)
Hole Location (x):    0.3*L to 0.7*L (linspace sampling)
Hole Location (y):    0.3*W to 0.7*W (linspace sampling)
Edge Tolerance:       30 mm (minimum distance from hole to edge)
```

### Sampling Strategy

For the full dataset (500 samples):
```python
N = 25  # Hole position configurations (linspace)
M = 20  # Displacement loading values
Total samples = N × M = 500
```

For testing (10 samples):
```python
N = 2   # Hole position configurations
M = 5   # Displacement loading values
Total samples = N × M = 10
```

---

## Solver Settings

### Analysis Step Configuration

**Step Type:** Explicit Dynamics  
```python
mymodel.ExplicitDynamicsStep(
    name='Step-1',
    previous='Initial',
    timePeriod=0.01,           # 0.01 seconds (10 ms)
    nlgeom=OFF,                # Geometric nonlinearity disabled
    improvedDtMethod=ON        # Automatic time incrementation
)
```

**Why explicit dynamics?**
- Handles material nonlinearity efficiently with UMAT
- Stable for progressive damage simulations
- No convergence issues with softening behavior

---

## Boundary Conditions

### Fixed Support (Encastre)
```python
# Applied to: Left and right edges (mask '[#4040]')
# Constraints: All DOFs fixed (u1=u2=u3=ur1=ur2=ur3=0)
BC-1: EncastreBC on edges
```

### Displacement Loading
```python
# Applied to: Top and bottom edges (mask '[#808]')
# Loading: u1 = displacement (variable), u2=u3=free
# Amplitude: Ramped from 0 to 1 over 0.01s

mymodel.TabularAmplitude(
    name='Ramp',
    data=((0.0, 0.0), (0.01, 1.0))  # Linear ramp
)

mymodel.DisplacementBC(
    name='BC-2',
    u1=displacement,      # Applied displacement (varies by sample)
    amplitude='Ramp',
    distributionType=UNIFORM
)
```

**Displacement values:** Vary per sample in dataset generation

---

## Mesh Configuration

### Element Properties

**Element Type:** S4R (4-node shell, reduced integration, hourglass control)
```python
elemType1 = mesh.ElemType(
    elemCode=S4R,
    elemLibrary=STANDARD,
    secondOrderAccuracy=OFF,
    hourglassControl=DEFAULT
)
```

**Backup Element:** S3 (3-node triangular shell for irregular regions)

### Mesh Controls

**Target element count:** ~1000 elements per sample  
**Mesh shape:** Triangular (TRI)  
**Adaptive seeding:** Yes

```python
expected_number_of_elements = 1000

# Calculate adaptive seed size based on geometry
def calculate_seed_size(L, W, hole_radius, expected_number_of_elements):
    total_area = L * W
    hole_area = π * (hole_radius ** 2)
    effective_area = total_area - hole_area
    target_element_area = effective_area / expected_number_of_elements
    seed_size = sqrt(target_element_area)
    return seed_size

p.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
```

**Mesh refinement:** Automatic around hole due to geometric constraints

---

## Field Output Requests

### Requested Variables (at nodes)
```python
variables = ('PEEQ', 'LE', 'SDV')
numIntervals = 1    # Output at end of step only
position = NODES    # Nodal averaging
```

**Extracted fields:**
- `LE11`: Logarithmic strain (warp direction)
- `LE22`: Logarithmic strain (fill direction)
- `LE12`: Logarithmic shear strain
- `S11`: Stress component (warp direction)
- `S22`: Stress component (fill direction)
- `S12`: Shear stress
- `PEEQ`: Equivalent plastic strain
- `SDV`: State-dependent variables (damage indicators)

---

## Job Configuration

```python
mdb.Job(
    name=jobname,
    model='Model-1',
    type=ANALYSIS,
    memory=90,                      # 90% of available memory
    memoryUnits=PERCENTAGE,
    explicitPrecision=SINGLE,       # Single precision for speed
    nodalOutputPrecision=SINGLE,
    numCpus=1,                      # Serial execution per job
    numDomains=1,
    multiprocessingMode=DEFAULT
)
```

**Parallel execution:** Handled by Python wrapper (`fclga_run_simulations.py`) using multiprocessing across samples

---

## Example Input File

A representative `.inp` file structure:

```inp
*HEADING
Composite plate with hole - Nonlinear analysis

*PART, NAME=PLATE
*NODE
...
*ELEMENT, TYPE=S4R
...
*END PART

*ASSEMBLY, NAME=ASSEMBLY
*INSTANCE, NAME=Plate-1, PART=PLATE
*END INSTANCE
*END ASSEMBLY

*MATERIAL, NAME=ABQ_PLY_FABRIC
*DENSITY
1.594e-09,
*DEPVAR
16,
*USER MATERIAL, CONSTANTS=40
27500.0, 27500.0, 0.11, 2900.0, 27500.0, 27500.0, 0.11, 0.0
604.0, 291.0, 604.0, 291.0, 75.0, 0.0, 0.0, 0.0
20.438, 4.6257, 20.438, 4.6257, 0.3221, 1.0, 0.0, 0.0
75.0, 0.0008, 0.552, 0.0, 0.0, 0.0, 0.0, 0.0
0.0, 1.0, 1e9, 1e9, -1e9, 0.0, 0.0, 0.0

*SHELL SECTION, ELSET=ALL, COMPOSITE, ORIENTATION=ORI-1
0.25, 4, ABQ_PLY_FABRIC, 45.0
0.25, 4, ABQ_PLY_FABRIC, -45.0
0.25, 4, ABQ_PLY_FABRIC, -45.0
0.25, 4, ABQ_PLY_FABRIC, 45.0

*STEP, NAME=Step-1
*DYNAMIC, EXPLICIT
, 0.01
*BOUNDARY
Encastre-Set, ENCASTRE
*BOUNDARY, AMPLITUDE=Ramp
Disp-Set, 1, 1, <displacement_value>
*OUTPUT, FIELD, NUMBER INTERVAL=1
*NODE OUTPUT
LE, S, PEEQ, SDV
*END STEP
```

**Note:** Actual `.inp` files generated in `src/preprocessing/INPs/` directory

---

## Post-Processing Workflow

### 1. ODB Extraction
```bash
python legacy/extractodb.py
# Copies .odb files for processing
```

### 2. Result Extraction (Abaqus CAE)
```bash
abaqus cae nogui=src/preprocessing/fclga_extract_results.py
# Extracts nodal E11, E22, E12, S11, S22, S12 to .txt files
```

### 3. Graph Construction
See [`fclga_extract_features.py`](../src/preprocessing/fclga_extract_features.py) for mesh-to-graph conversion

---

## Reproducibility Notes

### Critical Parameters for Exact Reproduction

1. **Random seed:** Fixed in geometry generation script
2. **Mesh seed:** Adaptive but deterministic based on geometry
3. **Time step:** Auto-computed by Abaqus explicit solver
4. **Material constants:** Fixed (no randomization)
5. **Displacement range:** Defined per sample (see `plate_geometry_data.pt`)

### System Requirements

- **Abaqus version:** 2021 or later (tested on 2021)
- **UMAT subroutine:** Custom progressive damage model (contact authors)
- **Computational time:** ~15-30 min/sample on single CPU
- **Total dataset generation:** ~125-250 CPU-hours for 500 samples

### Known Limitations

- UMAT subroutine not included in repository (proprietary)
- Results may vary slightly due to Abaqus solver version differences
- Explicit solver uses automatic time stepping (small numerical differences possible)

---

## Contact for UMAT

The User Material subroutine is proprietary. For access or questions:

**Luca Patrignani**  
Email: l.patrignani@imperial.ac.uk  
Institution: Imperial College London

---

## References

1. Abaqus User Manual, Version 2021
2. Pinho, S. T., et al. "Physically-based failure models for composite laminates" (related material model)
3. This repository: [`fclga_generate_geometry.py`](../src/preprocessing/fclga_generate_geometry.py)

---

## Quick Verification

To verify your Abaqus setup matches the paper:

```bash
# 1. Check generated .inp file
head -n 50 src/preprocessing/INPs/Plate_nonlinear_0.inp

# 2. Look for material definition
grep -A 5 "USER MATERIAL" src/preprocessing/INPs/Plate_nonlinear_0.inp

# 3. Verify step settings
grep "DYNAMIC, EXPLICIT" src/preprocessing/INPs/Plate_nonlinear_0.inp
```

Expected output should match the constants shown in this document.
