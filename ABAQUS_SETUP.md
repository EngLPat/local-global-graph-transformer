# Abaqus FEA Configuration Documentation

This document describes the Abaqus FEA configuration used to generate the training data for this work. These are the settings used in the published results—reproducibility requires using the provided data files or matching these exact parameters.

## Overview

The project simulates composite laminate plates with circular holes under displacement loading. Two material models are supported:

1. **Nonlinear (Progressive Damage)**: Uses Abaqus in-built VUMAT for fabric composites with Dynamic/Explicit solver
2. **Linear (Elastic)**: Uses orthotropic elastic material with Static/Implicit solver

---

## Nonlinear Material Model (Progressive Damage)

### Material Properties

#### Composite Material: `ABQ_PLY_FABRIC`

The material is defined using Abaqus's in-built User Material (VUMAT) subroutine with 40 mechanical constants for progressive damage modeling.

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

**Full UMAT Call:**
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

### Composite Layup (Nonlinear)

**4-ply symmetric layup:**

| Ply | Thickness | Orientation | Material |
|-----|-----------|-------------|----------|
| 1   | 0.25 mm   | +45°        | ABQ_PLY_FABRIC |
| 2   | 0.25 mm   | -45°        | ABQ_PLY_FABRIC |
| 3   | 0.25 mm   | -45°        | ABQ_PLY_FABRIC |
| 4   | 0.25 mm   | +45°        | ABQ_PLY_FABRIC |

**Total thickness:** 1.0 mm  
**Layup:** [+45/-45/-45/+45]

### Solver Settings (Nonlinear)

**Step Type:** Dynamic/Explicit
```python
mymodel.ExplicitDynamicsStep(
    name='Step-1',
    previous='Initial',
    timePeriod=0.01,           # 0.01 seconds (10 ms)
    nlgeom=OFF,                # Geometric nonlinearity disabled
    improvedDtMethod=ON        # Automatic time incrementation
)
```

### Field Output (Nonlinear)

**Requested Variables:**
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

## Linear Elastic Material Model

### Material Properties

#### Composite Material: `CFRP` (Orthotropic Elastic)

**Elastic Properties:**
```python
E1 = 150000.0 MPa      # In-plane modulus (fiber direction)
E2 = 9000.0 MPa        # In-plane modulus (transverse direction)
E3 = 9000.0 MPa        # Through-thickness modulus
nu12 = 0.34            # Poisson's ratio 12
nu13 = 0.34            # Poisson's ratio 13
nu23 = 0.4             # Poisson's ratio 23
G12 = 5000.0 MPa       # In-plane shear modulus
G13 = 5000.0 MPa       # Shear modulus 13
G23 = 5000.0 MPa       # Shear modulus 23
Density = 1.594e-09 tonne/mm³
```

**Abaqus Implementation:**
```python
mymodel.Material(name="CFRP")
mymodel.materials["CFRP"].Elastic(
    type=ENGINEERING_CONSTANTS,
    table=((FABRIC_E1, FABRIC_E2, FABRIC_E3, 
            FABRIC_NU12, FABRIC_NU13, FABRIC_NU23,
            FABRIC_G12, FABRIC_G13, FABRIC_G23),)
)
```

### Composite Layup (Linear)

**4-ply quasi-isotropic layup:**

| Ply | Thickness | Orientation | Material |
|-----|-----------|-------------|----------|
| 1   | 0.25 mm   | 0°          | CFRP |
| 2   | 0.25 mm   | +45°        | CFRP |
| 3   | 0.25 mm   | 90°         | CFRP |
| 4   | 0.25 mm   | -45°        | CFRP |

**Total thickness:** 1.0 mm  
**Layup:** [0/45/90/-45]

### Solver Settings (Linear)

**Step Type:** Static/Implicit
```python
mymodel.StaticStep(
    name='Step-1',
    previous='Initial',
    nlgeom=OFF                # Linear analysis
)
```

### Field Output (Linear)

**Requested Variables:**
```python
variables = ('LE', 'S')        # Logarithmic strain and stress
position = INTEGRATION_POINT   # Extract at integration points, then average to nodes
```

**Extracted fields:**
- `LE11`: Logarithmic strain (fiber direction)
- `LE22`: Logarithmic strain (transverse direction)
- `LE12`: Logarithmic shear strain
- `S11`: Stress component (fiber direction)
- `S22`: Stress component (transverse direction)
- `S12`: Shear stress

---

## Geometry Parameters

### Plate Dimensions

```python
Length (L):           100-200 mm (random, uniform)
Width (W):            100-200 mm (random, uniform)
Thickness:            1.0 mm (4 plies × 0.25 mm)
```

### Hole Parameters
```python
Radius:               10-20 mm (random, uniform)
Position (x):         0.3*L to 0.7*L (linspace sampling)
Position (y):         0.3*W to 0.7*W (linspace sampling)
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

**Displacement ranges:**
- Nonlinear: 1.0-3.0 mm (elastic-plastic regime)
- Linear: 1.0-2.0 mm (elastic regime only)

---

## Boundary Conditions (Both Models)

### Fixed Support (Encastre)
```python
# Applied to: Left and right edges
# Constraints: All DOFs fixed (u1=u2=u3=ur1=ur2=ur3=0)
BC-1: EncastreBC on edges
```

### Displacement Loading
```python
# Applied to: Top and bottom edges
# Loading: u1 = displacement (variable), u2=u3=free
# Amplitude: Ramped from 0 to 1 over step time

mymodel.TabularAmplitude(
    name='Ramp',
    data=((0.0, 0.0), (step_time, 1.0))  # Linear ramp
)

mymodel.DisplacementBC(
    name='BC-2',
    u1=displacement,      # Applied displacement (varies by sample)
    amplitude='Ramp',
    distributionType=UNIFORM
)
```

**Step duration:**
- Nonlinear: 0.01 seconds (explicit dynamics)
- Linear: No time parameter (static analysis)

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

**Backup Element:** S3 (3-node triangular shell)

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

## Job Configuration

```python
mdb.Job(
    name=jobname,
    model='Model-1',
    type=ANALYSIS,
    memory=90,                      # 90% of available memory
    memoryUnits=PERCENTAGE,
    explicitPrecision=SINGLE,       # Single precision (nonlinear)
    nodalOutputPrecision=SINGLE,
    numCpus=1,                      # Serial execution per job
    numDomains=1,
    multiprocessingMode=DEFAULT
)
```

**Parallel execution:** Handled by Python wrapper (`fclga_run_simulations.py`) using multiprocessing across samples

---

## Example Input File (Nonlinear)

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

**Note:** Actual `.inp` files located in `data/raw/{material_type}/geometry/`

---

## Reproducibility Notes

### Critical Parameters for Exact Reproduction

1. **Random seed:** Fixed in geometry generation scripts (GEOMETRY_SEED = 42)
2. **Important:** The files in `data/raw/` are the exact geometries used in the paper. Re-running the generation scripts will produce a consistent but different set of geometries.
3. **Mesh seed:** Adaptive but deterministic based on geometry
4. **Time step:** Auto-computed by Abaqus solver
5. **Material constants:** Fixed (no randomization)
6. **Displacement range:** Defined per sample (stored in geometry data)

### System Requirements

- **Abaqus version:** 2021 or later (tested on 2021)
- **VUMAT subroutine:** Abaqus in-built VUMAT for Fabric Reinforced Composites (nonlinear only)
- **No custom subroutines required** - all material models use Abaqus standard capabilities

---

## References

1. Abaqus User Manual, Version 2021
2. Dassault Systèmes, "VUMAT for Fabric Reinforced Composites," ABAQUS/Explicit User Material Documentation, Technical Report, 2008.
3. This repository: Geometry generation scripts in `src/preprocessing/{material_type}/`

---

## Quick Verification

To verify your Abaqus setup matches the paper:

```bash
# 1. Check nonlinear .inp file
head -n 50 data/raw/nonlinear/geometry/Plate_nonlinear_0.inp

# 2. Verify VUMAT material definition
grep -A 8 "User Material" data/raw/nonlinear/geometry/Plate_nonlinear_0.inp

# 3. Verify nonlinear solver
grep "Dynamic, Explicit" data/raw/nonlinear/geometry/Plate_nonlinear_0.inp

# 4. Check linear .inp file
head -n 50 data/raw/linear/geometry/Plate_linear_0.inp

# 5. Verify elastic material definition
grep -A 3 "Elastic" data/raw/linear/geometry/Plate_linear_0.inp

# 6. Verify linear solver
grep "Static" data/raw/linear/geometry/Plate_linear_0.inp
```
