# Abaqus Material Property Card

## Material Definition: ABQ_PLY_FABRIC

**Material Type:** Carbon Fiber/Epoxy Fabric Composite  
**Model:** User Material (UMAT) with Progressive Damage  
**Reference:** See manuscript Section 3.2 for theoretical background

---

## Input File Format (.inp)

```abaqus
*Material, name=ABQ_PLY_FABRIC
*Density
 1.594e-09,
*Depvar
    16,
*User Material, constants=40
** Elastic constants (8)
  27500.0,  27500.0,     0.11,   2900.0,  27500.0,  27500.0,
      0.11,      0.0,
** Strength constants (8)
    604.0,    291.0,    604.0,    291.0,     75.0,      0.0,
      0.0,      0.0,
** Damage evolution constants (12)
  20.438,   4.6257,  20.438,   4.6257,   0.3221,      1.0,
      0.0,      0.0,     75.0,   0.0008,    0.552,      0.0,
** Numerical parameters (12)
      0.0,      0.0,      0.0,      0.0,      0.0,      1.0,
 1.0e+09,  1.0e+09,  -1.0e+09,     0.0,      0.0,      0.0
```

---

## Python API Format (Abaqus/CAE)

```python
# Material creation
mymodel.Material(name='ABQ_PLY_FABRIC')

# Density
mymodel.materials['ABQ_PLY_FABRIC'].Density(table=((1.594e-09, ), ))

# State variables for damage tracking
mymodel.materials['ABQ_PLY_FABRIC'].Depvar(n=16)

# User material constants (40 parameters)
mymodel.materials['ABQ_PLY_FABRIC'].UserMaterial(
    mechanicalConstants=(
        # Elastic constants (1-8)
        27500.0,    # E11 (MPa) - In-plane modulus, warp direction
        27500.0,    # E22 (MPa) - In-plane modulus, fill direction
        0.11,       # nu12 (-) - Major Poisson's ratio
        2900.0,     # G12 (MPa) - In-plane shear modulus
        27500.0,    # E11 (MPa) - Repeated for damage model
        27500.0,    # E22 (MPa) - Repeated for damage model
        0.11,       # nu12 (-) - Repeated for damage model
        0.0,        # (unused)
        
        # Strength constants (9-16)
        604.0,      # X_T (MPa) - Longitudinal tensile strength
        291.0,      # X_C (MPa) - Longitudinal compressive strength
        604.0,      # Y_T (MPa) - Transverse tensile strength
        291.0,      # Y_C (MPa) - Transverse compressive strength
        75.0,       # S_12 (MPa) - In-plane shear strength
        0.0,        # (unused)
        0.0,        # (unused)
        0.0,        # (unused)
        
        # Damage evolution (17-28)
        20.438,     # G_Ift (N/mm) - Mode I fracture toughness (tension)
        4.6257,     # G_Ifc (N/mm) - Mode I fracture toughness (compression)
        20.438,     # G_IIft (N/mm) - Mode II fracture toughness (tension)
        4.6257,     # G_IIfc (N/mm) - Mode II fracture toughness (compression)
        0.3221,     # eta_BK (-) - Benzeggagh-Kenane coefficient
        1.0,        # n_BK (-) - Benzeggagh-Kenane exponent
        0.0,        # (unused)
        0.0,        # (unused)
        75.0,       # S_12 (MPa) - Shear strength (repeated)
        0.0008,     # epsilon_f (-) - Failure strain
        0.552,      # alpha_d (-) - Damage coefficient
        0.0,        # (unused)
        
        # Numerical parameters (29-40)
        0.0,        # (unused)
        0.0,        # (unused)
        0.0,        # (unused)
        0.0,        # (unused)
        0.0,        # (unused)
        1.0,        # (unused)
        1.0e9,      # Viscosity penalty for numerical stability
        1.0e9,      # Viscosity penalty (duplicate)
        -1.0e9,     # Numerical penalty
        0.0,        # (unused)
        0.0,        # (unused)
        0.0         # (unused)
    )
)
```

---

## Composite Layup Configuration

**Ply Stack:** [+45°/-45°/-45°/+45°]

```python
# Create composite layup
compositeLayup = mymodel.parts['PLATE-1'].CompositeLayup(
    name='CompositeLayup-1',
    offsetType=MIDDLE_SURFACE,
    symmetric=False,
    thicknessAssignment=FROM_SECTION
)

# Ply 1: +45°
compositeLayup.CompositePly(
    plyName='Ply-1',
    region=region1,
    material='ABQ_PLY_FABRIC',
    thicknessType=SPECIFY_THICKNESS,
    thickness=0.25,
    orientationType=SPECIFY_ORIENT,
    orientationValue=45.0,
    numIntPoints=1
)

# Ply 2: -45°
compositeLayup.CompositePly(
    plyName='Ply-2',
    region=region1,
    material='ABQ_PLY_FABRIC',
    thicknessType=SPECIFY_THICKNESS,
    thickness=0.25,
    orientationType=SPECIFY_ORIENT,
    orientationValue=-45.0,
    numIntPoints=1
)

# Ply 3: -45°
compositeLayup.CompositePly(
    plyName='Ply-3',
    region=region1,
    material='ABQ_PLY_FABRIC',
    thicknessType=SPECIFY_THICKNESS,
    thickness=0.25,
    orientationType=SPECIFY_ORIENT,
    orientationValue=-45.0,
    numIntPoints=1
)

# Ply 4: +45°
compositeLayup.CompositePly(
    plyName='Ply-4',
    region=region1,
    material='ABQ_PLY_FABRIC',
    thicknessType=SPECIFY_THICKNESS,
    thickness=0.25,
    orientationType=SPECIFY_ORIENT,
    orientationValue=45.0,
    numIntPoints=1
)
```

**Total Laminate Thickness:** 1.0 mm (4 plies × 0.25 mm)

---

## Mesh Configuration

```python
# Element type: Shell with composite layup
elemType1 = mesh.ElemType(
    elemCode=S4R,              # 4-node shell with reduced integration
    elemLibrary=EXPLICIT,      # For explicit dynamics
    hourglassControl=DEFAULT
)

# Approximate global element size
approximate_size = 2.0  # mm

# Mesh generation
p.seedPart(size=approximate_size, deviationFactor=0.1, minSizeFactor=0.1)
p.generateMesh()
```

**Expected Mesh:**
- Element type: S4R (4-node shell, reduced integration)
- Approximate element count: 1000-2500 elements

---

## Analysis Configuration

**Step:** Explicit Dynamics
```python
mymodel.ExplicitDynamicsStep(
    name='Step-1',
    previous='Initial',
    timePeriod=0.01,        # 10 ms total time
    nlgeom=OFF,             # Geometric nonlinearity off
    improvedDtMethod=ON     # Automatic stable time increment
)
```

**Field Outputs:**
```python
mymodel.fieldOutputRequests['F-Output-1'].setValues(
    variables=('S', 'LE', 'U', 'RF', 'CSTRESS', 'CSTRAIN')
)
```
- `S` - Stress components
- `LE` - Logarithmic strain
- `U` - Displacements
- `RF` - Reaction forces
- `CSTRESS` - Stress in composite plies
- `CSTRAIN` - Strain in composite plies

---

## Boundary Conditions

**Fixed Support (BC-1):**
```python
# Location: Left and right edges
mymodel.EncastreBC(
    name='BC-1',
    createStepName='Step-1',
    region=region2
)
```
All DOFs constrained: u1=u2=u3=ur1=ur2=ur3=0

**Displacement Loading (BC-2):**
```python
# Location: Top and bottom edges
# Amplitude: Linear ramp (0 to 1 over 0.01s)
mymodel.TabularAmplitude(
    name='Ramp',
    data=((0.0, 0.0), (0.01, 1.0))
)

mymodel.DisplacementBC(
    name='BC-2',
    createStepName='Step-1',
    region=region3,
    u1=displacement,  # Variable: 1-3 mm for nonlinear, 1-2 mm for linear
    amplitude='Ramp',
    distributionType=UNIFORM
)
```

---

## Units System

| Quantity | Unit | Note |
|----------|------|------|
| Length | mm | Millimeters |
| Force | N | Newtons |
| Mass | tonne | 1 tonne = 1000 kg |
| Time | s | Seconds |
| Stress | MPa | N/mm² |
| Energy | N·mm | Millijoules |
| Density | tonne/mm³ | 1.594e-09 tonne/mm³ = 1594 kg/m³ |

---

## Post-Processing

**Strain Extraction:**
- Field: `LE` (Logarithmic strain)
- Component: `LE11` (In-plane principal strain)
- Location: Element integration points → averaged to nodes
- Coordinate system: Global Cartesian (transformed from local ply coordinates)

**Python extraction script:**
```python
# See src/preprocessing/*/fclga_extract_results.py
# Extracts E11 component at nodal locations
# Transforms from local ply coordinates to global system
```

---

## Example Input Deck

Complete example `.inp` file available in:
- `data/raw/nonlinear/geometry/Plate_0.inp` (after running geometry generation)
- `data/raw/linear/geometry/Plate_0.inp` (elastic version)

**To generate:**
```bash
conda activate fclga
abaqus cae nogui=src/preprocessing/nonlinear/fclga_generate_geometry.py
```

---

## Validation

Material properties validated against:
1. Manufacturer datasheets (T300/5208 carbon/epoxy system)
2. Literature values for fabric composites
3. Progressive damage model verified in previous work

**Reference:** Manuscript Section 3.2 - Material Characterization

---

## Contact

For questions about material properties or UMAT implementation:
- **Luca Patrignani** - l.patrignani@imperial.ac.uk
- **Silvestre T. Pinho** - silvestre.pinho@imperial.ac.uk

Department of Aeronautics, Imperial College London
