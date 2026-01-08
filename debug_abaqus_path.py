"""Debug script to check Abaqus working directory."""
from pathlib import Path

PROJECT_ROOT = Path.cwd()
SIMULATIONS_DIR = PROJECT_ROOT / "data" / "raw" / "linear" / "simulations"

print("=" * 80)
print("DEBUG: Abaqus Working Directory Check")
print("=" * 80)
print(f"Current working directory: {PROJECT_ROOT}")
print(f"SIMULATIONS_DIR: {SIMULATIONS_DIR}")
print(f"SIMULATIONS_DIR exists: {SIMULATIONS_DIR.exists()}")

if SIMULATIONS_DIR.exists():
    odb_files = sorted(SIMULATIONS_DIR.rglob("*.odb"))
    print(f"Number of ODB files found: {len(odb_files)}")
    if odb_files:
        print(f"First 5 ODB files:")
        for f in odb_files[:5]:
            print(f"  - {f}")
else:
    print("ERROR: SIMULATIONS_DIR does not exist!")
print("=" * 80)
