"""
FCLGA GraphTransformer - Preprocessing Test Script

Quick script to test if preprocessing scripts work correctly.
Run this to verify each step before integration.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} NOT FOUND")
        return False


def check_directory(dirpath, pattern, description):
    """Check if directory exists and count files matching pattern"""
    if os.path.exists(dirpath):
        files = list(Path(dirpath).rglob(pattern))
        print(f"  ✓ {description}: {dirpath} ({len(files)} files)")
        return len(files) > 0
    else:
        print(f"  ✗ {description}: {dirpath} NOT FOUND")
        return False


def test_preprocessing_pipeline():
    """Test all preprocessing steps"""
    
    print("\n" + "="*60)
    print("FCLGA GraphTransformer - Preprocessing Test")
    print("="*60 + "\n")
    
    base_dir = "c:/Users/lpatrign/Desktop/python_paper"
    results = {}
    
    # Test 1: Geometry Generation
    print("Test 1: Geometry Generation (fclga_generate_geometry.py)")
    print("-" * 60)
    inp_dir = os.path.join(base_dir, "INPs")
    geom_tensor = os.path.join(base_dir, "plate_geometry_data.pt")
    
    test1a = check_directory(inp_dir, "*.inp", "INP files")
    test1b = check_file_exists(geom_tensor, "Geometry tensor")
    results['geometry_generation'] = test1a and test1b
    print()
    
    # Test 2: FEA Simulation
    print("Test 2: FEA Simulation (fclga_run_simulations.py)")
    print("-" * 60)
    odb_dir = os.path.join(base_dir, "ODBs")
    
    test2 = check_directory(odb_dir, "**/*.odb", "ODB files")
    results['fea_simulation'] = test2
    print()
    
    # Test 3: Feature Extraction
    print("Test 3: Feature Extraction (fclga_extract_features.py)")
    print("-" * 60)
    node_data = os.path.join(base_dir, "node_gnn_data.pt")
    triangulation = os.path.join(base_dir, "triangulation_data.pkl")
    
    test3a = check_file_exists(node_data, "Node GNN data")
    test3b = check_file_exists(triangulation, "Triangulation data")
    results['feature_extraction'] = test3a and test3b
    print()
    
    # Test 4: Result Extraction
    print("Test 4: Result Extraction (fclga_extract_results.py)")
    print("-" * 60)
    strains_dir = os.path.join(base_dir, "strains")
    
    test4 = check_directory(strains_dir, "E11_*.txt", "Strain files")
    results['result_extraction'] = test4
    print()
    
    # Test 5: Dataset Building
    print("Test 5: Dataset Building (fclga_build_dataset.py)")
    print("-" * 60)
    strains_tensor = os.path.join(base_dir, "strains.pt")
    
    test5 = check_file_exists(strains_tensor, "Strains tensor")
    results['dataset_building'] = test5
    print()
    
    # Final check
    print("="*60)
    print("Summary:")
    print("="*60)
    for step, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {step}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Preprocessing pipeline is complete!")
    else:
        print("✗ SOME TESTS FAILED - Check missing files above")
        print("\nNext steps:")
        if not results['geometry_generation']:
            print("  1. Run: abaqus cae nogui=src/preprocessing/fclga_generate_geometry.py")
        if not results['fea_simulation']:
            print("  2. Run: python src/preprocessing/fclga_run_simulations.py")
        if not results['feature_extraction']:
            print("  3. Run: python src/preprocessing/fclga_extract_features.py")
        if not results['result_extraction']:
            print("  4. Run: abaqus cae nogui=src/preprocessing/fclga_extract_results.py")
        if not results['dataset_building']:
            print("  5. Run: python src/preprocessing/fclga_build_dataset.py")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = test_preprocessing_pipeline()
    sys.exit(0 if success else 1)
