# FCLGA GraphTransformer - Small Test Run (10 Samples)
# Quick test of preprocessing pipeline

Write-Host "`n=========================================="
Write-Host "FCLGA GraphTransformer - Small Test"
Write-Host "Creating 10 sample dataset"
Write-Host "==========================================`n"

$base_dir = "c:\Users\lpatrign\Desktop\python_paper"
cd $base_dir

Write-Host "Step 1: Generate Geometry (10 samples)"
Write-Host "--------------------------------------"
Write-Host "Command: abaqus cae nogui=FCLGA_GraphTransformer\src\preprocessing\fclga_generate_geometry.py"
Write-Host "Expected time: ~2-5 minutes"
Write-Host "Expected output: INPs directory with 10 .inp files"
Write-Host "`nPress Enter when ready to start, or Ctrl+C to cancel..."
Read-Host

abaqus cae nogui=FCLGA_GraphTransformer\src\preprocessing\fclga_generate_geometry.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Step 1 Complete!`n"
    $count = (Get-ChildItem .\INPs -Filter *.inp -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "Created $count .inp files`n"
} else {
    Write-Host "`n✗ Step 1 Failed!`n"
    exit 1
}

Write-Host "Step 2: Run FEA Simulations"
Write-Host "--------------------------------------"
Write-Host "Command: python FCLGA_GraphTransformer\src\preprocessing\fclga_run_simulations.py"
Write-Host "Expected time: ~10-30 minutes (depends on CPU)"
Write-Host "Expected output: ODBs directory with simulation results"
Write-Host "`nPress Enter to continue, or Ctrl+C to stop..."
Read-Host

python FCLGA_GraphTransformer\src\preprocessing\fclga_run_simulations.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Step 2 Complete!`n"
} else {
    Write-Host "`n✗ Step 2 Failed!`n"
    exit 1
}

Write-Host "Step 3: Extract Features from INP files"
Write-Host "--------------------------------------"
Write-Host "Command: python FCLGA_GraphTransformer\src\preprocessing\fclga_extract_features.py"
Write-Host "Expected time: < 1 minute"
Write-Host "`nPress Enter to continue..."
Read-Host

python FCLGA_GraphTransformer\src\preprocessing\fclga_extract_features.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Step 3 Complete!`n"
} else {
    Write-Host "`n✗ Step 3 Failed!`n"
    exit 1
}

Write-Host "Step 4: Copy ODB files"
Write-Host "--------------------------------------"
Write-Host "Command: python FCLGA_GraphTransformer\legacy\extractodb.py"
Write-Host "`nPress Enter to continue..."
Read-Host

python FCLGA_GraphTransformer\legacy\extractodb.py

Write-Host "Step 5: Extract Results from ODB files"
Write-Host "--------------------------------------"
Write-Host "Command: abaqus cae nogui=FCLGA_GraphTransformer\src\preprocessing\fclga_extract_results.py"
Write-Host "Expected time: ~5-10 minutes"
Write-Host "`nPress Enter to continue..."
Read-Host

abaqus cae nogui=FCLGA_GraphTransformer\src\preprocessing\fclga_extract_results.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Step 5 Complete!`n"
} else {
    Write-Host "`n✗ Step 5 Failed!`n"
    exit 1
}

Write-Host "Step 6: Build Final Dataset"
Write-Host "--------------------------------------"
Write-Host "Command: python FCLGA_GraphTransformer\src\preprocessing\fclga_build_dataset.py"
Write-Host "`nPress Enter to continue..."
Read-Host

python FCLGA_GraphTransformer\src\preprocessing\fclga_build_dataset.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Step 6 Complete!`n"
} else {
    Write-Host "`n✗ Step 6 Failed!`n"
    exit 1
}

Write-Host "`n=========================================="
Write-Host "Testing Complete!"
Write-Host "==========================================`n"

Write-Host "Running validation..."
cd FCLGA_GraphTransformer
python test_preprocessing.py

Write-Host "`n=========================================="
Write-Host "All Done! Dataset with 10 samples created."
Write-Host "==========================================`n"
