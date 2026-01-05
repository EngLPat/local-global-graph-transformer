# FCLGA GraphTransformer - Quick Start Commands
# Run these commands one by one to create a 10-sample test dataset

# Navigate to base directory
Set-Location "c:\Users\lpatrign\Desktop\python_paper"

Write-Host @"

========================================
FCLGA GraphTransformer - Quick Test
========================================

Modified scripts to generate 10 samples:
- N = 2 (hole positions)
- M = 5 (displacement values)
- Total: 10 samples

Estimated total time: 30-60 minutes

========================================

STEP 1: Generate Geometry
------------------------------------------
"@

Write-Host "Run this command:"
Write-Host "  abaqus cae nogui=FCLGA_GraphTransformer\src\preprocessing\fclga_generate_geometry.py" -ForegroundColor Cyan

Write-Host "`nExpected output:"
Write-Host "  - INPs\ directory with 10 .inp files"
Write-Host "  - plate_geometry_data.pt"
Write-Host "`nExpected time: 2-5 minutes`n"
