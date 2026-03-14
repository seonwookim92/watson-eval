# Run script for NER inference
Write-Host "Running NER inference test (infer.py)..."

if (Test-Path .\venv_ner\Scripts\Activate.ps1) {
    .\venv_ner\Scripts\Activate.ps1
    python infer.py
    Write-Host "Done. Check sample_output.txt for the results."
} else {
    Write-Host "Virtual environment not found. Please run setup_ner.ps1 first."
}
