# Run script for Relation Extraction
Write-Host "Running Relation Extraction test (train_supervised.py)..."

if (Test-Path .\venv_re\Scripts\Activate.ps1) {
    .\venv_re\Scripts\Activate.ps1
    python train_supervised.py
    Write-Host "Done."
} else {
    Write-Host "Virtual environment not found. Please run setup_re.ps1 first."
}
