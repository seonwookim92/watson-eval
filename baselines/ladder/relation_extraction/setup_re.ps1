Write-Host "Setting up Python virtual environment for Relation Extraction..."

# Attempt to remove existing venv_re if it exists
if (Test-Path ".\venv_re") {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Path ".\venv_re" -Recurse -Force -ErrorAction SilentlyContinue
}

python -m venv venv_re
if ($?) {
    Write-Host "Activating virtual environment..."
    .\venv_re\Scripts\Activate.ps1
    
    Write-Host "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    Write-Host ""
    Write-Host "=================================================================="
    Write-Host "Relation Extraction environment setup attempt finished."
    Write-Host "WARNING: The requirements (like torch==1.4.0) are very old."
    Write-Host "If installation failed, please use Python 3.7 or 3.8 to run this."
    Write-Host "=================================================================="
} else {
    Write-Host "Failed to create virtual environment."
}
