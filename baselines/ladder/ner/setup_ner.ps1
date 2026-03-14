Write-Host "Setting up Python virtual environment for NER..."

# Attempt to remove existing venv_ner if it exists
if (Test-Path ".\venv_ner") {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Path ".\venv_ner" -Recurse -Force -ErrorAction SilentlyContinue
}

python -m venv venv_ner
if ($?) {
    Write-Host "Activating virtual environment..."
    .\venv_ner\Scripts\Activate.ps1
    
    Write-Host "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    Write-Host "Installing missing cyner module..."
    pip install git+https://github.com/aiforsec/CyNER.git
    
    Write-Host ""
    Write-Host "=================================================================="
    Write-Host "NER environment setup attempt finished."
    Write-Host "WARNING: The requirements (like spacy==2.1.8) are very old."
    Write-Host "If installation failed, please use Python 3.7 or 3.8 to run this."
    Write-Host "=================================================================="
} else {
    Write-Host "Failed to create virtual environment."
}
