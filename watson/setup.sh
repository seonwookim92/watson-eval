#!/bin/bash

# Exit on error
set -e

echo "[*] Setting up Cyber Data Provider environments..."

# 1. Create directory structure if not exists
mkdir -p input ontology mcp

# 2. Setup UCO Ontology
echo "[*] Downloading UCO Ontology..."
if [ -d "ontology/uco" ]; then
    echo "[!] ontology/uco already exists. Skipping download."
else
    TEMP_UCO_DIR="temp_uco_repo"
    git clone https://github.com/ucoProject/UCO.git $TEMP_UCO_DIR
    mkdir -p ontology
    mv $TEMP_UCO_DIR/ontology/uco ontology/
    rm -rf $TEMP_UCO_DIR
    echo "[+] UCO Ontology setup complete."
fi

# 3. Setup Universal Ontology MCP
echo "[*] Downloading Universal Ontology MCP..."
if [ -d "mcp/universal-ontology-mcp" ]; then
    echo "[!] mcp/universal-ontology-mcp already exists. Skipping download."
else
    git clone https://github.com/seonwookim92/universal-ontology-mcp mcp/universal-ontology-mcp
    echo "[+] Universal Ontology MCP setup complete."
fi

# 4. Environment File
if [ ! -f ".env" ]; then
    echo "[*] Creating .env from .env.sample..."
    cp .env.sample .env
    echo "[!] Please edit .env and add your API keys."
fi

# 5. Dependencies
echo "[*] Installing dependencies..."
pip install -r requirements.txt

echo "[+] Setup complete! You are ready to run main.py."
