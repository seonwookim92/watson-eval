#!/usr/bin/env bash

# Exit on error
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"

echo "[*] Setting up Cyber Data Provider environments..."
echo "[*] Root: $ROOT"

# 1. Create directory structure if not exists
mkdir -p "$ROOT/input" "$ROOT/ontology" "$ROOT/mcp"

# 2. Setup UCO Ontology
echo "[*] Downloading UCO Ontology..."
if [ -d "$ROOT/ontology/uco" ]; then
    echo "[!] ontology/uco already exists. Skipping download."
else
    TEMP_UCO_DIR="$(mktemp -d)"
    git clone https://github.com/ucoProject/UCO.git "$TEMP_UCO_DIR"
    mv "$TEMP_UCO_DIR/ontology/uco" "$ROOT/ontology/"
    rm -rf "$TEMP_UCO_DIR"
    echo "[+] UCO Ontology setup complete."
fi

# 3. Setup Universal Ontology MCP
echo "[*] Downloading Universal Ontology MCP..."
if [ -d "$ROOT/mcp/universal-ontology-mcp" ]; then
    echo "[!] mcp/universal-ontology-mcp already exists. Skipping download."
else
    git clone https://github.com/seonwookim92/universal-ontology-mcp "$ROOT/mcp/universal-ontology-mcp"
    echo "[+] Universal Ontology MCP setup complete."
fi

# 4. Environment File
if [ ! -f "$ROOT/.env" ]; then
    if [ -f "$ROOT/.env.sample" ]; then
        echo "[*] Creating .env from .env.sample..."
        cp "$ROOT/.env.sample" "$ROOT/.env"
        echo "[!] Please edit .env and add your API keys."
    else
        echo "[!] .env.sample not found — skipping .env creation. Create .env manually."
    fi
fi

# 5. Virtual environment
if [ -d "$VENV" ]; then
    echo "[!] venv already exists at $VENV — skipping creation (delete to recreate)"
else
    echo "[*] Creating virtual environment at $VENV..."
    "$PYTHON" -m venv "$VENV"
    echo "[+] Created venv: $VENV"
fi

# 6. Dependencies
echo "[*] Installing dependencies..."
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"

echo "[+] Setup complete! You are ready to run main.py."
echo "[*] Activate with: source $VENV/bin/activate"
