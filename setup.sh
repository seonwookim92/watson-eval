#!/usr/bin/env bash
# =============================================================================
# setup.sh — Create virtual environments and install dependencies for all models
#
# Run once from the repo root:
#   bash setup.sh
#
# Options:
#   bash setup.sh watson            # only set up our model
#   bash setup.sh ctinexus ttpdrill # set up specific baselines
#   bash setup.sh all               # all models (default)
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"   # override with: PYTHON=python3.11 bash setup.sh

GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; }
sep()   { echo -e "\n${YELLOW}── $* ──${NC}"; }

# Check Python version (>=3.10 required)
check_python() {
    local py="${1:-$PYTHON}"
    local ver
    ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || {
        error "Python not found: $py"
        return 1
    }
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [[ $major -lt 3 || ($major -eq 3 && $minor -lt 10) ]]; then
        error "Python $ver found but >=3.10 required"
        return 1
    fi
    info "Python $ver ($py)"
}

# ─── watson ───────────────────────────────────────────────────────────────────
setup_watson() {
    sep "watson (our model)"
    local DIR="$ROOT/watson"
    local VENV="$DIR/.venv"

    check_python "$PYTHON"

    if [[ -d "$VENV" ]]; then
        warn "venv already exists at $VENV — skipping creation (delete to recreate)"
    else
        "$PYTHON" -m venv "$VENV"
        info "Created venv: $VENV"
    fi

    local PIP="$VENV/bin/pip"
    "$PIP" install --upgrade pip -q
    "$PIP" install -r "$DIR/requirements.txt"
    info "Installed watson dependencies"

    info "watson ready  →  source watson/.venv/bin/activate"
}

# ─── ctinexus baseline ────────────────────────────────────────────────────────
setup_ctinexus() {
    sep "ctinexus baseline"
    local DIR="$ROOT/baselines/ctinexus"
    local VENV="$DIR/.venv"

    check_python "$PYTHON"

    if [[ -d "$VENV" ]]; then
        warn "venv already exists at $VENV — skipping creation"
    else
        "$PYTHON" -m venv "$VENV"
        info "Created venv: $VENV"
    fi

    local PIP="$VENV/bin/pip"
    "$PIP" install --upgrade pip -q
    # Install the ctinexus package itself (pyproject.toml, editable)
    "$PIP" install -e "$DIR"
    info "Installed ctinexus and its dependencies"

    info "ctinexus ready  →  source baselines/ctinexus/.venv/bin/activate"
}

# ─── ttpdrill baseline ────────────────────────────────────────────────────────
setup_ttpdrill() {
    sep "ttpdrill baseline"
    local DIR="$ROOT/baselines/ttpdrill"
    local VENV="$DIR/.venv_ttpdrill"

    check_python "$PYTHON"

    if [[ -d "$VENV" ]]; then
        warn "venv already exists at $VENV — skipping creation"
    else
        "$PYTHON" -m venv "$VENV"
        info "Created venv: $VENV"
    fi

    local PIP="$VENV/bin/pip"
    local PY="$VENV/bin/python"
    "$PIP" install --upgrade pip -q
    "$PIP" install spacy rank-bm25 tqdm pandas python-dotenv

    # Download spacy English model (needed for dependency parsing)
    if "$PY" -m spacy validate 2>/dev/null | grep -q "en_core_web_sm"; then
        info "spacy en_core_web_sm already installed"
    else
        "$PY" -m spacy download en_core_web_sm
        info "Downloaded spacy model: en_core_web_sm"
    fi

    info "ttpdrill ready  →  source baselines/ttpdrill/.venv_ttpdrill/bin/activate"
}

# ─── gtikg baseline ───────────────────────────────────────────────────────────
setup_gtikg() {
    sep "gtikg baseline"
    local DIR="$ROOT/baselines/gtikg"
    local VENV="$DIR/.venv_gtikg"

    check_python "$PYTHON"

    if [[ -d "$VENV" ]]; then
        warn "venv already exists at $VENV — skipping creation"
    else
        "$PYTHON" -m venv "$VENV"
        info "Created venv: $VENV"
    fi

    local PIP="$VENV/bin/pip"
    "$PIP" install --upgrade pip -q
    "$PIP" install openai tqdm python-dotenv litellm

    info "gtikg ready  →  source baselines/gtikg/.venv_gtikg/bin/activate"
}

# ─── watson-new (OntologyExtractor backend) ───────────────────────────────────
setup_watson_new() {
    sep "watson-new (OntologyExtractor backend)"
    local DIR="$ROOT/watson-new"
    local VENV="$DIR/.venv"

    check_python "$PYTHON"

    if [[ -d "$VENV" ]]; then
        warn "venv already exists at $VENV — skipping creation (delete to recreate)"
    else
        "$PYTHON" -m venv "$VENV"
        info "Created venv: $VENV"
    fi

    local PIP="$VENV/bin/pip"
    "$PIP" install --upgrade pip -q
    "$PIP" install -r "$DIR/requirements.txt"
    info "Installed watson-new dependencies"

    info "watson-new ready  →  source watson-new/.venv/bin/activate"
}

# ─── ladder_ner baseline ──────────────────────────────────────────────────────
setup_ladder_ner() {
    sep "ladder NER baseline"
    local DIR="$ROOT/baselines/ladder/ner"
    local VENV="$DIR/.venv_ladder_ner"

    check_python "$PYTHON"

    if [[ -d "$VENV" ]]; then
        warn "venv already exists at $VENV — skipping creation"
    else
        "$PYTHON" -m venv "$VENV"
        info "Created venv: $VENV"
    fi

    local PIP="$VENV/bin/pip"
    "$PIP" install --upgrade pip -q
    "$PIP" install -r "$DIR/requirements.txt"
    "$PIP" install git+https://github.com/aiforsec/CyNER.git
    info "Installed ladder NER dependencies (including CyNER)"

    info "ladder_ner ready  →  source baselines/ladder/ner/.venv_ladder_ner/bin/activate"
}

# ─── ladder_re baseline ───────────────────────────────────────────────────────
setup_ladder_re() {
    sep "ladder Relation Extraction baseline"
    local DIR="$ROOT/baselines/ladder/relation_extraction"
    local VENV="$DIR/.venv_ladder_re"

    check_python "$PYTHON"

    if [[ -d "$VENV" ]]; then
        warn "venv already exists at $VENV — skipping creation"
    else
        "$PYTHON" -m venv "$VENV"
        info "Created venv: $VENV"
    fi

    local PIP="$VENV/bin/pip"
    "$PIP" install --upgrade pip -q
    "$PIP" install -r "$DIR/requirements.txt"
    # Force GPU-enabled PyTorch (CUDA 12.1) — adjust index-url for your CUDA version
    "$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    info "Installed ladder RE dependencies (PyTorch GPU/CUDA 12.1)"

    info "ladder_re ready  →  source baselines/ladder/relation_extraction/.venv_ladder_re/bin/activate"
}

# ─── evaluate (shared, in watson venv) ────────────────────────────────────────
setup_evaluate() {
    sep "evaluate.py dependencies"
    local VENV="$ROOT/watson/.venv"
    if [[ ! -d "$VENV" ]]; then
        warn "watson venv not found — run setup_watson first"
        return 0
    fi
    local PIP="$VENV/bin/pip"
    # Optional: sentence-transformers for embedding match mode
    "$PIP" install sentence-transformers -q && info "sentence-transformers installed (embedding match mode)"
    # Support Gemini for LLM evaluation
    "$PIP" install langchain-google-genai -q && info "langchain-google-genai installed (Gemini evaluation mode)"
    # Support Claude for LLM evaluation
    "$PIP" install langchain-anthropic -q && info "langchain-anthropic installed (Claude evaluation mode)"
    info "evaluate.py ready — uses watson/.venv"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
TARGETS=("${@:-all}")

# Expand "all"
if [[ " ${TARGETS[*]} " == *" all "* ]]; then
    TARGETS=(watson watson-new ctinexus ttpdrill gtikg ladder_ner ladder_re)
    # Automatically add 'evaluate' if watson is in targets
    TARGETS+=(evaluate)
fi

echo ""
echo "============================================================"
echo "  CTI Ontology Evaluation — Environment Setup"
echo "  Root: $ROOT"
echo "  Python: $($PYTHON --version 2>&1)"
echo "  Targets: ${TARGETS[*]}"
echo "============================================================="

for target in "${TARGETS[@]}"; do
    case "$target" in
        watson)      setup_watson      ;;
        watson-new)  setup_watson_new  ;;
        ctinexus)    setup_ctinexus    ;;
        ttpdrill)    setup_ttpdrill    ;;
        gtikg)       setup_gtikg       ;;
        ladder_ner)  setup_ladder_ner  ;;
        ladder_re)   setup_ladder_re   ;;
        evaluate)    setup_evaluate    ;;
        *)           error "Unknown target: $target (choose: watson watson-new ctinexus ttpdrill gtikg ladder_ner ladder_re all)" ;;
    esac
done

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env in the root directory with your API keys"
echo "  2. Configure watson-new/OntologyExtractor/config.json (LLM/Neo4j endpoints)"
echo "  3. Run a smoke test:"
echo "     python run.py --model all --schema uco --limit 3 --dry-run"
echo "     python run.py --model watson --schema uco --limit 3"
echo "     python run.py --model watson-new --schema uco --limit 3"
echo "     python run.py --model ladder_ner ladder_re --dry-run"
echo "============================================================"
