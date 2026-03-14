#!/usr/bin/env bash
# Ember 3.0 — one-command setup for any platform
# Usage: ./setup.sh [--with-data]
#   --with-data: also install the pre-migrated database (if ember3.db is present)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Find or install uv ───────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    UV=uv
elif [ -x /usr/local/bin/uv ]; then
    UV=/usr/local/bin/uv
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi

# ── Create venv ──────────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    "$UV" venv .venv --python 3.12 || "$UV" venv .venv
fi

# Determine python path (cross-platform)
if [ -f .venv/bin/python3 ]; then
    PYTHON=.venv/bin/python3
elif [ -f .venv/Scripts/python.exe ]; then
    PYTHON=.venv/Scripts/python.exe
else
    echo "Error: could not find python in .venv" >&2
    exit 1
fi

# ── Install package ──────────────────────────────────────────────────────────
echo "Installing ember3..."
"$UV" pip install --python "$PYTHON" -e ".[fuzzy]"

# ── Determine data directory ─────────────────────────────────────────────────
if [ -n "${EMBER_DATA_DIR:-}" ]; then
    DATA_DIR="$EMBER_DATA_DIR"
elif [ "$(uname)" = "Linux" ] || [ "$(uname)" = "Darwin" ]; then
    DATA_DIR="$HOME/.ember"
else
    # Windows (Git Bash / MSYS2)
    DATA_DIR="${APPDATA:-$HOME}/ember"
fi

mkdir -p "$DATA_DIR/models/all-MiniLM-L6-v2"

# ── Install model files if bundled ───────────────────────────────────────────
if [ -f models/model.onnx ] && [ ! -f "$DATA_DIR/models/all-MiniLM-L6-v2/model.onnx" ]; then
    echo "Installing ONNX model..."
    cp models/model.onnx "$DATA_DIR/models/all-MiniLM-L6-v2/"
    cp models/tokenizer.json "$DATA_DIR/models/all-MiniLM-L6-v2/"
fi

# ── Install database if --with-data ──────────────────────────────────────────
if [ "${1:-}" = "--with-data" ] && [ -f ember3.db ]; then
    if [ -f "$DATA_DIR/ember3.db" ]; then
        echo "Database already exists at $DATA_DIR/ember3.db — skipping (back up manually if you want to replace)"
    else
        echo "Installing pre-migrated database..."
        cp ember3.db "$DATA_DIR/ember3.db"
    fi
fi

# ── Print MCP config ────────────────────────────────────────────────────────
FULL_PYTHON="$(cd "$SCRIPT_DIR" && pwd)/$PYTHON"
echo ""
echo "============================================================"
echo "  Ember 3.0 installed successfully!"
echo "============================================================"
echo ""
echo "Data directory: $DATA_DIR"
echo "Database: $DATA_DIR/ember3.db (auto-created on first run if absent)"
echo ""
echo "Add to your MCP config (~/.claude.json or Claude Desktop):"
echo ""
echo '  "ember": {'
echo '    "type": "stdio",'
echo "    \"command\": \"$FULL_PYTHON\","
echo '    "args": ["-m", "ember"],'
echo '    "env": { "EMBER_AGENT": "claude" }'
echo '  }'
echo ""
echo "To migrate existing ember-full data:"
echo "  $FULL_PYTHON migrate.py --source ~/.ember --execute"
echo ""
