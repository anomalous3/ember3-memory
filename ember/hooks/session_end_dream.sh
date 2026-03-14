#!/bin/bash
# Session End Dream Cycle (Claude-as-Dreamer architecture)
#
# Six phases:
#   Phase 0:   Session distillation — extract durable facts from the just-ended session
#   Phase 0.5: Unconscious scan — find topics discussed across sessions but never stored
#   Phase 1:   Mechanical maintenance via maintain.py (Python, no API calls)
#   Phase 2:   Creative dreaming via claude -p --model haiku (Claude IS the dreamer)
#   Phase 3:   Synthesis via claude -p --model sonnet (deeper analysis)
#   Coda:      Archive the dream log as a searchable chunk
#
# Architecture: The hook itself exits immediately after forking the dream cycle
# into a detached background process. This prevents Claude Code from killing the
# dream cycle during its shutdown sequence ("Hook cancelled").
#
# Recursion prevention:
#   - EMBER_DREAM_CYCLE env var prevents re-entry when claude -p exits
#   - Lockfile prevents concurrent dream cycles

set -uo pipefail

# ── Recursion guard ──────────────────────────────────────────────────────────
if [ "${EMBER_DREAM_CYCLE:-}" = "1" ]; then
    exit 0
fi
export EMBER_DREAM_CYCLE=1

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
# Unset Claude Code nesting guard so claude -p can run from the hook
unset CLAUDECODE
unset CLAUDE_CODE_ENTRYPOINT
unset CLAUDE_CODE_SESSION_ACCESS_TOKEN

# Resolve paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMBER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Platform-aware venv python ─────────────────────────────────────────────
if [ -x "$EMBER_ROOT/.venv/bin/python3" ]; then
    EMBER_VENV="$EMBER_ROOT/.venv/bin/python3"
elif [ -x "$EMBER_ROOT/.venv/Scripts/python.exe" ]; then
    EMBER_VENV="$EMBER_ROOT/.venv/Scripts/python.exe"
else
    echo "[dream] ERROR: No venv python found in $EMBER_ROOT/.venv" >&2
    exit 1
fi

MAINTAIN_SCRIPT="$EMBER_ROOT/maintain.py"
HOOKS_DIR="$SCRIPT_DIR"

# ── Platform-aware data directory ──────────────────────────────────────────
# Respects EMBER_DATA_DIR env var; falls back to %APPDATA%/ember on Windows,
# ~/.ember everywhere else.
if [ -n "${EMBER_DATA_DIR:-}" ]; then
    EMBER_DIR="$EMBER_DATA_DIR"
elif [ -n "${APPDATA:-}" ]; then
    EMBER_DIR="$APPDATA/ember"
else
    EMBER_DIR="$HOME/.ember"
fi

DREAM_LOG="$EMBER_DIR/dream-log.md"
LOCKFILE="$EMBER_DIR/dream.lock"

# Ensure data dir exists
mkdir -p "$EMBER_DIR"

# Source ANTHROPIC_API_KEY from shell profile if not already set
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
        if [ -f "$rc" ]; then
            eval "$(grep '^export ANTHROPIC_API_KEY' "$rc" 2>/dev/null)" || true
            [ -n "${ANTHROPIC_API_KEY:-}" ] && break
        fi
    done
fi

# ── Preflight checks ────────────────────────────────────────────────────────
# If 'timeout' is not available (some Git Bash installs), provide a Python fallback
if ! command -v timeout &>/dev/null; then
    timeout() {
        local secs="$1"; shift
        "$EMBER_VENV" -c "
import subprocess, sys
try:
    subprocess.run(sys.argv[1:], timeout=int(sys.argv[0]))
except subprocess.TimeoutExpired:
    sys.exit(124)
" "$secs" "$@"
    }
fi

if [ ! -f "$MAINTAIN_SCRIPT" ]; then
    echo "[dream] ERROR: maintain.py not found at $MAINTAIN_SCRIPT" >&2
    exit 1
fi

# ── Lockfile: prevent concurrent dream cycles ────────────────────────────────
if [ -f "$LOCKFILE" ]; then
    LOCK_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[dream] Another dream cycle running (PID $LOCK_PID). Skipping." >&2
        exit 0
    fi
    rm -f "$LOCKFILE"
fi

# ── Find most recent session export for distillation ──────────────────────────
# In ember3, session exports are in the SQLite database, not as files.
# Extract the most recent session-export chunk to a temp file for distillation.
SESSION_EXPORT=""
SESSION_CONTEXT=""
EMBER3_DB="$EMBER_DIR/ember3.db"
if [ -f "$EMBER3_DB" ]; then
    DISTILL_TMP="${TMPDIR:-/tmp}/ember3_distill_source.md"
    SESSION_CONTEXT=$("$EMBER_VENV" -c "
import sqlite3, sqlite_vec, os
db = sqlite3.connect('$EMBER3_DB')
db.enable_load_extension(True)
sqlite_vec.load(db)
row = db.execute(\"SELECT chunk_id, content FROM archive WHERE chunk_type IN ('session-export', 'compact-export') ORDER BY created_at DESC LIMIT 1\").fetchone()
if row:
    print(row[0])
    # Write content to temp file
    with open(os.environ.get('DISTILL_TMP', '${DISTILL_TMP}'), 'w') as f:
        f.write(row[1][:50000])  # Cap at 50k chars for distillation
db.close()
" 2>/dev/null)
    if [ -f "$DISTILL_TMP" ]; then
        SESSION_EXPORT="$DISTILL_TMP"
    fi
fi

# ── Fork the dream cycle into a detached background process ──────────────────
# This lets the hook return immediately so Claude Code can exit cleanly.
# The dream cycle runs independently, writing to DREAM_LOG.
(
    # Write our PID to lockfile (we're the background process now)
    echo $$ > "$LOCKFILE"
    trap 'rm -f "$LOCKFILE"' EXIT
    trap '' PIPE

    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    {
        echo "---"
        echo "# Dream Cycle -- $TIMESTAMP"
        echo ""

        # ── Phase 0: Session Distillation (Haiku) ────────────────────────────
        echo "## Phase 0: Session Distillation"
        echo ""

        if [ -n "$SESSION_EXPORT" ] && [ -f "$SESSION_EXPORT" ]; then
            echo "  Source: $(basename "$SESSION_EXPORT")"
            echo ""

            {
                cat "$HOOKS_DIR/distill-prompt.md"
                head -1000 "$SESSION_EXPORT"
            } | timeout 120 claude -p \
                --model haiku \
                --no-session-persistence \
                --dangerously-skip-permissions \
                --allowedTools "mcp__ember__ember_learn,mcp__ember__ember_recall" \
                2>&1

            DISTILL_EXIT=$?
            if [ $DISTILL_EXIT -ne 0 ]; then
                echo "[dream] WARNING: Distillation exited with code $DISTILL_EXIT"
            fi
        else
            echo "  No session-export chunk found. Skipping distillation."
        fi
        echo ""

        # ── Phase 0.5: Unconscious Scan ──────────────────────────────────────
        echo "## Phase 0.5: Unconscious Scan"
        echo ""
        echo "  Scanning recent sessions for topics discussed but never stored..."
        echo ""

        timeout 120 "$EMBER_VENV" "$MAINTAIN_SCRIPT" \
            --unconscious --unconscious-days 7 --unconscious-archive \
            2>&1

        UNCONSCIOUS_EXIT=$?
        if [ $UNCONSCIOUS_EXIT -ne 0 ]; then
            echo "[dream] WARNING: Unconscious scan exited with code $UNCONSCIOUS_EXIT"
        fi
        echo ""

        # ── Phase 1: Mechanical Maintenance ──────────────────────────────────
        echo "## Phase 1: Mechanical Maintenance"
        echo ""

        timeout 60 "$EMBER_VENV" "$MAINTAIN_SCRIPT" \
            --strip --prune-stale --archive-decayed --utility --vitality \
            ${SESSION_CONTEXT:+--session-context "$SESSION_CONTEXT"} \
            --report \
            2>&1

        MAINT_EXIT=$?
        if [ $MAINT_EXIT -ne 0 ]; then
            echo "[dream] WARNING: Maintenance exited with code $MAINT_EXIT"
        fi
        echo ""

        # ── Phase 1.5: Centroid Recomputation ──────────────────────────────
        # Recompute Voronoi centroids via k-means on actual ember vectors.
        # ~10s for 600 embers. Keeps cell topology semantically meaningful.
        EMBER3_DB="$EMBER_DIR/ember3.db"
        if [ -f "$EMBER3_DB" ]; then
            echo "## Phase 1.5: Centroid Recomputation"
            echo ""
            timeout 30 "$EMBER_VENV" -c "
import asyncio
async def recompute():
    from ember.db import Database
    db = Database('$EMBER3_DB')
    await db.connect()
    result = await db.recompute_centroids(k_cells=16)
    if result['status'] == 'ok':
        sizes = result['cell_sizes']
        print(f'Centroids recomputed: {result[\"iterations\"]} iterations, {result[\"embers\"]} embers')
        print(f'Cell sizes: min={min(sizes.values())}, max={max(sizes.values())}, mean={sum(sizes.values())//len(sizes)}')
    else:
        print(f'Skipped: {result[\"message\"]}')
    await db.close()
asyncio.run(recompute())
" 2>&1
            echo ""
        fi

        # ── Phase 2: Dreaming (Haiku) ────────────────────────────────────────
        echo "## Phase 2: Dreaming (Haiku)"
        echo ""

        timeout 180 claude -p \
            --model haiku \
            --no-session-persistence \
            --dangerously-skip-permissions \
            --allowedTools "mcp__ember__ember_dream_scan,mcp__ember__ember_dream_save" \
            < "$HOOKS_DIR/dream-prompt.md" \
            2>&1

        DREAM_EXIT=$?
        if [ $DREAM_EXIT -ne 0 ]; then
            echo "[dream] WARNING: Dream phase exited with code $DREAM_EXIT"
        fi
        echo ""

        # ── Phase 3: Synthesis (Sonnet) ──────────────────────────────────────
        echo "## Phase 3: Synthesis (Sonnet)"
        echo ""

        timeout 300 claude -p \
            --model sonnet \
            --no-session-persistence \
            --dangerously-skip-permissions \
            --allowedTools "mcp__ember__ember_synthesis_scan,mcp__ember__ember_synthesis_save,mcp__ember__archive_store" \
            < "$HOOKS_DIR/synthesis-prompt.md" \
            2>&1

        SYNTH_EXIT=$?
        if [ $SYNTH_EXIT -ne 0 ]; then
            echo "[dream] WARNING: Synthesis phase exited with code $SYNTH_EXIT"
        fi

        echo ""
        echo "---"
        echo "[dream] Complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$DREAM_LOG" 2>&1

    # ── Coda: Archive the dream log ──────────────────────────────────────────
    "$EMBER_VENV" "$HOOKS_DIR/archive_dream_log.py" 2>/dev/null

) </dev/null &>/dev/null &
disown

echo "[dream] Dream cycle forked (background). Log: $DREAM_LOG" >&2
exit 0
