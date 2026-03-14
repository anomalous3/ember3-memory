"""Configuration constants and path resolution for Ember 3."""

import os
import sys
from pathlib import Path

# ── Vector / Model ────────────────────────────────────────────────────────────
DIMENSION = 384
K_CELLS = 16
HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"

# ── HESTIA Scoring ────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.4
SHADOW_DELTA = 0.3
SHADOW_EPSILON = 0.05
SHADOW_GAMMA = 2.0
NOSTALGIA_ALPHA = 0.1
SHADOW_K = 10

# ── Project Scoping ──────────────────────────────────────────────────────────
PROJECT_BOOST = 0.5
PROJECT_PENALTY = 0.7

# ── Utility Feedback ──────────────────────────────────────────────────────────
UTILITY_WEIGHT = 0.15

# ── Shadow Archive ────────────────────────────────────────────────────────────
SHADOW_ARCHIVE_THRESHOLD = 0.95

# ── Archive ───────────────────────────────────────────────────────────────────
MAX_CONTENT_SIZE = 2 * 1024 * 1024  # 2 MB
DEFAULT_RETENTION_DAYS = 30
SUBCONSCIOUS_RETENTION_DAYS = 14

CHUNK_TYPES = {
    "session", "debug", "snapshot", "compact-export",
    "subconscious", "reference", "general", "dream-log",
}

# ── Decay Half-Lives (days) by importance ─────────────────────────────────────
DECAY_HALF_LIVES = {
    "fact": 365.0,
    "decision": 30.0,
    "preference": 60.0,
    "context": 7.0,
    "learning": 90.0,
}

# ── Preview ───────────────────────────────────────────────────────────────────
PREVIEW_CHARS = 150


# ── Path Resolution ──────────────────────────────────────────────────────────

def get_data_dir() -> Path:
    """Determine the data directory, platform-aware.

    Resolution order:
    1. EMBER_DATA_DIR env var (explicit override)
    2. %APPDATA%/ember on Windows
    3. ~/.ember everywhere else
    """
    env = os.environ.get("EMBER_DATA_DIR", "").strip()
    if env:
        return Path(env)
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "ember"
    return Path.home() / ".ember"


def get_db_path() -> Path:
    """Path to the unified SQLite database."""
    return get_data_dir() / "ember3.db"


def get_model_dir() -> Path:
    """Path to the ONNX model directory."""
    return get_data_dir() / "models" / "all-MiniLM-L6-v2"
