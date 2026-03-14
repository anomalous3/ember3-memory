#!/usr/bin/env python3
"""
Pre-Compact Export Hook — Full session transcript preservation.

Triggered BEFORE /compact (manual or auto). Copies the current session's
JSONL conversation log, parses it into a readable markdown transcript,
and writes it as archive chunk(s) for BM25 search and line-range navigation.

The raw JSONL is also saved to the archive exports directory as a backup.

Then injects a system message guiding Claude through mindful summarization
of what should remain in the context window post-compaction.
"""

import json
import hashlib
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────

PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Ember data dir (respects EMBER_DATA_DIR env var, defaults to ~/.ember)
def _ember_data_dir() -> Path:
    env = os.environ.get("EMBER_DATA_DIR", "").strip()
    if env:
        return Path(env)
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "ember"
    return Path.home() / ".ember"

EMBER_DIR = _ember_data_dir()
ARCHIVE_DIR = EMBER_DIR / "archive"
CHUNKS_DIR = ARCHIVE_DIR / "chunks"
EXPORTS_DIR = ARCHIVE_DIR / "exports"
INDEX_PATH = ARCHIVE_DIR / "index.json"

MAX_CHUNK_BYTES = 1_800_000   # 1.8MB per chunk (under 2MB limit)
MAX_TOOL_RESULT_LINES = 200   # Truncate individual tool results in transcript
MAX_TOC_ENTRIES = 150          # Cap table of contents length

# ── i18n (inline to avoid import path issues) ─────────────────────────

LANG = os.environ.get("RLM_LANG", "en")

MESSAGES = {
    "en": {
        "compact_title": "COMPACT DETECTED - SESSION EXPORTED",
        "mindful_summary": (
            "**Mindful Summarization — what should survive compaction:**\n\n"
            "The full transcript is already saved. This is about crafting a\n"
            "compaction summary that lets you recover context *fast*.\n\n"
            "Write a concise summary structured as:\n\n"
            "1. **Active task** — What are you in the middle of? Next step?\n"
            "2. **Files in play** — Paths + line numbers + what changed\n"
            "3. **Decisions made** — What was decided and *why* (rationale)\n"
            "4. **Open threads** — Unresolved questions, blockers, follow-ups\n"
            "5. **Recovery pointers** — Specific `ember_recall` queries,\n"
            "   `archive_read` chunk IDs, or files to re-read after compaction\n\n"
            "Also promote durable facts to Ember via `ember_learn`.\n"
            "Then proceed with compaction — the transcript is your safety net."
        ),
    },
    "fr": {
        "compact_title": "COMPACT DÉTECTÉ - SESSION EXPORTÉE",
        "mindful_summary": (
            "**Résumé attentif — ce qui doit survivre au compactage :**\n\n"
            "La transcription complète est déjà sauvegardée. L'objectif est\n"
            "de rédiger un résumé de compactage pour récupérer le contexte\n"
            "*rapidement*.\n\n"
            "Écrivez un résumé concis structuré ainsi :\n\n"
            "1. **Tâche active** — En cours ? Prochaine étape ?\n"
            "2. **Fichiers en jeu** — Chemins + lignes + modifications\n"
            "3. **Décisions prises** — Quoi et *pourquoi*\n"
            "4. **Fils ouverts** — Questions non résolues, blocages\n"
            "5. **Pointeurs de récupération** — Requêtes `ember_recall`,\n"
            "   IDs `archive_read`, fichiers à relire après compactage\n\n"
            "Promouvez aussi les faits durables vers Ember via `ember_learn`.\n"
            "Puis procédez au compactage — la transcription est votre filet."
        ),
    },
}


def t(key: str) -> str:
    lang = LANG if LANG in MESSAGES else "en"
    return MESSAGES[lang].get(key, MESSAGES["en"].get(key, key))


# ── JSONL Parsing (derived from context_surgeon) ─────────────────────

def find_project_dir(cwd=None):
    """Find the Claude Code project directory for the given working dir.

    Tries the exact CWD first, then walks up parent directories. This handles
    cases where the CWD is a subdirectory of the actual
    project root.
    """
    cwd = cwd or os.getcwd()
    # Try exact match, then walk up parents
    candidates = [cwd]
    parent = Path(cwd).parent
    while str(parent) != parent.root:
        candidates.append(str(parent))
        parent = parent.parent

    for candidate in candidates:
        slug = candidate.replace("/", "-")
        project_dir = PROJECTS_DIR / slug
        if project_dir.exists():
            return project_dir
        # Try without leading dash
        slug_stripped = slug.lstrip("-")
        project_dir = PROJECTS_DIR / slug_stripped
        if project_dir.exists():
            return project_dir

    raise FileNotFoundError(f"No project dir for {cwd} (checked {len(candidates)} parents)")


def find_session_file(session_id, project_dir):
    """Find the JSONL file for a given session or the most recent one."""
    if session_id:
        jsonl = project_dir / f"{session_id}.jsonl"
        if jsonl.exists():
            return jsonl
    # Most recent by mtime
    jsonl_files = sorted(
        project_dir.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No session files in {project_dir}")
    return jsonl_files[0]


def extract_content_text(msg):
    """Extract readable text content from any JSONL message type."""
    parts = []

    if msg.get("toolUseResult"):
        result = msg["toolUseResult"]
        if isinstance(result, dict):
            r = result.get("result", "")
            parts.append(r if isinstance(r, str) else json.dumps(r, default=str))
        elif isinstance(result, str):
            parts.append(result)

    message = msg.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                c = block.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(sub.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "unknown")
                inp = block.get("input", {})
                call_parts = [f"[Tool Call: {name}]"]
                for key, val in inp.items():
                    val_str = str(val)
                    if len(val_str) > 500:
                        val_str = val_str[:500] + "..."
                    call_parts.append(f"  {key}: {val_str}")
                parts.append("\n".join(call_parts))

    data = msg.get("data", {})
    if data.get("fullOutput"):
        parts.append(data["fullOutput"])
    elif data.get("output"):
        parts.append(data["output"])

    return "\n".join(p for p in parts if p)


def build_tool_name_index(messages):
    """Map tool_use_id → tool_name from assistant messages."""
    index = {}
    for msg in messages:
        if msg.get("type") != "assistant":
            continue
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = block.get("id", "")
                name = block.get("name", "")
                if tid and name:
                    index[tid] = name
    return index


# ── Transcript Generation ─────────────────────────────────────────────

def classify_message(msg, tool_index):
    """Classify a message and return (label, detail)."""
    msg_type = msg.get("type", "unknown")

    if msg_type == "assistant":
        content = msg.get("message", {}).get("content", [])
        tool_calls = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(block.get("name", "unknown"))
        if tool_calls:
            return "Assistant + Tools", ", ".join(tool_calls)
        return "Assistant", ""

    if msg_type == "user":
        if msg.get("userType") == "external":
            return "User", ""
        content = msg.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    tool_name = tool_index.get(tool_use_id, "unknown")
                    return f"Tool Result: {tool_name}", ""
        if msg.get("toolUseResult"):
            return "Tool Result", ""
        return "User (internal)", ""

    if msg_type == "progress":
        return "Progress", ""
    if msg_type == "system":
        return "System", msg.get("subtype", "")

    return msg_type, ""


def truncate_lines(text, max_lines):
    """Truncate text to max_lines, adding a note if truncated."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    omitted = len(lines) - max_lines
    kept.append(f"\n[... {omitted} more lines — see raw JSONL export ...]")
    return "\n".join(kept)


def build_transcript(messages, tool_index):
    """Convert parsed JSONL messages into a readable markdown transcript."""
    sections = []
    toc_entries = []

    skip_types = {"progress"}
    skip_subtypes = {"compact_boundary", "microcompact_boundary"}

    msg_num = 0
    for msg in messages:
        msg_type = msg.get("type", "")

        if msg_type in skip_types:
            continue
        if msg.get("subtype", "") in skip_subtypes:
            continue

        msg_num += 1
        label, detail = classify_message(msg, tool_index)
        content = extract_content_text(msg)

        if not content.strip():
            continue

        if "Tool Result" in label:
            content = truncate_lines(content, MAX_TOOL_RESULT_LINES)

        header = f"## [{msg_num}] {label}"
        if detail:
            header += f" ({detail})"

        sections.append(f"{header}\n\n{content}\n")

        if len(toc_entries) < MAX_TOC_ENTRIES:
            preview = content.replace("\n", " ")[:80].strip()
            toc_entries.append(f"- [{msg_num}] {label}: {preview}")

    if len(toc_entries) >= MAX_TOC_ENTRIES:
        toc_entries.append(f"- ... and {msg_num - MAX_TOC_ENTRIES} more messages")

    toc = "## Table of Contents\n\n" + "\n".join(toc_entries) + "\n\n---\n\n"
    return toc, sections


# ── Archive Chunk Writing ─────────────────────────────────────────────

def write_archive_chunk(chunk_id, summary, content, tags_list, project,
                        chunk_type="compact-export", from_agent="hook:pre-compact"):
    """Write a chunk to the ember3 SQLite database.

    Uses synchronous sqlite3 (hooks run outside the async MCP server).
    Falls back to file-based archive if ember3.db doesn't exist.
    """
    import sqlite3
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    content_hash = hashlib.sha256(
        content.lower().encode("utf-8", errors="replace")
    ).hexdigest()
    tokens_est = len(content) // 4

    db_path = EMBER_DIR / "ember3.db"
    if not db_path.exists():
        # Fallback: no ember3 database yet
        print(f"[export] WARNING: {db_path} not found, skipping archive write")
        return None

    try:
        import sqlite_vec
        db = sqlite3.connect(str(db_path))
        db.enable_load_extension(True)
        sqlite_vec.load(db)
    except Exception as e:
        print(f"[export] WARNING: Could not open ember3 database: {e}")
        return None

    try:
        # Dedup check
        row = db.execute(
            "SELECT chunk_id FROM archive WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()
        if row:
            db.close()
            return row[0]  # Already exists

        # Insert archive chunk (FTS5 triggers handle indexing automatically)
        db.execute(
            """INSERT INTO archive (
                chunk_id, summary, content, tags, project, domain, chunk_type,
                from_agent, for_agent, entity, category, reply_to, status,
                tokens_estimate, content_hash, access_count, last_accessed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', '', '', 'active', ?, ?, 0, ?, ?)""",
            (
                chunk_id, summary, content, json.dumps(tags_list), project, "",
                chunk_type, from_agent, tokens_est, content_hash, now, now,
            ),
        )
        db.commit()
    except Exception as e:
        print(f"[export] WARNING: Archive write failed: {e}")
        db.close()
        return None

    db.close()
    return chunk_id


# ── Stdin Parsing ─────────────────────────────────────────────────────

def parse_stdin():
    """Read context stats and session info from stdin."""
    ctx_pct = 0
    session_id = None
    cwd = None

    try:
        input_data = sys.stdin.read()
        if not input_data:
            return ctx_pct, session_id, cwd

        data = json.loads(input_data)
        ctx_window = data.get("context_window", {})
        usage = ctx_window.get("current_usage", {})
        size = ctx_window.get("context_window_size", 1)

        if usage and size > 0:
            current = (
                usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
            )
            ctx_pct = int(current * 100 / size)

        session_id = data.get("session_id")
        cwd = data.get("cwd")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return ctx_pct, session_id, cwd


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ctx_pct, session_id, cwd = parse_stdin()

    # Find the session JSONL
    try:
        project_dir = find_project_dir(cwd)
        jsonl_path = find_session_file(session_id, project_dir)
    except FileNotFoundError as e:
        msg = (
            f"[🔄 {t('compact_title')}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⚠️ Could not locate session JSONL: {e}\n\n"
            f"{t('mindful_summary')}"
        )
        print(json.dumps({"systemMessage": msg}))
        return

    # ── Phase 1: Copy the raw JSONL ──

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    session_name = jsonl_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORTS_DIR / f"{session_name}_{timestamp}.jsonl"

    try:
        shutil.copy2(jsonl_path, export_path)
    except Exception as e:
        msg = (
            f"[🔄 {t('compact_title')}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⚠️ JSONL copy failed: {e}\n\n"
            f"{t('mindful_summary')}"
        )
        print(json.dumps({"systemMessage": msg}))
        return

    # ── Phase 2: Parse the copy into messages ──

    messages = []
    try:
        with open(export_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    continue
                try:
                    messages.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        msg = (
            f"[🔄 {t('compact_title')}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⚠️ JSONL parse failed: {e}\n"
            f"📁 Raw backup saved: `{export_path}`\n\n"
            f"{t('mindful_summary')}"
        )
        print(json.dumps({"systemMessage": msg}))
        return

    # ── Phase 3: Build readable transcript ──

    tool_index = build_tool_name_index(messages)
    toc, sections = build_transcript(messages, tool_index)

    # Determine project name
    project = (cwd or os.getcwd()).rstrip("/").split("/")[-1]
    today = datetime.now().strftime("%Y-%m-%d")
    time_suffix = datetime.now().strftime("%H%M%S")

    # ── Phase 4: Write archive chunk(s) ──

    tags = ["compact-export", "session", "full-transcript"]
    chunk_ids = []
    full_transcript = toc + "\n".join(sections)
    transcript_bytes = len(full_transcript.encode("utf-8"))

    if transcript_bytes <= MAX_CHUNK_BYTES:
        chunk_id = f"{today}_{project}_compact-export_{time_suffix}"
        write_archive_chunk(
            chunk_id=chunk_id,
            summary=(
                f"Pre-compact full session export"
                f" ({len(messages)} msgs, ctx {ctx_pct}%)"
            ),
            content=full_transcript,
            tags_list=tags,
            project=project,
        )
        chunk_ids.append(chunk_id)
    else:
        part_num = 0
        current_content = toc
        current_count = 0

        for section in sections:
            test_size = len((current_content + section).encode("utf-8"))

            if test_size > MAX_CHUNK_BYTES and current_count > 0:
                part_num += 1
                chunk_id = (
                    f"{today}_{project}_compact-export"
                    f"_{time_suffix}_p{part_num}"
                )
                write_archive_chunk(
                    chunk_id=chunk_id,
                    summary=(
                        f"Pre-compact export part {part_num}"
                        f" ({current_count} msgs, ctx {ctx_pct}%)"
                    ),
                    content=current_content,
                    tags_list=tags + [f"part-{part_num}"],
                    project=project,
                )
                chunk_ids.append(chunk_id)
                current_content = toc
                current_count = 0

            current_content += section
            current_count += 1

        if current_count > 0:
            part_num += 1
            chunk_id = (
                f"{today}_{project}_compact-export"
                f"_{time_suffix}_p{part_num}"
            )
            write_archive_chunk(
                chunk_id=chunk_id,
                summary=(
                    f"Pre-compact export part {part_num}"
                    f" ({current_count} msgs, ctx {ctx_pct}%)"
                ),
                content=current_content,
                tags_list=tags + [f"part-{part_num}"],
                project=project,
            )
            chunk_ids.append(chunk_id)

    # ── Phase 5: Build system message ──

    ctx_info = f" (ctx: {ctx_pct}%)" if ctx_pct > 0 else ""
    chunk_list = "\n".join(f"  - `{cid}`" for cid in chunk_ids)
    export_size_kb = export_path.stat().st_size / 1024

    message = (
        f"[🔄 {t('compact_title')}]{ctx_info}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"✅ **Session exported** — {len(messages)} messages →"
        f" {len(chunk_ids)} chunk(s):\n"
        f"{chunk_list}\n\n"
        f"📁 Raw JSONL: `{export_path}` ({export_size_kb:.0f} KB)\n"
        f"🔍 Searchable via `archive_search` / `archive_grep`\n"
        f"📖 Navigate via `archive_read(<chunk_id>, start=N, end=M)`\n\n"
        f"---\n\n"
        f"{t('mindful_summary')}"
    )

    print(json.dumps({"systemMessage": message}))


if __name__ == "__main__":
    main()
