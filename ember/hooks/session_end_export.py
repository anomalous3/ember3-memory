#!/usr/bin/env python3
"""
Session End Export Hook — Save session transcript on exit.

Triggered at SessionEnd. Reuses the JSONL parsing and transcript building
from pre_compact_export.py but writes with chunk_type "session-export"
and produces no systemMessage (session is ending, nobody to read it).

Deduplicates against any pre-compact export that already saved this session.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Import shared logic from the pre-compact module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ember.hooks.pre_compact_export import (
    find_project_dir,
    find_session_file,
    build_tool_name_index,
    build_transcript,
    write_archive_chunk,
    EXPORTS_DIR,
    MAX_CHUNK_BYTES,
)


def parse_stdin():
    """Read session info from stdin (SessionEnd hook format)."""
    session_id = None
    cwd = None
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            session_id = data.get("session_id")
            cwd = data.get("cwd")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return session_id, cwd


def main():
    session_id, cwd = parse_stdin()

    # Fork into background so Claude Code can exit without killing us.
    # Stdin is already consumed, so the child has everything it needs.
    pid = os.fork()
    if pid > 0:
        # Parent returns immediately — Claude Code sees clean exit
        return
    # Child continues in background
    os.setsid()  # Detach from terminal

    try:
        project_dir = find_project_dir(cwd)
        jsonl_path = find_session_file(session_id, project_dir)
    except FileNotFoundError:
        return  # No session to save, exit silently

    # Copy the raw JSONL
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    session_name = jsonl_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORTS_DIR / f"{session_name}_{timestamp}.jsonl"

    try:
        shutil.copy2(jsonl_path, export_path)
    except Exception:
        return  # Silent failure — session is ending

    # Parse messages
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
    except Exception:
        return

    if not messages:
        return

    # Build transcript
    tool_index = build_tool_name_index(messages)
    toc, sections = build_transcript(messages, tool_index)

    project = (cwd or os.getcwd()).rstrip("/").split("/")[-1]
    today = datetime.now().strftime("%Y-%m-%d")
    time_suffix = datetime.now().strftime("%H%M%S")

    tags = ["session-export", "session", "full-transcript"]
    full_transcript = toc + "\n".join(sections)
    transcript_bytes = len(full_transcript.encode("utf-8"))

    if transcript_bytes <= MAX_CHUNK_BYTES:
        chunk_id = f"{today}_{project}_session-export_{time_suffix}"
        write_archive_chunk(
            chunk_id=chunk_id,
            summary=f"Session end export ({len(messages)} msgs)",
            content=full_transcript,
            tags_list=tags,
            project=project,
            chunk_type="session-export",
            from_agent="hook:session-end",
        )
    else:
        part_num = 0
        current_content = toc
        current_count = 0

        for section in sections:
            test_size = len((current_content + section).encode("utf-8"))

            if test_size > MAX_CHUNK_BYTES and current_count > 0:
                part_num += 1
                chunk_id = (
                    f"{today}_{project}_session-export"
                    f"_{time_suffix}_p{part_num}"
                )
                write_archive_chunk(
                    chunk_id=chunk_id,
                    summary=f"Session end export part {part_num} ({current_count} msgs)",
                    content=current_content,
                    tags_list=tags + [f"part-{part_num}"],
                    project=project,
                    chunk_type="session-export",
                    from_agent="hook:session-end",
                )
                current_content = toc
                current_count = 0

            current_content += section
            current_count += 1

        if current_count > 0:
            part_num += 1
            chunk_id = (
                f"{today}_{project}_session-export"
                f"_{time_suffix}_p{part_num}"
            )
            write_archive_chunk(
                chunk_id=chunk_id,
                summary=f"Session end export part {part_num} ({current_count} msgs)",
                content=current_content,
                tags_list=tags + [f"part-{part_num}"],
                project=project,
            )

    # No systemMessage — session is ending, no one to read it


if __name__ == "__main__":
    main()
