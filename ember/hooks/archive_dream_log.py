#!/usr/bin/env python3
"""
Archive the dream cycle log as a searchable archive chunk.

Called at the end of the dream cycle to preserve dream history.
Deduplicates by content hash, so re-running on the same log is safe.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ember.config import get_data_dir
from ember.hooks.pre_compact_export import write_archive_chunk


def main():
    dream_log = get_data_dir() / "dream-log.md"

    if not dream_log.exists():
        return

    content = dream_log.read_text(encoding="utf-8")
    if not content.strip():
        return

    today = datetime.now().strftime("%Y-%m-%d")
    time_suffix = datetime.now().strftime("%H%M%S")

    write_archive_chunk(
        chunk_id=f"{today}_dream-log_{time_suffix}",
        summary=f"Dream cycle log {today}",
        content=content,
        tags_list=["dream-log", "dream-cycle", "maintenance"],
        project="dream-cycle",
        chunk_type="dream-log",
        from_agent="hook:dream-cycle",
    )


if __name__ == "__main__":
    main()
