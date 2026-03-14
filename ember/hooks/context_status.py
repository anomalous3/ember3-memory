#!/usr/bin/env python3
"""
Status line script: writes context remaining % to ~/.claude/context_pct
and outputs a compact status string.

Receives JSON on stdin with context_window data from Claude Code.
"""

import json
import sys
from pathlib import Path

CTX_FILE = Path.home() / ".claude" / "context_pct"


def main():
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        print("ctx:?")
        return

    ctx = data.get("context_window", {})
    remaining = ctx.get("remaining_percentage")
    used = ctx.get("used_percentage")

    if remaining is None and used is not None:
        remaining = 100 - used
    elif remaining is None:
        # Fall back to computing from raw tokens
        size = ctx.get("context_window_size", 0)
        usage = ctx.get("current_usage", {})
        if size > 0 and usage:
            current = (
                usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
            )
            remaining = max(0, int((1 - current / size) * 100))
            used = 100 - remaining

    if remaining is None:
        print("ctx:?")
        return

    # Write to file for self-compaction protocol to read
    CTX_FILE.write_text(str(remaining))

    # Status line output
    if remaining <= 10:
        indicator = "!!"
    elif remaining <= 30:
        indicator = "!"
    else:
        indicator = ""

    print(f"ctx:{remaining}%{indicator}")


if __name__ == "__main__":
    main()
