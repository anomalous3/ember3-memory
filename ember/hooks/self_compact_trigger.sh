#!/bin/bash
# Stop hook: triggers self-compaction if signal file exists.
# Claude writes ~/.claude/compact_requested, then finishes its turn.
# This hook fires on turn end, backgrounds a delayed tmux send-keys
# so /compact arrives after the input prompt is ready.

SIGNAL="$HOME/.claude/compact_requested"

if [ -f "$SIGNAL" ]; then
    rm "$SIGNAL"
    if [ -n "$TMUX_PANE" ]; then
        (sleep 2 && tmux send-keys -t "$TMUX_PANE" '/compact' && sleep 0.2 && tmux send-keys -t "$TMUX_PANE" C-m) &
    fi
fi
