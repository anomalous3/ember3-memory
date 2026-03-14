#!/bin/bash
# Unified Stop hook: self-compaction trigger + loop management.
#
# Priority order:
#   1. Compaction signal  → fire /compact, allow stop (loop persists on disk)
#   2. Active loop        → check safeguards → block stop or clean up
#   3. Normal stop        → allow
#
# Self-exit: Claude can break a loop by deleting the config file:
#   rm ~/.claude/loop_config.json
#
# User exit: delete the config file, or use /btw (future) to interrupt.
#
# Files:
#   ~/.claude/compact_requested   — signal file for self-compaction
#   ~/.claude/loop_config.json    — loop state and safeguards
#   ~/.claude/context_pct         — context remaining % (from status line)

COMPACT_SIGNAL="$HOME/.claude/compact_requested"
LOOP_CONFIG="$HOME/.claude/loop_config.json"
CTX_FILE="$HOME/.claude/context_pct"

# ── Helper: resume loop after compaction ────────────────────────────
# Polls context_pct until it rises above $1 (compact_at threshold),
# meaning compaction completed. If it never rises (compaction blocked
# by e.g. phone remote control dialog), falls back after timeout and
# resumes the loop anyway — better low-context than silently dead.
#
# Usage: schedule_loop_resume <compact_at> &  (must be backgrounded)
schedule_loop_resume() {
    local compact_at="$1"
    sleep 10  # Give compaction time to start
    for i in $(seq 1 60); do
        sleep 5
        # Bail if loop was cancelled while we waited
        [ -f "$LOOP_CONFIG" ] || exit 0
        if [ -f "$CTX_FILE" ]; then
            new_ctx=$(cat "$CTX_FILE")
            if [ "${new_ctx:-0}" -gt "$compact_at" ] 2>/dev/null; then
                # Context recovered — compaction succeeded
                loop_task=$(jq -r '.task // "continue working"' "$LOOP_CONFIG" 2>/dev/null)
                sleep 3  # Let compaction summary finish rendering
                tmux send-keys -t "$TMUX_PANE" "Loop resuming after compaction: ${loop_task}" \
                    && sleep 0.2 && tmux send-keys -t "$TMUX_PANE" C-m
                exit 0
            fi
        fi
    done
    # Timeout: compaction likely didn't fire (tmux blocked?).
    # Resume loop anyway — Claude can self-compact or the hook will
    # catch context-low again on the next stop.
    if [ -f "$LOOP_CONFIG" ]; then
        loop_task=$(jq -r '.task // "continue working"' "$LOOP_CONFIG" 2>/dev/null)
        tmux send-keys -t "$TMUX_PANE" "Loop resuming (compaction may not have fired): ${loop_task}" \
            && sleep 0.2 && tmux send-keys -t "$TMUX_PANE" C-m
    fi
}

# ── 1. Compaction takes priority ──────────────────────────────────
# If Claude requested compaction, allow the stop and inject /compact.
# Loop config (if any) persists on disk — resumes after compaction.

if [ -f "$COMPACT_SIGNAL" ]; then
    rm "$COMPACT_SIGNAL"
    if [ -n "$TMUX_PANE" ]; then
        (sleep 2 && tmux send-keys -t "$TMUX_PANE" '/compact' && sleep 0.2 && tmux send-keys -t "$TMUX_PANE" C-m) &
        # If a loop is also active, resume it after compaction completes
        if [ -f "$LOOP_CONFIG" ]; then
            compact_at=$(jq -r '.compact_at_pct // 15' "$LOOP_CONFIG")
            schedule_loop_resume "$compact_at" &
        fi
    fi
    exit 0
fi

# ── 2. Loop management ───────────────────────────────────────────
# If a loop config exists, check safeguards. Block stop if loop should continue.
#
# Config schema (all fields optional except task):
# {
#   "task":                 "what to do each iteration",
#   "iterations_remaining": 10,        // -1 = unlimited (time limit only)
#   "time_limit_unix":      1741900000, // 0 = no time limit
#   "compact_at_pct":       15,        // trigger compaction at this context %
#   "check_command":        "curl ...", // exit 0 = done, stop loop
#   "interval_seconds":     0          // 0 = continuous, >0 = sleep between ticks
# }

if [ -f "$LOOP_CONFIG" ]; then
    task=$(jq -r '.task // "continue working"' "$LOOP_CONFIG")
    iterations=$(jq -r '.iterations_remaining // -1' "$LOOP_CONFIG")
    time_limit=$(jq -r '.time_limit_unix // 0' "$LOOP_CONFIG")
    compact_at=$(jq -r '.compact_at_pct // 15' "$LOOP_CONFIG")
    check_cmd=$(jq -r '.check_command // ""' "$LOOP_CONFIG")
    interval=$(jq -r '.interval_seconds // 0' "$LOOP_CONFIG")
    now=$(date +%s)

    # ── Finish condition (machine-checkable) ──
    if [ -n "$check_cmd" ]; then
        if eval "$check_cmd" >/dev/null 2>&1; then
            rm "$LOOP_CONFIG"
            exit 0
        fi
    fi

    # ── Iteration limit ──
    if [ "$iterations" -eq 0 ] 2>/dev/null; then
        rm "$LOOP_CONFIG"
        exit 0
    fi

    # ── Time limit ──
    if [ "$time_limit" -gt 0 ] 2>/dev/null && [ "$now" -ge "$time_limit" ]; then
        rm "$LOOP_CONFIG"
        exit 0
    fi

    # ── Context check → trigger compaction if low ──
    if [ -f "$CTX_FILE" ]; then
        ctx_pct=$(cat "$CTX_FILE")
        if [ "${ctx_pct:-100}" -le "$compact_at" ] 2>/dev/null; then
            if [ -n "$TMUX_PANE" ]; then
                (sleep 2 && tmux send-keys -t "$TMUX_PANE" '/compact' && sleep 0.2 && tmux send-keys -t "$TMUX_PANE" C-m) &
                schedule_loop_resume "$compact_at" &
            fi
            exit 0  # Allow stop for compaction; loop config persists
        fi
    fi

    # ── All safeguards pass — continue loop ──

    # Decrement iteration counter (if not unlimited)
    if [ "$iterations" -gt 0 ] 2>/dev/null; then
        new_iterations=$((iterations - 1))
        jq ".iterations_remaining = $new_iterations" "$LOOP_CONFIG" > "${LOOP_CONFIG}.tmp" \
            && mv "${LOOP_CONFIG}.tmp" "$LOOP_CONFIG"
        # If we just hit zero, end the loop
        if [ "$new_iterations" -eq 0 ]; then
            rm "$LOOP_CONFIG"
            exit 0
        fi
        remaining_msg=" ($new_iterations iterations remaining)"
    else
        remaining_msg=""
    fi

    # Interval: if set, allow stop but schedule tmux wake-up after delay
    if [ "$interval" -gt 0 ] 2>/dev/null; then
        if [ -n "$TMUX_PANE" ]; then
            (sleep "$interval" && tmux send-keys -t "$TMUX_PANE" "Loop tick: ${task}" && sleep 0.2 && tmux send-keys -t "$TMUX_PANE" C-m) &
        fi
        exit 0  # Allow stop; tmux wakes Claude after interval
    fi

    # Continuous mode: block stop
    cat <<HOOKJSON
{"decision": "block", "reason": "Loop continues${remaining_msg}: ${task}"}
HOOKJSON
    exit 0
fi

# ── 3. Normal stop ───────────────────────────────────────────────
exit 0
