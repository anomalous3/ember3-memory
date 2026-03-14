#!/usr/bin/env python3
"""Ember 3 Maintenance — SQL-native maintenance tasks.

Replaces the file-based maintain.py from ember-full. All operations
work directly on the ember3 SQLite database.

Usage:
  python maintain.py                          # Report only
  python maintain.py --utility --vitality     # Compute scores
  python maintain.py --prune-stale            # Delete stale embers
  python maintain.py --archive-decayed        # Archive heavily-shadowed embers
  python maintain.py --report                 # Full report
  python maintain.py --all                    # Everything
"""

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure ember package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ember.config import get_data_dir, get_db_path, SHADOW_ARCHIVE_THRESHOLD, K_CELLS
from ember.db import Database


async def compute_utility(db: Database, days: int = 7) -> dict:
    """Compute utility scores from recall_log.

    Embers surfaced AND read get utility boost.
    Embers surfaced but NOT read get utility decay.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Get recall events from recent sessions
    cursor = await db.conn.execute(
        "SELECT ember_id, event_type FROM recall_log WHERE timestamp >= ?",
        (cutoff,),
    )
    events = await cursor.fetchall()

    if not events:
        return {"status": "no_events", "updated": 0}

    # Aggregate: surfaced vs read
    surfaced = set()
    read = set()
    for row in events:
        eid = row["ember_id"]
        if row["event_type"] == "surfaced":
            surfaced.add(eid)
        elif row["event_type"] == "read":
            read.add(eid)
        elif row["event_type"] == "graph":
            read.add(eid)  # Graph traversal counts as engagement

    updated = 0
    # Boost embers that were read after being surfaced
    for eid in surfaced & read:
        ember = await db.get_ember(eid)
        if ember:
            new_utility = min(1.0, ember["utility_score"] + 0.05)
            if new_utility != ember["utility_score"]:
                await db.update_ember(eid, utility_score=new_utility)
                updated += 1

    # Decay embers surfaced but never engaged with
    for eid in surfaced - read:
        ember = await db.get_ember(eid)
        if ember:
            new_utility = max(0.0, ember["utility_score"] - 0.02)
            if new_utility != ember["utility_score"]:
                await db.update_ember(eid, utility_score=new_utility)
                updated += 1

    await db.conn.commit()
    return {
        "status": "ok",
        "surfaced": len(surfaced),
        "read": len(read),
        "boosted": len(surfaced & read),
        "decayed": len(surfaced - read),
        "updated": updated,
    }


async def compute_vitality(db: Database, days: int = 14) -> dict:
    """Compute per-cell vitality from recent ember access patterns.

    Vitality = sum of recency-weighted access events in each cell.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    now = datetime.now(timezone.utc)
    lambda_decay = 0.05  # Per-day decay rate

    # Get recently accessed embers with their cells
    cursor = await db.conn.execute(
        """SELECT e.cell_id, e.last_accessed_at
           FROM embers e
           WHERE e.last_accessed_at >= ? AND e.is_stale = 0""",
        (cutoff,),
    )
    rows = await cursor.fetchall()

    cell_vitality: dict[int, float] = {}
    for row in rows:
        cell_id = row["cell_id"]
        accessed = datetime.fromisoformat(
            row["last_accessed_at"].replace("Z", "+00:00")
        )
        age_days = (now - accessed).total_seconds() / 86400.0
        contribution = math.exp(-lambda_decay * age_days)
        cell_vitality[cell_id] = cell_vitality.get(cell_id, 0.0) + contribution

    # Update region_stats
    updated = 0
    for cell_id in range(K_CELLS):
        vitality = cell_vitality.get(cell_id, 0.0)
        stats = (await db.get_all_region_stats()).get(cell_id, {})
        await db.update_region(
            cell_id, vitality, stats.get("shadow_accum", 0.0)
        )
        updated += 1

    return {
        "status": "ok",
        "cells_updated": updated,
        "active_cells": len(cell_vitality),
        "max_vitality": max(cell_vitality.values()) if cell_vitality else 0.0,
    }


async def prune_stale(db: Database) -> dict:
    """Delete embers marked as stale."""
    cursor = await db.conn.execute(
        "SELECT ember_id, name FROM embers WHERE is_stale = 1"
    )
    stale = await cursor.fetchall()

    deleted = 0
    for row in stale:
        await db.delete_ember(row["ember_id"])
        deleted += 1

    return {"status": "ok", "deleted": deleted}


async def archive_decayed(db: Database) -> dict:
    """Archive embers with shadow_load >= threshold."""
    cursor = await db.conn.execute(
        "SELECT ember_id, name, content, tags, shadow_load FROM embers WHERE shadow_load >= ? AND is_stale = 0",
        (SHADOW_ARCHIVE_THRESHOLD,),
    )
    rows = await cursor.fetchall()

    archived = 0
    for row in rows:
        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
        tags_with_meta = tags + ["decayed", "auto-archived"]

        project = ""
        for tag in tags:
            if tag.startswith("project:"):
                project = tag[len("project:"):]
                break

        chunk_id = await db.archive_next_chunk_id(
            project or "memory-archive", "general"
        )
        result = await db.archive_store(
            chunk_id=chunk_id,
            content=row["content"],
            summary=f"[decayed] {row['name']}",
            tags=tags_with_meta,
            project=project or "memory-archive",
            domain="decayed-embers",
        )

        if result["status"] == "ok":
            # Mark as stale after archiving
            await db.update_ember(
                row["ember_id"], content_changed=True,
                is_stale=True,
                stale_reason="Archived due to heavy shadowing",
            )
            archived += 1

    return {"status": "ok", "archived": archived}


async def detect_unconscious(
    db: Database, days: int = 7, min_words: int = 20, threshold: float = 0.35,
    archive_results: bool = False,
) -> dict:
    """Detect topics discussed in sessions but never stored as embers.

    Compares recent archive content (raw experience) against vec_embers
    (stored knowledge) to find what was discussed but never remembered.

    Uses sqlite-vec for comparison — much simpler than the old FAISS approach.
    """
    from ember.embedder import Embedder, serialize_vector
    import re
    import numpy as np

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Get recent session chunks
    cursor = await db.conn.execute(
        """SELECT chunk_id, content FROM archive
           WHERE chunk_type IN ('session-export', 'compact-export', 'session')
           AND created_at >= ? AND status = 'active'
           ORDER BY created_at DESC LIMIT 20""",
        (cutoff,),
    )
    chunks = await cursor.fetchall()
    if not chunks:
        return {"status": "no_chunks", "unconscious": []}

    # Extract meaningful text segments from chunks
    MAX_SEGMENTS = 500  # Cap to keep dream cycle under 2 minutes

    segments = []
    for chunk in chunks:
        content = chunk["content"]
        # Split into paragraphs, filter out code/tool output/short lines
        for para in re.split(r'\n\n+', content):
            if len(segments) >= MAX_SEGMENTS:
                break
            para = para.strip()
            # Skip code blocks
            if para.startswith('```') or para.startswith('    '):
                continue
            # Skip tool output markers
            if any(para.startswith(p) for p in ['Tool:', 'Result:', '> ', '│', '┌', '└', '├']):
                continue
            # Skip lines that look like JSON, paths, or structured data
            if para.startswith('{') or para.startswith('[') or para.startswith('/'):
                continue
            # Skip very short segments
            words = para.split()
            if len(words) < min_words:
                continue
            # Skip segments that are mostly punctuation/symbols
            alpha_ratio = sum(1 for c in para if c.isalpha()) / max(len(para), 1)
            if alpha_ratio < 0.5:
                continue
            segments.append(para[:500])  # Cap segment length
        if len(segments) >= MAX_SEGMENTS:
            break

    if not segments:
        return {"status": "no_segments", "unconscious": []}

    # Embed segments and compare against vec_embers
    embedder = Embedder()
    unconscious = []

    # Process in batches to avoid overwhelming memory
    batch_size = 32
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        vectors = embedder.batch_embed(batch)

        for seg, vec in zip(batch, vectors):
            emb = serialize_vector(vec)
            # Find nearest ember
            results = await db.search_embers(emb, k=1)
            if not results:
                unconscious.append({"text": seg, "max_sim": 0.0})
                continue

            _, dist = results[0]
            cos_sim = 1.0 - (dist / 2.0)  # L2 normalized

            if cos_sim < threshold:
                unconscious.append({
                    "text": seg[:200],
                    "max_sim": round(cos_sim, 3),
                })

    # Deduplicate similar unconscious segments
    unique = []
    for item in unconscious:
        is_dup = False
        for existing in unique:
            # Simple word overlap check
            words_a = set(item["text"].lower().split())
            words_b = set(existing["text"].lower().split())
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap > 0.6:
                is_dup = True
                break
        if not is_dup:
            unique.append(item)

    # Sort by lowest similarity (most "unconscious")
    unique.sort(key=lambda x: x["max_sim"])

    # Archive results if requested
    if archive_results and unique:
        report_lines = [f"Unconscious Scan — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                       f"Scanned {len(chunks)} chunks, {len(segments)} segments",
                       f"Found {len(unique)} unconscious topics (threshold: {threshold})", ""]
        for item in unique[:20]:
            report_lines.append(f"[sim={item['max_sim']:.3f}] {item['text']}")
            report_lines.append("")

        chunk_id = await db.archive_next_chunk_id("memory", "subconscious")
        await db.archive_store(
            chunk_id=chunk_id,
            content="\n".join(report_lines),
            summary=f"Unconscious scan: {len(unique)} topics found",
            tags=["unconscious", "dream-cycle", "maintenance"],
            project="memory",
            chunk_type="subconscious",
            from_agent="maintain.py",
        )

    return {
        "status": "ok",
        "chunks_scanned": len(chunks),
        "segments_extracted": len(segments),
        "unconscious_count": len(unique),
        "unconscious": unique[:15],  # Top 15 most unconscious
    }


async def report(db: Database) -> str:
    """Generate maintenance report."""
    lines = []

    # Counts
    total = await db.count_embers()
    stale_cursor = await db.conn.execute("SELECT COUNT(*) FROM embers WHERE is_stale = 1")
    stale = (await stale_cursor.fetchone())[0]
    shadow_cursor = await db.conn.execute(
        f"SELECT COUNT(*) FROM embers WHERE shadow_load >= {SHADOW_ARCHIVE_THRESHOLD}"
    )
    heavily_shadowed = (await shadow_cursor.fetchone())[0]
    edge_cursor = await db.conn.execute("SELECT COUNT(*) FROM edges")
    edges = (await edge_cursor.fetchone())[0]
    archive_cursor = await db.conn.execute("SELECT COUNT(*) FROM archive WHERE status = 'active'")
    archives = (await archive_cursor.fetchone())[0]

    lines.append(f"── Ember 3 Maintenance Report ──")
    lines.append(f"  Embers: {total} ({stale} stale, {heavily_shadowed} heavily shadowed)")
    lines.append(f"  Edges: {edges} (avg {edges/max(total,1):.1f}/ember)")
    lines.append(f"  Archive: {archives} active chunks")

    # Cell distribution
    cell_counts: dict[int, int] = {}
    cursor = await db.conn.execute("SELECT cell_id, COUNT(*) FROM embers WHERE is_stale = 0 GROUP BY cell_id")
    for row in await cursor.fetchall():
        cell_counts[row[0]] = row[1]

    lines.append(f"\n  Cell distribution:")
    for i in range(K_CELLS):
        count = cell_counts.get(i, 0)
        bar = "█" * min(count, 25)
        lines.append(f"    Cell {i:2d}: {bar} {count}")

    # Vitality
    stats = await db.get_all_region_stats()
    if stats:
        vitalities = [s.get("vitality_score", 0.0) for s in stats.values()]
        lines.append(f"\n  Vitality: min={min(vitalities):.3f} max={max(vitalities):.3f} avg={sum(vitalities)/len(vitalities):.3f}")

    # Recall log stats
    recall_cursor = await db.conn.execute("SELECT COUNT(*) FROM recall_log")
    recall_count = (await recall_cursor.fetchone())[0]
    lines.append(f"  Recall log: {recall_count} events")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Ember 3 Maintenance")
    parser.add_argument("--utility", action="store_true", help="Compute utility scores from recall logs")
    parser.add_argument("--vitality", action="store_true", help="Compute per-cell vitality")
    parser.add_argument("--prune-stale", action="store_true", help="Delete stale embers")
    parser.add_argument("--archive-decayed", action="store_true", help="Archive heavily-shadowed embers")
    parser.add_argument("--report", action="store_true", help="Generate maintenance report")
    parser.add_argument("--session-context", type=str, default="", help="Session context (unused, for compat)")
    parser.add_argument("--unconscious", action="store_true", help="Detect topics discussed but never stored")
    parser.add_argument("--unconscious-days", type=int, default=7, help="Days to scan (default 7)")
    parser.add_argument("--unconscious-archive", action="store_true", help="Archive unconscious scan results")
    parser.add_argument("--unconscious-classify", action="store_true", help="(compat, no-op) Classify unconscious")
    parser.add_argument("--all", action="store_true", help="Run all maintenance tasks")
    # Compat flags (no-ops in ember3)
    parser.add_argument("--strip", action="store_true", help="(no-op) Strip YAML headers")

    args = parser.parse_args()

    # Default to report if no flags
    if not any([args.utility, args.vitality, args.prune_stale, args.archive_decayed, args.report, args.all, args.strip, args.unconscious]):
        args.report = True

    if args.all:
        args.utility = args.vitality = args.prune_stale = args.archive_decayed = args.report = True

    db = Database()
    await db.connect()

    try:
        if args.strip:
            print("── Stripping YAML Headers ──")
            print("No-op in ember3 (no YAML files).")
            print()

        if args.utility:
            print("── Utility Scores ──")
            result = await compute_utility(db)
            if result["status"] == "ok":
                print(f"  Surfaced: {result['surfaced']}, Read: {result['read']}")
                print(f"  Boosted: {result['boosted']}, Decayed: {result['decayed']}")
                print(f"  Updated: {result['updated']} embers")
            else:
                print(f"  {result['status']}: {result.get('updated', 0)} updated")
            print()

        if args.vitality:
            print("── Region Vitality ──")
            result = await compute_vitality(db)
            print(f"  Active cells: {result['active_cells']}/{result['cells_updated']}")
            print(f"  Max vitality: {result['max_vitality']:.3f}")
            print()

        if args.prune_stale:
            print("── Pruning Stale Embers ──")
            result = await prune_stale(db)
            print(f"  Deleted: {result['deleted']} stale embers")
            print()

        if args.archive_decayed:
            print("── Archiving Decayed Embers ──")
            result = await archive_decayed(db)
            print(f"  Archived: {result['archived']} heavily-shadowed embers")
            print()

        if args.unconscious:
            print("── Unconscious Scan ──")
            result = await detect_unconscious(
                db, days=args.unconscious_days,
                archive_results=args.unconscious_archive,
            )
            if result["status"] == "ok":
                print(f"  Scanned: {result['chunks_scanned']} chunks, {result['segments_extracted']} segments")
                print(f"  Unconscious topics: {result['unconscious_count']}")
                for item in result["unconscious"][:10]:
                    print(f"    [sim={item['max_sim']:.3f}] {item['text'][:80]}...")
            else:
                print(f"  {result['status']}")
            print()

        if args.report:
            print(await report(db))
            print()

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
