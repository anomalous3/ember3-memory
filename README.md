# Ember3

Persistent memory for AI coding agents. Semantic knowledge + searchable archives in a single SQLite database, served as an MCP server.

Ember3 gives your AI assistant memory that persists across sessions, consolidates between sessions, and gets smarter over time. It works with Claude Code, OpenAI Codex, Gemini CLI, or anything that speaks MCP.

## What makes it different

- **Two-layer architecture**: Semantic memory (vector search) for durable knowledge + archive layer (full-text search) for session logs, debug records, and raw history. One database, two ways of knowing.
- **HESTIA scoring**: Retrieval ranked by cosine similarity, shadow-decay suppression, region vitality, utility feedback, and project relevance — not just vector distance. Stale knowledge auto-deprioritizes; actively-used knowledge gets boosted.
- **Dream cycle**: Memory consolidation that runs between sessions — distilling facts, detecting "unconscious" topics (discussed but never stored), bridging isolated knowledge, finding contradictions, generating questions. Your agent wakes up more coherent than when it went to sleep.
- **Bridge-ember technique**: How you *word* a memory determines its graph connectivity. A bare fact creates an island node; a relational statement creates edges in vector space. This is confirmed experimentally, not theoretical.
- **Cross-platform**: macOS, Linux, Windows. ONNX embeddings (not PyTorch), SQLite (not Postgres).
- **Agent-agnostic**: The MCP server works with any client. Claude Code gets the full experience (hooks, dream cycle); other agents get the core memory tools.

## Quick start

```bash
git clone https://github.com/anomalous3/ember3-memory.git
cd ember3-memory
./setup.sh          # creates venv, installs deps, downloads embedding model
```

Add to your Claude Code config (`~/.claude.json`):

```json
{
  "mcpServers": {
    "ember": {
      "type": "stdio",
      "command": "/path/to/ember3-memory/.venv/bin/python3",
      "args": ["-m", "ember"],
      "env": {
        "EMBER_AGENT": "claude"
      }
    }
  }
}
```

For Codex or Gemini, add the equivalent MCP config for your tool. The server speaks standard MCP stdio.

On first run, Ember auto-downloads the ONNX embedding model (~23MB) and creates the database. No API keys or external services required for the core memory system. (The dream cycle's AI-powered phases need Claude CLI with an `ANTHROPIC_API_KEY`.)

Restart your agent. Call `ember_auto("hello")` to verify it's working.

## Architecture

```
                    ┌──────────────────────────────┐
                    │         MCP Server            │
                    │        (29 tools)             │
                    └──────┬───────────┬────────────┘
                           │           │
              ┌────────────▼──┐   ┌────▼────────────┐
              │   Semantic    │   │    Archive       │
              │   Layer       │   │    Layer         │
              │               │   │                  │
              │ Vector KNN    │   │ FTS5 + regex     │
              │ HESTIA scored │   │ BM25 ranked      │
              │ Shadow decay  │   │ Chunk types      │
              └───────┬───────┘   └────┬─────────────┘
                      │                │
                      └──────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │     SQLite       │
                    │  sqlite-vec +    │
                    │  FTS5 + WAL      │
                    │  (one .db file)  │
                    └──────────────────┘
```

**Semantic layer** stores durable knowledge — facts, decisions, preferences, learnings. Each ember is embedded via all-MiniLM-L6-v2 (ONNX, 384 dimensions) and indexed with sqlite-vec for KNN search. A knowledge graph (edges table) tracks relationships. HESTIA scoring re-ranks results beyond raw vector similarity.

**Archive layer** stores everything else — session transcripts, debug records, experiment logs, dream cycle output. Indexed with FTS5 for BM25 keyword search. Also supports regex search and line-range navigation for large documents.

**`deep_recall`** searches both layers simultaneously.

One database. Backup is `cp`. Sync is file copy. Introspection is `sqlite3`.

## MCP Tools (29)

### Core memory
| Tool | What it does |
|------|-------------|
| `ember_store` | Store a new ember with content, tags, and metadata |
| `ember_recall` | Semantic search ranked by HESTIA score |
| `ember_learn` | Auto-extract and store facts from conversation context |
| `ember_contradict` | Update outdated knowledge (preserves lineage via supersession chain) |
| `ember_read` | Read full content of a specific ember |
| `ember_wonder` | Store an open question as a first-class graph citizen |
| `ember_auto` | Context-aware retrieval — call at session start |

### Archive
| Tool | What it does |
|------|-------------|
| `archive_store` | Store a chunk (session, debug, snapshot, reference, etc.) |
| `archive_search` | BM25 keyword search across all chunks |
| `archive_grep` | Regex or fuzzy pattern search |
| `archive_read` | Read a specific chunk (with line-range pagination) |
| `archive_list` | List recent chunks, filtered by type/project/tags |
| `archive_update` | Update metadata (tags, summary, status) on a chunk |
| `archive_delete` | Remove an archive chunk |
| `archive_retention` | Preview or execute retention cleanup of old chunks |
| `deep_recall` | Search both semantic and archive layers simultaneously |

### Knowledge graph
| Tool | What it does |
|------|-------------|
| `ember_graph_search` | Traverse related knowledge by hops (BFS to configurable depth) |
| `ember_explain` | Full HESTIA scoring breakdown for a specific ember |

### Dream cycle
| Tool | What it does |
|------|-------------|
| `ember_dream_scan` | Analyze graph topology, find isolated embers and bridge candidates |
| `ember_dream_save` | Store a dream-generated bridge ember (with dedup) |

### Health and maintenance
| Tool | What it does |
|------|-------------|
| `ember_health` | Hallucination risk assessment across all stored knowledge |
| `ember_drift_check` | Identify memory regions going stale or losing vitality |
| `ember_inspect` | Voronoi cell distribution, ember counts, region stats |
| `ember_recompute_centroids` | Re-run k-means to update vector space partitioning |

### Session and management
| Tool | What it does |
|------|-------------|
| `ember_save_session` | Store session summary, decisions, and next steps |
| `ember_list` | List embers with optional tag filter and pagination |
| `ember_delete` | Remove an ember |
| `ember_update_tags` | Add, remove, or replace tags on an ember |
| `ember_import_markdown` | Bulk import embers from structured markdown |

## HESTIA Scoring

Retrieval isn't just "nearest vector." HESTIA computes:

```
score = cos_sim * (1 - shadow_load)^gamma * vitality_factor * utility_factor
```

- **cos_sim**: Semantic similarity between query and memory (0-1)
- **shadow_load**: How much this memory has been superseded by newer, similar knowledge. Computed via the Shadow-Decay framework — newer memories "shadow" older ones when they're highly similar, pushing stale information down without deleting it.
- **vitality_factor**: How active the memory's region is. Knowledge in areas you're actively working in gets boosted; dormant regions are deprioritized.
- **utility_factor**: Whether this memory was actually *used* when previously surfaced. Memories surfaced but never read decay; memories read and acted on get boosted. Tracked via the recall log.

Additional modifiers:
- **Project scoping**: Memories tagged with the active project get a boost; unrelated project memories get a penalty.
- **Importance-based half-lives**: Facts decay slowly (365 days), context decays fast (7 days), decisions (30 days), preferences (60 days), learnings (90 days).

## The Dream Cycle

The dream cycle runs as a Claude Code `SessionEnd` hook, forked into a detached background process so it doesn't block shutdown. Total runtime is typically 3-8 minutes.

| Phase | What it does | Engine |
|-------|-------------|--------|
| **0: Session Distillation** | Read the just-ended session transcript, extract 3-7 durable facts, store as embers | Claude Haiku |
| **0.5: Unconscious Scan** | Compare recent session content against the ember graph to find topics discussed but never stored | Python + sqlite-vec |
| **1: Mechanical Maintenance** | Compute utility scores, update vitality, prune stale embers, archive heavily-shadowed ones | Python (no API) |
| **1.5: Centroid Recomputation** | Re-run k-means on ember vectors to keep Voronoi cell topology semantically meaningful | Python (no API) |
| **2: Dreaming** | Find isolated memories and write associative paragraphs connecting them to well-connected ones | Claude Haiku |
| **3: Synthesis** | Theme emergence, contradiction resolution, duplicate consolidation, wonder generation, foundation audit | Claude Sonnet |
| **Coda: Archival** | Save the dream log as a searchable archive chunk | Python |

Phases 0, 2, and 3 use Claude CLI (`claude -p`) with restricted tool access — the dreaming AI can only use specific Ember tools, not arbitrary system access. Each phase has a timeout (60-300s) so a stuck phase doesn't block the rest.

The dream cycle is optional and Claude Code specific. The core memory system works without it.

## Key Concepts

### Bridge-ember technique

How you word a memory determines its graph connectivity. Semantic embeddings encode relationships, not just topics. Storing connective phrasing literally creates edges in vector space:

```
Isolated:   "User has RTX 3090"                    -> 0 graph connections (island)
Connected:  "RTX 3090 enables local model training  -> 18 connections via 2 hops
             for discriminator models, connecting
             the multi-agent architecture to
             on-device compute"
```

Template: `[fact] -- [why it matters] in the context of [what it connects to], which means [implication for future work]`

### Three gates before storing

1. **Is this durable?** Will it be true next week? If not → archive, not ember.
2. **Does a similar ember exist?** Check first. Update, don't duplicate.
3. **Is it worded as a relationship?** Connect it to what it relates to.

### Store aggressively, search judiciously

Storage is cheap; lost context is expensive. Store everything. But be selective about what you load back into context — only recall what's needed for the current task.

## Multi-agent support

Multiple agents can share a single Ember database. Each agent identifies itself via the `EMBER_AGENT` env var, which is recorded on every operation for provenance.

| Agent | Core memory | Archive | Dream cycle | Session hooks |
|-------|------------|---------|-------------|---------------|
| Claude Code | Full | Full | Full | Full |
| Codex CLI | Full | Full | No | No |
| Gemini CLI | Full | Full | No | No |
| Any MCP client | Full | Full | No | No |

The dream cycle and session hooks are Claude Code features (they use Claude Code's hook system). The 29 MCP tools work with any client.

## Advanced Patterns

The `examples/` directory covers techniques for getting more out of Ember:

- **[CLAUDE.md.example](examples/CLAUDE.md.example)** -- Starter configuration for teaching your agent to use Ember effectively. Session protocol, three gates, bridge-ember technique, attention bell practice.
- **[subagent-recipes.md](examples/subagent-recipes.md)** -- Using ember loading as context shaping: presence-first routing (when to experience directly vs. delegate), model cascade, loading recipes for different cognitive modes, externalization protocols. Includes findings on why over-delegation degrades judgment quality.
- **[meta-embers.md](examples/meta-embers.md)** -- Self-advising knowledge: store instructions as embers that surface alongside the knowledge they advise about. The graph advises itself through the LLM. Lightweight recursive self-modification without server changes.

## Configuration

### Environment variables

| Env var | Default | Purpose |
|---------|---------|---------|
| `EMBER_DATA_DIR` | `~/.ember` (Mac/Linux), `%APPDATA%/ember` (Windows) | Data directory for database, models, logs |
| `EMBER_AGENT` | `""` | Agent identifier stored with memories for provenance |
| `ANTHROPIC_API_KEY` | *(none)* | Required for dream cycle phases that use Claude CLI |

### Data directory layout

```
~/.ember/                          # or %APPDATA%/ember on Windows
├── ember3.db                      # The single database file
├── models/
│   └── all-MiniLM-L6-v2/
│       ├── model.onnx             # Embedding model (~23MB, auto-downloaded)
│       └── tokenizer.json
├── dream-log.md                   # Most recent dream cycle output
└── archive/
    └── exports/                   # Raw JSONL session backups
```

### Tunable constants

Edit `ember/config.py` to adjust scoring behavior:

| Constant | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | 0.4 | Minimum cosine similarity for knowledge graph edges |
| `SHADOW_DELTA` | 0.3 | Width of the shadow activation window |
| `SHADOW_GAMMA` | 2.0 | Exponent controlling shadow suppression strength |
| `PROJECT_BOOST` | 0.5 | Score boost for same-project memories |
| `PROJECT_PENALTY` | 0.7 | Score multiplier for other-project memories |
| `UTILITY_WEIGHT` | 0.15 | How much utility feedback affects ranking |
| `SHADOW_ARCHIVE_THRESHOLD` | 0.95 | Shadow load above which embers auto-archive |

## Hooks (Claude Code, Unix/macOS)

Ember ships with hooks for Claude Code's lifecycle events. Hooks require Unix-like systems (macOS, Linux) — the core MCP server is cross-platform, but hooks use `os.fork()`, bash, and optionally tmux/jq:

| Hook | Event | What it does |
|------|-------|-------------|
| `session_end_export.py` | SessionEnd | Save full session transcript as archive chunk |
| `pre_compact_export.py` | PreCompact | Save transcript before context compaction |
| `session_end_dream.sh` | SessionEnd | Run the dream cycle |
| `context_status.py` | StatusLine | Show remaining context % |
| `unified_stop_hook.sh` | Stop | Self-compaction trigger + autonomous loops |

Example hook configuration in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "type": "command",
        "command": "/path/to/ember3/.venv/bin/python3 /path/to/ember3/ember/hooks/session_end_export.py"
      },
      {
        "type": "command",
        "command": "/path/to/ember3/ember/hooks/session_end_dream.sh"
      }
    ],
    "PreCompact": [
      {
        "type": "command",
        "command": "/path/to/ember3/.venv/bin/python3 /path/to/ember3/ember/hooks/pre_compact_export.py"
      }
    ]
  }
}
```

## Maintenance

`maintain.py` runs memory housekeeping (the non-AI phases of the dream cycle):

```bash
python maintain.py                    # Report only (default)
python maintain.py --all              # Run all maintenance tasks
python maintain.py --utility          # Compute utility scores from recall logs
python maintain.py --vitality         # Update per-cell vitality
python maintain.py --prune-stale      # Delete embers marked as stale
python maintain.py --archive-decayed  # Archive heavily-shadowed embers
python maintain.py --unconscious      # Detect topics discussed but never stored
python maintain.py --report           # Print full maintenance report
```

## Dependencies

- Python >= 3.10
- [mcp](https://pypi.org/project/mcp/) >= 1.0.0 -- MCP server framework
- [onnxruntime](https://pypi.org/project/onnxruntime/) >= 1.17.0 -- Embedding inference
- [tokenizers](https://pypi.org/project/tokenizers/) >= 0.15.0 -- Fast tokenization
- [sqlite-vec](https://pypi.org/project/sqlite-vec/) ~= 0.1.6 -- Vector search in SQLite
- [numpy](https://pypi.org/project/numpy/) >= 1.24.0 -- Vector operations
- [aiosqlite](https://pypi.org/project/aiosqlite/) >= 0.20.0 -- Async SQLite
- [huggingface-hub](https://pypi.org/project/huggingface-hub/) >= 0.20.0 -- Model download

Optional: [thefuzz](https://pypi.org/project/thefuzz/) for fuzzy string matching in dedup.

## Lineage

Ember3 originated as a fork of [Arkya-AI/ember-mcp](https://github.com/Arkya-AI/ember-mcp) (no longer available). The original provided basic semantic memory with FAISS indexing. From there, development diverged substantially:

- **Ember 1.x** (fork) -- JSON files per memory, FAISS index, numpy vector storage, separate BM25 library for archive search, YAML-based archive chunks. Functional but fragile — state scattered across a dozen file formats.
- **Ember 2.x** -- Introduced HESTIA scoring, shadow-decay framework, the dream cycle, Voronoi cell topology, utility feedback, bridge-ember technique, and the archive layer. The ideas that define the system.
- **Ember 3.0** -- Ground-up storage rewrite. One SQLite database. sqlite-vec replaces FAISS. FTS5 replaces bm25s. Same 29 MCP tools, same dream cycle. Backup is one file copy, introspection is `sqlite3`.

The continuity of ideas across versions matters more than the continuity of code. Lineage, not reincarnation.

## License

MIT
