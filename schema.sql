-- Ember 3.0 Schema: Unified SQLite with sqlite-vec + FTS5
--
-- Replaces:
--   embers/*.json        → embers table
--   vectors.faiss/.npy   → vec_embers virtual table (sqlite-vec)
--   centroids.npy        → vec_centroids virtual table (sqlite-vec)
--   archive/chunks/*.md  → archive table + archive_fts (FTS5)
--   archive/index.json   → archive metadata columns
--   cells/stats.db       → edges + region_stats tables
--   bm25s library        → FTS5 BM25 built-in

-- ============================================================================
-- Semantic Layer (Embers)
-- ============================================================================

CREATE TABLE embers (
    ember_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,                      -- JSON array: '["tag1", "tag2"]'
    cell_id INTEGER DEFAULT -1,

    -- Temporal intelligence
    importance TEXT DEFAULT 'context',  -- fact, decision, preference, context, learning
    supersedes_id TEXT,
    is_stale BOOLEAN DEFAULT 0,
    stale_reason TEXT,

    -- Access tracking
    last_accessed_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,

    -- Utility feedback (MemRL-inspired)
    utility_score REAL DEFAULT 0.5,  -- 0.0-1.0

    -- Session and source
    session_id TEXT,
    source TEXT DEFAULT 'manual',    -- manual, auto, session, dream, wonder
    source_path TEXT,
    agent TEXT,                       -- claude, gemini, codex, etc.

    -- Shadow-Decay fields
    shadow_load REAL DEFAULT 0.0,
    shadowed_by TEXT,
    shadow_updated_at TIMESTAMP,
    related_ids TEXT,                 -- JSON array: '["id1", "id2"]'
    superseded_by_id TEXT,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embers_cell_id ON embers(cell_id);
CREATE INDEX idx_embers_created_at ON embers(created_at);
CREATE INDEX idx_embers_importance ON embers(importance);
CREATE INDEX idx_embers_is_stale ON embers(is_stale);
CREATE INDEX idx_embers_tags ON embers(tags);

-- ============================================================================
-- Semantic Vectors (sqlite-vec virtual table)
-- ============================================================================

CREATE VIRTUAL TABLE vec_embers USING vec0(
    embedding float[384]
);

-- Map between embers and their embeddings
-- (vec0 uses rowid, we need ember_id -> rowid mapping)
CREATE TABLE vec_embers_map (
    rowid INTEGER PRIMARY KEY,
    ember_id TEXT UNIQUE NOT NULL REFERENCES embers(ember_id) ON DELETE CASCADE
);

-- Vector search for region centroids
CREATE VIRTUAL TABLE vec_centroids USING vec0(
    embedding float[384]
);

CREATE TABLE vec_centroids_map (
    rowid INTEGER PRIMARY KEY,
    cell_id INTEGER UNIQUE NOT NULL REFERENCES region_stats(cell_id) ON DELETE CASCADE
);

-- ============================================================================
-- Knowledge Graph (Edges)
-- ============================================================================

CREATE TABLE edges (
    source_id TEXT NOT NULL REFERENCES embers(ember_id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES embers(ember_id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,      -- relates_to, depends_on, implements, references, similar_to, contains
    weight REAL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(edge_type);

-- ============================================================================
-- Archive Layer (Sessions, Exports, Debug Records)
-- ============================================================================

CREATE TABLE archive (
    id TEXT PRIMARY KEY,
    summary TEXT,
    content TEXT NOT NULL,
    tags TEXT,                      -- JSON array: '["tag1", "tag2"]'
    project TEXT,
    domain TEXT,
    chunk_type TEXT DEFAULT 'general',  -- session, debug, snapshot, compact-export, reference, general, dream-log

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    tokens_estimate INTEGER,
    content_hash TEXT,               -- SHA1 for dedup
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP,

    status TEXT DEFAULT 'active'     -- active, archived, deprecated
);

CREATE INDEX idx_archive_chunk_type ON archive(chunk_type);
CREATE INDEX idx_archive_project ON archive(project);
CREATE INDEX idx_archive_created_at ON archive(created_at);
CREATE INDEX idx_archive_content_hash ON archive(content_hash);

-- ============================================================================
-- Archive Full-Text Search (FTS5 + BM25)
-- ============================================================================

CREATE VIRTUAL TABLE archive_fts USING fts5(
    id UNINDEXED,       -- store chunk ID without indexing
    summary,
    content,
    content = archive,
    content_rowid = rowid
);

-- Trigger to keep FTS5 index in sync
CREATE TRIGGER archive_ai AFTER INSERT ON archive BEGIN
    INSERT INTO archive_fts(rowid, id, summary, content)
    VALUES (new.rowid, new.id, new.summary, new.content);
END;

CREATE TRIGGER archive_ad AFTER DELETE ON archive BEGIN
    INSERT INTO archive_fts(archive_fts, rowid, id, summary, content)
    VALUES('delete', old.rowid, old.id, old.summary, old.content);
END;

CREATE TRIGGER archive_au AFTER UPDATE ON archive BEGIN
    INSERT INTO archive_fts(archive_fts, rowid, id, summary, content)
    VALUES('delete', old.rowid, old.id, old.summary, old.content);
    INSERT INTO archive_fts(rowid, id, summary, content)
    VALUES (new.rowid, new.id, new.summary, new.content);
END;

-- ============================================================================
-- Region Statistics (per-cell metadata)
-- ============================================================================

CREATE TABLE region_stats (
    cell_id INTEGER PRIMARY KEY,
    vitality_score REAL DEFAULT 0.0,
    shadow_accum REAL DEFAULT 0.0,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_region_stats_vitality ON region_stats(vitality_score);

-- ============================================================================
-- Metrics Log (hallucination detection, health tracking)
-- ============================================================================

CREATE TABLE metrics_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metric_type TEXT NOT NULL,      -- hallucination_risk, drift_check, health_snapshot, etc.
    value REAL,
    details TEXT,                   -- JSON for complex data
    CHECK (metric_type IN ('hallucination_risk', 'drift_check', 'health_snapshot', 'shadow_decay', 'vitality_decay'))
);

CREATE INDEX idx_metrics_log_type ON metrics_log(metric_type);
CREATE INDEX idx_metrics_log_timestamp ON metrics_log(timestamp);

-- ============================================================================
-- Migration Metadata (tracks what's been migrated)
-- ============================================================================

CREATE TABLE migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_type TEXT NOT NULL,   -- from_json, from_yaml, from_npy, etc.
    source_path TEXT,
    target_table TEXT,
    status TEXT DEFAULT 'pending',  -- pending, done, failed
    error_message TEXT,
    migrated_at TIMESTAMP,
    record_count INTEGER DEFAULT 0
);

CREATE INDEX idx_migration_log_status ON migration_log(status);

-- ============================================================================
-- Metadata (versioning, last sync, etc.)
-- ============================================================================

CREATE TABLE db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Initialize metadata
INSERT OR IGNORE INTO db_metadata (key, value) VALUES
    ('schema_version', '3.0'),
    ('embedding_model', 'all-MiniLM-L6-v2'),
    ('embedding_dimension', '384'),
    ('last_migration', ''),
    ('last_backup', '');
