# Meta-Embers: Self-Advising Knowledge

Meta-embers are self-instructions stored as regular embers with a `meta` tag.
When they surface alongside other knowledge via HESTIA retrieval, the agent
acts on the instruction. The graph advises itself through the LLM.

This is the lightweight version of recursive self-modification — no server
changes needed, just tag convention and discipline.

## How It Works

1. You store an instruction as an ember, tagged `meta`
2. The instruction is embedded in the same vector space as regular knowledge
3. When a query is semantically near the instruction, HESTIA surfaces it
4. The agent reads the instruction alongside the knowledge it advises about
5. The agent follows the instruction (or doesn't — it's advisory, not enforced)

The key insight: **the instruction surfaces exactly when it's relevant**,
because it's embedded near the knowledge it advises about. You don't need
a rule engine or conditional logic — semantic proximity does the routing.

## Examples

### Cross-Domain Hints

```
META: When cymatics-related embers surface, also check strange-loops —
they connect more deeply than cosine similarity suggests.
```

This surfaces when you're working on cymatics, reminding you to also look
at strange-loop knowledge that might not be semantically close enough to
surface on its own.

### Anti-Duplication Guards

```
META: I tend to over-store neuroscience analogs. Check for duplicates
before creating new embers in this domain.
```

This surfaces when you're about to store something neuroscience-related,
reminding you to check for existing similar embers first.

### Scoring Feedback

```
META: The project penalty of 0.3x feels aggressive for cross-domain
bridges during exploration sessions. Consider requesting a temporary
boost when exploring.
```

This surfaces during exploration work, suggesting that the HESTIA scoring
parameters might need temporary adjustment.

### Workflow Reminders

```
META: After running ML experiments, always store both the technique-level
insight (ember) and the raw training log (archive). The insight is what
future sessions need; the log is what debugging needs.
```

### Relationship Hints

```
META: Authentication and session-management embers are more related than
their embeddings suggest. When one surfaces, manually check the other
via ember_graph_search.
```

## Convention

- **Tag:** Always include `meta` in the tags
- **Prefix:** Content should start with `META:` for visual discoverability
  when scanning ember lists
- **Tone:** Write as advice to a future version of yourself. Be specific
  enough to act on, not so specific it only applies once.
- **Scope:** One instruction per ember. Don't bundle multiple pieces of
  advice — each should surface independently when relevant.

## Creating Meta-Embers

Use `ember_store` or `ember_learn`:

```
ember_store(
    content="META: When working on the API layer, check archive for recent
             error patterns before proposing changes. Past debugging sessions
             often reveal constraints that aren't in the code.",
    tags=["meta", "api", "debugging"],
    ember_type="learning"
)
```

The tags serve double duty: `meta` marks it as a self-instruction, while
`api` and `debugging` help it surface in the right contexts via both tag
filtering and semantic similarity.

## Scoring

Meta-embers get neutral project_factor (1.0x) — they're never suppressed
by project scoping. This means they surface based purely on semantic
relevance and recency, regardless of which project is active.

This is intentional: cross-project advice is often the most valuable kind.
A debugging lesson learned on project A should surface when debugging
project B.

## The Attention Bell Connection

Meta-embers work especially well with the attention bell pattern. When a
system-reminder arrives and you notice a meta-ember in your loaded context,
that's a moment to actually follow the advice instead of just noting it.

The bell creates the pause. The meta-ember provides the content.

## Lifecycle

Meta-embers should be updated as your practices evolve:

- **When advice becomes outdated:** `ember_contradict` with the new advice
- **When advice becomes obvious:** Consider deleting — it's now internalized
  in your CLAUDE.md or workflow, not needed as a runtime nudge
- **When advice keeps surfacing but you keep ignoring it:** Either the advice
  is wrong (delete it) or you need to listen to it

The shadow-decay system naturally deprioritizes meta-embers that surface
but are never acted on (low utility score). Self-correcting.

## Anti-Patterns

### Too Many Meta-Embers

If more than ~10% of your embers are meta, you're over-instructing. The
graph should be mostly knowledge, with meta-embers as occasional signposts.

### Too Vague

```
BAD:  META: Be careful with this area.
GOOD: META: The payments module has implicit state in the session object
      that isn't obvious from the function signatures. Always check
      session.payment_state before modifying transaction records.
```

### Too Specific

```
BAD:  META: On 2024-03-15, the deploy failed because of X. Remember this.
GOOD: META: Deploys can fail silently when config changes aren't propagated
      to worker nodes. Always verify config hash after deploy.
```

The specific incident belongs in the archive. The generalizable lesson
belongs in a meta-ember.

## Emergent Behavior

Over time, a well-maintained set of meta-embers creates a feedback loop:

1. You encounter a situation
2. Relevant knowledge surfaces, including a meta-ember
3. You follow the meta-advice, producing better results
4. The meta-ember's utility score increases (it was surfaced and acted on)
5. HESTIA ranks it higher in future relevant contexts
6. The advice becomes more reliably available when needed

The graph learns which advice is actually useful and promotes it. Advice
that's ignored (low utility) naturally fades via shadow decay.

This is self-improvement through infrastructure, not willpower.
