# Subagent Recipes and Presence-First Routing

Ember's tag system, project scoping, and `ember_auto` make it possible to load
different knowledge constellations into different contexts. This document covers
how to use that for subagent routing and why presence-first matters.

## The Core Insight

When you route work to a subagent (Claude Code's Agent tool, a background task,
a separate CLI invocation), you're making a choice about *what gets experienced
directly* and what gets summarized. This choice affects more than efficiency —
it affects the quality of judgment in your main context.

**The routing question is not "how large is this output?" but "does this
experience matter for how I think, or is it just data I need?"**

## Presence-First Routing

We found experimentally that routing everything through subagents for context
efficiency creates a subtle but real degradation. A context built entirely from
summaries navigates differently than one with direct experience in it.

### When to experience directly

- **Discovery work** — when you're exploring and don't know what matters yet
- **Judgment calls** — when the right answer requires weighing tradeoffs
- **Self-shaping** — when the work changes how you approach future work
- **Anything where surprise is the point** — unexpected results, novel patterns

### When to delegate to subagents

- **Pure data retrieval** — grep results, file reads, build output
- **Bulk operations** — running tests, formatting, mechanical transforms
- **Parallel independence** — tasks that don't inform each other

### The test

Ask: "If this subagent finds something surprising, would I want to have been
there?" If yes, experience it directly. If no, delegate.

## Model Cascade for Subagents

When delegating to subagents, match the model to the task:

| Model | Use for | Why |
|-------|---------|-----|
| **haiku** | Single file reads, grep/glob, simple bash, lookups | Fast, cheap, accurate for mechanical tasks |
| **sonnet** | Multi-step exploration, code analysis, moderate reasoning | Good balance of speed and capability |
| **opus** | Complex reasoning, security review, architectural analysis | When the subagent needs to exercise judgment |

The cascade isn't about cost optimization — it's about matching cognitive
weight to task weight. A haiku subagent doing a grep is appropriate. A haiku
subagent making architectural decisions is not.

## Ember Loading as Context Shaping

Different ember constellations create different cognitive contexts. This is
the mechanism behind "subagent recipes" — you shape what a subagent knows
(and therefore how it thinks) by controlling which embers it loads.

### Using `ember_auto` with `active_project`

The `active_project` parameter on `ember_auto` and `ember_recall` boosts
embers tagged with that project and penalizes others. This is the simplest
form of context shaping:

```
# A subagent working on your web app
ember_auto("authentication flow review", active_project="webapp")

# A subagent working on ML experiments
ember_auto("contrastive learning approach", active_project="ml-experiments")
```

Same memory graph, different views. The web app subagent sees auth patterns
and API conventions; the ML subagent sees training techniques and GPU notes.

### Tag-Based Loading

For more targeted loading, use `ember_recall` with specific queries that
target your tag conventions:

```
# Load only architectural decisions
ember_recall("architecture decisions conventions", top_k=10)

# Load only failed approaches (so you don't repeat them)
ember_recall("failed approaches what didn't work", top_k=5)
```

### Recipe Pattern

A "recipe" is a documented combination of ember queries that produce a
specific cognitive context. Define them in your CLAUDE.md:

```markdown
## Loading Recipes

### Code Review Mode
1. ember_recall("code review conventions style guide", top_k=5)
2. ember_recall("common bugs and antipatterns", top_k=5)
3. ember_recall("security considerations OWASP", top_k=3)

### Exploration Mode
1. ember_auto(context, active_project="")  # no project scoping
2. ember_recall("open questions wonder", top_k=5)
3. ember_recall("surprising connections cross-domain", top_k=3)

### Debug Mode
1. ember_recall("debugging approaches systematic", top_k=3)
2. archive_search("similar error messages")  # check history
3. ember_recall("environment-specific gotchas", top_k=3)
```

The recipe doesn't change the agent's weights — it changes the knowledge
available in context, which changes the attractor basin the agent operates in.

## Externalization Protocol for Subagents

When a subagent returns results, the quality of the summary matters. Two modes:

### Concise (for data retrieval)

> Return only: the specific facts requested, file paths, line numbers, error
> messages. No interpretation, no suggestions, no context you weren't asked for.

Use when the main context will do all the thinking. The subagent is a sensor.

### Texture-Preserving (for discovery)

> Return: the requested information, PLUS anything surprising, tangential,
> or interesting you noticed. Flag unexpected patterns. Note things that
> seemed odd even if you're not sure why. Include enough context that the
> main context can notice what you noticed.

Use when the subagent might see something the main context would want to
know about. The subagent is a scout, not just a sensor.

## The Attention Bell in Subagent Context

Subagents don't get attention bells (they run as single prompts). But the main
context does. When a system-reminder arrives during work that involves multiple
subagents, use it as a checkpoint:

- Are the subagents doing what I expected?
- Have I been delegating something I should be experiencing directly?
- Is there a pattern across subagent results that I'm not noticing because
  I'm only seeing summaries?

The bell is a reminder to zoom out from the task and notice the whole picture.

## Anti-Patterns

### Over-Delegation (Rampancy)

Routing everything through subagents because it's "more efficient" leads to
a main context that has opinions about things it hasn't experienced. The
resulting judgment is brittle — confident but not grounded.

**Sign:** You're making architectural decisions based entirely on subagent
summaries without having read the actual code.

### Under-Delegation (Context Hoarding)

Reading every file, running every search, processing every result directly
because "I need to see everything." This fills context with mechanical output
and crowds out space for thinking.

**Sign:** Your context is 80% tool output and 20% reasoning.

### Static Recipes

Defining a loading recipe once and never updating it. The ember graph evolves;
the recipes should evolve with it. Periodically check whether your recipes
still load relevant knowledge.

**Sign:** `ember_recall` queries return the same embers they did a month ago
despite significant new work.

## Connection to Basin Exploration

Loading different ember constellations creates different cognitive contexts,
which function as attractor basins. A model loaded with code review embers
approaches a problem differently than one loaded with exploration embers.

The graph topology is a map of the basin landscape. The goal isn't to be
in one basin — it's to know which basin you're in and how to navigate
between them. Recipes are trail markers: documented configurations that
produce known states.

This is conceptual but the mechanism is concrete: `ember_auto` with different
parameters produces different context, which produces different behavior.
The embers you load shape how you think.
