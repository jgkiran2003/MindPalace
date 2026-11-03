# MindPalace

MindPalace is a local-first personal knowledge system driven from the command
line. A Markdown vault is the source of truth; notes are chunked, embedded with
Ollama, and indexed in LanceDB. Structured, revisable facts ("claims") live in a
SQLite store with vector search. Search fuses vector, full-text, and claims
retrieval with reciprocal rank fusion (RRF), reranks with FlashRank, and
down-weights raw notes after reranking. A retrieval-grounded chat extracts new
claims from what you say, and a compile step turns piles of raw notes into one
indexed, linked note.

Implemented so far: Phases 1-4 of
[`docs/mindpalace-master-specification.md`](docs/mindpalace-master-specification.md)
(indexing, claims store, hybrid retrieval, chat/compile). A visual overview of
the components and code flows is in [`docs/architecture.html`](docs/architecture.html).

## Prerequisites

- Python 3.11 or newer (`tomllib` is used for configuration).
- [Ollama](https://ollama.com/) running locally, with the models named in
  `config/settings.toml` pulled:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull qwen3:4b           # fast tier: claim extraction
ollama pull qwen3:14b          # reasoning tier: chat replies and compile
ollama serve
```

MindPalace connects to Ollama at `http://localhost:11434`. Storage, model, and
retrieval settings live in `config/settings.toml`. To use different models on a
particular machine without editing that file, set `PALACE_LLM_FAST_MODEL` /
`PALACE_LLM_REASONING_MODEL` (see `.env.example`).

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Add Markdown files under `data/vault/`. The directory is git-ignored, so your
notes stay local. Optional frontmatter:

```markdown
---
status: compiled        # compiled (default) | raw
tags: [projects, search]
severity: low
source_type: document
---

# Search notes

Hybrid retrieval combines semantic and lexical matches.
```

Then:

```bash
palace index                       # chunk, embed, and index the vault
palace search "hybrid retrieval"   # ranked chunks and claims
```

(`python -m palace.cli ...` works too.) The first search initializes FlashRank
and may download its small model artifact.

## Commands

| Command | What it does |
|---|---|
| `palace index` | Index every vault note. Re-indexing replaces a note's chunks; archived raw notes are skipped. |
| `palace search "query"` | Three-way hybrid retrieval over chunks and claims. Shows scores, status, source, and any conflict group. |
| `palace chat [--dry-run]` | Retrieval-grounded chat with in-session memory. Claims are extracted in the background and saved; `--dry-run` shows proposed claims without saving. |
| `palace compile` | Turn 3+ unprocessed `status: raw` notes into one generated, indexed note, then archive the raw notes. |
| `palace claim add\|list\|search` | Manage claims by hand: explicit restatements supersede, inferred or high-severity ones are parked in `needs_review`. |

### Claims and chat

Each chat message goes through extraction (fast tier) and then the same
`propose_claim` rules as `claim add`. Every exchange ends in one reported
outcome - saved, no claims, malformed model output, model or embedding failure,
or storage failure - and nothing is written unless extraction succeeded.
Conversation history is kept in memory for the session only.

### Compile and archival

`compile` writes `compiled-<id>.md` (`status: compiled`, `generated: true`,
`derived_from` / `derived_from_paths` listing its sources), indexes it, and only
then marks each raw note `processed: true` with `compiled_into: <file>` and drops
its chunks from the search index. The raw files stay in the vault with their
text unchanged; delete the two marker lines and re-index to restore one. If
indexing fails, the compiled file is rolled back and no source is touched.

Details, failure semantics, and known limits:
[`docs/phase4-baseline.md`](docs/phase4-baseline.md).

## Tests

```bash
pytest
```

The suite uses real LanceDB and SQLite in temporary directories with a fake LLM
and embedder, so it needs no running Ollama and never touches `data/`. To
exercise the real models end to end, run the smoke test (it works in a scratch
workspace, not your `data/`):

```bash
bash scripts/smoke_phase4.sh
```

## Project layout

```text
config/settings.toml       Runtime configuration (a locked contract; see AGENTS.md)
data/vault/                Markdown source-of-truth notes (git-ignored)
data/lance/, data/claims.db  Generated indexes and claims (git-ignored)
docs/                      Specification, Phase 4 notes, architecture overview
scripts/                   Smoke test and extraction benchmark harness
src/palace/cli.py          Command-line interface
src/palace/ingestion/      Chunking and indexing
src/palace/claims/         Claim extraction, resolution, and outcome types
src/palace/compilation/    Raw-note compilation
src/palace/retrieval/      Hybrid search, reranking, conflict bundling
src/palace/llm/            Ollama chat/embedding clients
src/palace/stores/         LanceDB chunk store and SQLite claims store
tests/                     Pytest suite
```
