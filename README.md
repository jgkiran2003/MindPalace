# MindPalace

MindPalace Phase 1 is a local-first command-line pipeline for searching a
Markdown knowledge vault. It parses YAML frontmatter and Markdown headings,
embeds each chunk with Ollama, and stores the chunks in LanceDB. Search combines
vector similarity with LanceDB native full-text search using two-way reciprocal
rank fusion (RRF), reranks the candidate pool with FlashRank, and applies the
configured penalty to raw notes after reranking.

## Prerequisites

- Python 3.11 or newer. Python's built-in `tomllib` is used for configuration.
- [Ollama](https://ollama.com/) installed and running locally.
- The configured embedding model downloaded:

```bash
ollama pull nomic-embed-text
ollama serve
```

By default, MindPalace connects to Ollama at `http://localhost:11434`. Storage,
model, and retrieval settings live in `config/settings.toml`.

## Quickstart

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install MindPalace in editable mode with its test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Add one or more Markdown files under `data/vault/`. Files may include optional
frontmatter understood by the Phase 1 chunker:

```markdown
---
status: compiled
tags: [projects, search]
severity: low
source_type: document
---

# Search notes

Hybrid retrieval combines semantic and lexical matches.
```

Index every Markdown file in the vault:

```bash
python -m palace.cli index
# or, after installation:
palace index
```

Search the index:

```bash
python -m palace.cli search "hybrid retrieval"
# or:
palace search "hybrid retrieval"
```

The first search initializes the configured FlashRank model and may download its
small model artifact. Search results show the final score, note status, source
path, chunk index, and chunk text.

## Tests

Run the complete test suite from the repository root:

```bash
pytest
```

The Phase 1 tests exercise Markdown/frontmatter parsing, real LanceDB vector and
native FTS retrieval, and post-FlashRank raw-status penalty scoring. They do not
require a running Ollama server or a downloaded reranker model.

## Project layout

```text
config/settings.toml       Runtime configuration
data/vault/                Markdown source-of-truth notes
src/palace/ingestion/      Chunking and indexing pipeline
src/palace/llm/            Ollama embedding client
src/palace/retrieval/      FlashRank reranking and retrieval types
src/palace/stores/         LanceDB storage adapter
tests/                     Pytest suite
```

Generated LanceDB data is written to `data/lance/` and is intentionally excluded
from version control. Vault notes under `data/vault/` are not ignored, so sample
notes can be committed when desired.

