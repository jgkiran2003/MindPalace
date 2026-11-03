# Phase 4 baseline — status, lifecycle, and verification

Source of truth for scope: `docs/mindpalace-master-specification.md` §5, Phase 4.
This note records what "baseline-complete" means for Phase 4, how the pieces
behave, and how to verify them. It deliberately does **not** cover extraction
accuracy or prompt quality.

## Spec requirements vs. state

| Requirement (spec §5, Phase 4) | State |
|---|---|
| Fast/reasoning tier routing (`llm/client.py`) | Complete |
| Validated `ClaimProposal` extraction (`claims/extractor.py`) | Complete (accuracy deferred) |
| Resolver wiring extraction → `propose_claim` (`claims/resolver.py`) | Complete |
| `chat` command with dry-run showing proposed claims | Complete |
| Compilation with `derived_from` linking | Complete |
| Compiled note is indexed (spec "done when") | Complete |
| Raw chunks archived on compile (spec "done when") | Complete — see "Archival" |
| Raw staging / voice-audio retention (`ingestion/capture.py`, `normalizer.py`) | **Deferred**: no capture surface exists yet (voice and UI are Phase 5) |
| Claims derived from compiled content | **Deferred**: not required by the spec text; the extractor is dialogue-specific and compiled text is LLM-generated |

## Claim pipeline

`chat` → `_process_claim_exchange` → `extract_claim_proposals` →
`resolve_and_store_proposals` → `SQLiteClaimsStore.propose_claim`
(supersede / conflict logic unchanged).

Every exchange ends in exactly one `ClaimOutcome` (`claims/outcomes.py`):

| Outcome | Meaning | Database touched? |
|---|---|---|
| `success` | Proposals resolved (or previewed under `--dry-run`) | Yes, unless dry-run |
| `no_claims` | Model returned a valid empty list | No |
| `malformed_output` | Model answered, output failed JSON/schema validation | No |
| `llm_failure` | Ollama unreachable/timeout/HTTP error, **or the embedding call failed** | No |
| `storage_failure` | `propose_claim` raised | Possibly partially — see below |
| `unexpected_error` | Anything else; never swallowed | No |

* Embeddings for all proposals are computed **before** any write, so an
  embedding failure writes nothing.
* Each `propose_claim` is its own transaction. If proposal *k* fails, proposals
  before it stay committed and are listed in the report
  (`ClaimStorageError.persisted`). Atomic multi-claim writes are a possible
  future change to the store API.
* In normal chat a finished background task prints one `[claims] …` line
  (stdout on success, stderr on failure) before the next prompt, or at
  shutdown. `no_claims` is silent. Under `--dry-run` the reply is followed by
  the proposals table, plus the error if extraction failed.
* `--dry-run` writes **no claim rows** (verified: 0 rows in `claims` and
  `claim_history`). Retrieval may still create an empty `claims.db` schema on
  first use, as `palace search` does.

## Chat lifecycle

* History is **in memory for the session only**; there is no persistent chat
  store in Phase 4. Durable knowledge goes through claims.
* Each turn: retrieve (`HybridSearcher`) → system prompt with context + full
  history → stream the reply (reasoning tier). Extraction runs concurrently as
  a background task over the last exchange plus the new message; greetings skip it.
* A retrieval or streaming error is printed (`[error] could not answer: …`),
  the turn is dropped from history, and the REPL continues.
* On `quit`/`exit`/Ctrl-C the REPL waits for in-flight extractions, reports
  them, then closes the HTTP client.
* Known limit: history is unbounded, so very long sessions will eventually
  exceed the model's context window.

## Compile lifecycle

`palace compile` (`compilation/compiler.py`):

1. Select notes with `status: raw` and not `processed: true`; fewer than 3 → stop.
2. Reasoning-tier LLM drafts the compiled body. **Empty output aborts here**
   with nothing written.
3. Write `compiled-<uuid5 of source paths>.md` atomically
   (`status: compiled`, `generated: true`, tag `generated`, `derived_from`
   titles, `derived_from_paths` vault-relative paths). The same sources always
   produce the same filename, so retries overwrite rather than duplicate.
4. Index it. On failure the file is rolled back (removed, or restored if it
   pre-existed) and **no source is touched**; rerun the command.
5. Only then archive each source: add `processed: true` and
   `compiled_into: <file>` to its frontmatter and drop its chunks from the index.

### Archival

`ChunkStatus` only has `raw` and `compiled` and `domain/models.py` /
`lance_schema.py` are protected, so "archived" is **vault-driven**: the raw
note keeps `status: raw` and gains `processed: true`. The indexer skips such
notes (and deletes any stale chunks), so a full `palace index` stays consistent.
The original note is never deleted and its body is byte-for-byte unchanged; to
restore it, delete the two marker lines and re-index.

If archiving fails midway the command exits non-zero and says which sources
failed. The vault is the source of truth, so `palace index` reconciles the index.

### Provenance limits

The compiled note links to its sources through frontmatter. The single-valued
`derived_from_chunk_id` chunk field cannot represent many sources, so it is not
set on compiled chunks.

## Configuration

* `.env.example` documents the only three variables the code reads:
  `PALACE_LLM_FAST_MODEL`, `PALACE_LLM_REASONING_MODEL`,
  `PALACE_OLLAMA_KEEP_ALIVE`. Nothing auto-loads a `.env` file.
* The spec and `config/settings.toml` agree on the tiers (`qwen3:4b` fast,
  `qwen3:14b` reasoning). For a machine that cannot run the 14B model, set
  `PALACE_LLM_REASONING_MODEL` instead of editing the locked settings file.

## Verification

Unit tests (no Ollama needed; `tests/conftest.py` forbids a default
`VaultIndexer`, so tests can never write to your real `data/lance`):

```bash
pytest
```

Real-Ollama smoke test, in an isolated scratch workspace (your `data/` is not
touched):

```bash
ollama serve   # with the models from config/settings.toml pulled
bash scripts/smoke_phase4.sh
```

Expected: raw notes indexed → `compile` creates and indexes a compiled note →
re-index skips 3 archived notes → search returns the compiled chunk and no raw
ones → second `compile` is a no-op → `chat --dry-run` prints a proposals table
and leaves 0 claim rows → `chat` saves claims. LLM wording varies; those state
transitions must not.
