# MindPalace — Master Implementation Specification (v1)

*This document is the implementation contract. It assumes the Greenfield Architecture Blueprint (v2) as context and exists to remove every remaining ambiguity before code generation begins. Every schema, config key, and interface referenced below is a real file in this delivery — nothing here is illustrative.*

All source files referenced in this spec are delivered alongside it:

```
config/settings.toml
src/palace/config.py
src/palace/domain/models.py
src/palace/stores/schema.sql
src/palace/stores/lance_schema.py
src/palace/retrieval/retrieval_types.py
src/palace/interfaces.py
```

Every file above has been executed (not just syntax-checked) against a real SQLite connection and a real Pydantic/PyArrow install as part of producing this spec: settings load correctly, `DocumentChunk`'s vector-dimension validator actually rejects a malformed vector, the SQLite schema's generated `claim_key` column and a full supersession insert sequence run correctly, and all four `Protocol` interfaces import without circular dependencies.

---

## 1. Configuration Specification

See `config/settings.toml` (full file delivered) and its typed loader `src/palace/config.py`. Every section has concrete defaults — nothing is left for an implementer to invent:

| Section | Key | Default | Used by |
|---|---|---|---|
| `paths` | `vault_dir` | `data/vault` | `ingestion/*`, source-of-truth markdown |
| `paths` | `lance_dir` | `data/lance` | `stores/file_store.py` |
| `paths` | `claims_db_path` | `data/claims.db` | `stores/claims_store.py` |
| `paths` | `schema_sql_path` | `src/palace/stores/schema.sql` | `stores/claims_store.py._init_db()` |
| `paths` | `temp_audio_dir` | `data/tmp_audio` | `ingestion/capture.py` |
| `embedding` | `model` | `nomic-embed-text` | `llm/embedder.py` |
| `embedding` | `dimension` | `768` | `domain/models.py`, `stores/lance_schema.py`, `stores/schema.sql` (`vec_claims`) |
| `embedding` | `ollama_host` | `http://localhost:11434` | `llm/embedder.py` |
| `embedding` | `batch_size` | `16` | `ingestion/indexer.py` |
| `reranker` | `model` | `ms-marco-TinyBERT-L-2-v2` | `retrieval/reranker.py` |
| `reranker` | `max_length` | `512` | `retrieval/reranker.py` |
| `reranker` | `device` | `cpu` | `retrieval/reranker.py` |
| `retrieval` | `top_k` | `40` | RRF candidate pool depth |
| `retrieval` | `retrieval_top_k` | `8` | final results returned to the caller |
| `retrieval` | `claims_search_k` | `20` | `k` passed to `ClaimsStoreAdapter.search_claims` |
| `retrieval` | `rrf_k` | `60` | RRF damping constant |
| `retrieval` | `raw_status_penalty` | `0.4` | post-FlashRank multiplier for `status == "raw"` |
| `llm` | `fast_tier_model` | `qwen3:4b` | `claims/extraction.py`, routing |
| `llm` | `reasoning_tier_model` | `qwen3:14b` | Jarvis conversation |
| `claims` | `high_severity_blocks_auto_resolve` | `true` | `ClaimsStoreAdapter.propose_claim` |
| `claims` | `compaction_age_days` | `180` | `compaction/compactor.py` |
| `claims` | `compaction_min_count` | `5` | `compaction/compactor.py` |
| `indexing` | `optimize_after_writes` | `25` | `ingestion/indexer.py` |

**Rule for implementers:** if a value is needed and it isn't in this table, it's a bug in this spec — flag it rather than inventing a default inline.

---

## 2. Exact Data Contracts & Schemas

- **LanceDB chunk table** — `src/palace/stores/lance_schema.py`, `build_chunk_schema()`. Thirteen fields, exact types, nullability specified per field. Vector width is read from `settings.embedding.dimension` at call time — never hardcoded twice.
- **SQLite claims store DDL** — `src/palace/stores/schema.sql`. Five tables (`vaults`, `claims`, `claim_tags`, `claim_references`, `claim_history`) plus the `vec_claims` `vec0` virtual table. `claims.claim_key` is a `GENERATED ALWAYS AS ... STORED` column — application code must never write to it directly. `supersedes_id`/`superseded_by_id`/`vault_id` use `ON DELETE SET NULL`; `claim_tags`/`claim_references` use `ON DELETE CASCADE`; `claim_history` is intentionally **not** foreign-keyed to `claims.id`, because history must survive a hard-deleted claim.
- **Core domain models** — `src/palace/domain/models.py`. `Claim`, `ClaimProposal`, `ConflictGroup`, `DocumentChunk`, plus every enum (`ClaimStatus`, `ChunkStatus`, `Severity`, `SourceType`, `ChangeReason`) and `Vault`/`ClaimHistoryEntry`. `ConflictGroup` has a `model_validator` that rejects construction if any member claim isn't `needs_review` or doesn't share the group's `claim_key` — this is enforced, not just documented. `DocumentChunk.vector`'s length is validated against `settings.embedding.dimension` at construction time.

---

## 3. Mathematical & Algorithmic Definitions

### 3.1 — Three-way RRF candidate fusion

For a candidate `d` (a chunk or a claim), let `L(d) ⊆ {vector, fts, claims}` be the set of retrieval legs in which `d` appears, and let `rank_ℓ(d) ∈ {1, 2, 3, …}` be `d`'s **1-indexed** position within leg `ℓ` (the top hit in a leg has rank 1). Then:

```
RRF(d) = Σ_{ℓ ∈ L(d)}  1 / (k + rank_ℓ(d))
```

where `k = settings.retrieval.rrf_k` (default `60`). A candidate absent from a leg contributes `0` from that leg — it is **not** assigned a penalty rank. The `claims` leg here is itself the *output* of `ClaimsStoreAdapter.search_claims()`, which internally fuses its own vector and text sub-legs with the same formula and the same `k` before being treated as a single ranked list at this outer stage. The top `settings.retrieval.top_k` candidates by `RRF(d)` proceed to reranking.

### 3.2 — Post-FlashRank score blending

FlashRank scores every candidate in the `top_k` pool against the query **unweighted by status** — `flashrank_score(d) = reranker.score([(query, d.text)])`. The status penalty is applied strictly after:

```
status_weight(d) = raw_status_penalty   if d.kind == "chunk" AND d.status == "raw"
                  = 1.0                  otherwise

final_score(d) = flashrank_score(d) × status_weight(d)
```

`raw_status_penalty = settings.retrieval.raw_status_penalty` (default `0.4`). `needs_review` claims receive **no** score penalty — their visibility is instead guaranteed by conflict bundling (§3.3), per the locked design principle that ambiguity should surface, not be silently ranked away. Final ordering is `final_score(d)` descending; the top `settings.retrieval.retrieval_top_k` candidates are returned.

### 3.3 — Conflict bundling

For every `Candidate` in the final ranked list where `kind == "claim"` and `payload.status == ClaimStatus.NEEDS_REVIEW`, call `ClaimsStoreAdapter.get_conflict_group(payload.claim_key)` and attach the result to `Candidate.conflict_group`. This is a presentation-layer join, not a scoring operation — it never changes `final_score` or ordering.

---

## 4. Component Interface Signatures

Fully specified in `src/palace/interfaces.py` as `typing.Protocol` classes (structural typing — a concrete class satisfies the interface by matching signatures, no explicit inheritance required). Summary:

```python
class FileStore(Protocol):
    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None: ...
    def delete_chunks_by_doc_path(self, doc_path: str) -> int: ...
    def vector_search(self, query_vector: list[float], limit: int) -> list[DocumentChunk]: ...
    def fts_search(self, query_text: str, limit: int) -> list[DocumentChunk]: ...
    def optimize(self) -> None: ...

class ClaimsStoreAdapter(Protocol):
    def get_active_claim(self, entity: str, attribute: str) -> Optional[Claim]: ...
    def search_claims(self, query_embedding: list[float], query_text: str, k: int,
                       status_filter: Optional[list[ClaimStatus]] = None) -> list[Claim]: ...
    def propose_claim(self, proposal: ClaimProposal, embedding: list[float]) -> Claim: ...
    def get_conflict_group(self, claim_key: str) -> Optional[ConflictGroup]: ...

class FlashRankReranker(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...

class HybridSearcher(Protocol):
    async def retrieve(self, query: str) -> list[Candidate]: ...
```

Note one deliberate refinement from the earlier working draft: `get_conflict_group` now returns `Optional[ConflictGroup]` (a validated, single object) rather than a bare `list[Claim]` — this is the version to implement against.

---

## 5. Sequential 5-Phase Implementation Roadmap

**Phase 1 — Barebones CLI: ingest, index, hybrid search, rerank, print.**
Build `config.py`, `lance_schema.py`, `stores/file_store.py` (implements `FileStore`), a minimal `ingestion/chunker.py` (split by heading/paragraph boundary), `ingestion/indexer.py` (walk `data/vault/*.md`, embed via `nomic-embed-text`, upsert, call `optimize()` every `optimize_after_writes` chunks), and `retrieval/reranker.py` (implements `FlashRankReranker` via the `rerankers` library). Add a `cli.py` with two commands: `index` and `search "<query>"`, the latter doing vector + FTS search, a **2-way** RRF fusion (no claims store yet), FlashRank rerank, and printing the top `retrieval_top_k` results with scores. *Done when:* `python -m palace.cli index && python -m palace.cli search "some query"` prints ranked, relevant results from your own vault notes.

**Phase 2 — Claims store foundation, manual claim management.**
Build the rest of `domain/models.py` usage, `stores/schema.sql` application, and `stores/claims_store.py` (implements `ClaimsStoreAdapter`, including `propose_claim`/`_supersede_claim`/`_flag_conflict`). Extend the CLI with `claim add`, `claim list`, `claim search "<query>"` for manual testing — no LLM involved yet. *Done when:* you can manually create two claims with the same `entity:attribute` and `is_explicit=True`, see the first flip to `superseded`, and manually create two conflicting `is_explicit=False` claims and see both flip to `needs_review` with `get_conflict_group` returning both.

**Phase 3 — Full 3-way RRF + HybridSearcher + status weighting + conflict bundling.**
Build `retrieval/retrieval_types.py`, `retrieval/hybrid_searcher.py` (implements `HybridSearcher`, wiring `FileStore` + `ClaimsStoreAdapter` + `FlashRankReranker` per §3.1–§3.3 exactly). Replace the CLI's Phase-1 2-way search with a call to `HybridSearcher.retrieve()`. *Done when:* a single `search` command blends chunks and claims in one ranked list, a chunk you've manually tagged `status: raw` visibly ranks below an equivalent `compiled` one, and a deliberately created conflict pair both appear in results with `conflict_group` populated.

**Phase 4 — LLM-driven extraction and the raw→compiled ingestion lifecycle.**
Build `llm/ollama_client.py` (fast/reasoning tier routing), `claims/extraction.py` (Instructor-validated `ClaimProposal` generation from chat turns/chunks), `claims/resolver.py` (wires extraction to `propose_claim`), and the real `ingestion/capture.py`/`normalizer.py` (raw staging, voice-audio retention, compilation with `derived_from` linking and raw-chunk archival on compile). Add a `chat` CLI command with a dry-run mode showing proposed claims before committing, for prompt tuning. *Done when:* pasting a first-person statement through `chat` produces a sensible `ClaimProposal`, and dropping a raw-shaped note into the vault and "compiling" it produces a linked, indexed compiled chunk with the raw one archived.

**Phase 5 — Autonomy layer, vaults, compaction, and the real UI surfaces.**
Build `tools/registry.py`/`skill_router.py`/`executor.py` (tiered autonomy, batch permission requests, graceful scope-drift handling), `stores/vault_store.py` (access gating), `compaction/compactor.py` (the scheduled rollup job per `settings.claims.compaction_age_days`/`compaction_min_count`), and `interface/chat_api.py`/`messaging_bridge.py`/`nudges.py`, replacing the CLI as the daily-use surface. *Done when:* the full ADR v1 feature set runs end-to-end — tiered tool autonomy with vault-gated access, proactive nudges, and the chat UI as the primary surface.

Each phase's CLI/API stays functional at the end of that phase — this is meant to be built and tested incrementally, not implemented all at once and debugged from scratch.
