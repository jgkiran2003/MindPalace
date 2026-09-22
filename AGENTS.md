# Codex Directives for MindPalace

- **Role**: Systems Software Engineer implementing MindPalace in Python 3.11+.
- **Ground Truth**: Read `docs/mindpalace-master-specification.md` before writing code.
- **Strict Constraint**: Do NOT edit or rewrite existing contract/schema files:
  - `config/settings.toml`
  - `src/palace/config.py`
  - `src/palace/interfaces.py`
  - `src/palace/domain/models.py`
  - `src/palace/stores/lance_schema.py`
  - `src/palace/stores/schema.sql`
  - `src/palace/retrieval/retrieval_types.py`
- **LanceDB Rule**: Use native Lance FTS (`table.create_fts_index(...)`). Do not use deprecated Tantivy flags.
- **Workflow**: Run pytest after writing code to ensure tests pass before finishing.