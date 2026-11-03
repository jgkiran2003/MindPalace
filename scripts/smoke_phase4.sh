#!/usr/bin/env bash
# Phase 4 end-to-end smoke test against a REAL local Ollama.
#
# Runs in a throwaway workspace whose config points every path at the scratch
# directory, so your data/vault, data/lance and data/claims.db are never touched.
# Needs: `ollama serve` running, the models named in config/settings.toml pulled
# (or PALACE_LLM_* overrides), and `pip install -e ".[dev]"` done in .venv.
# Output is inspected by eye: LLM answers vary, the lifecycle must not.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$REPO/.venv/bin/palace"
PY="$REPO/.venv/bin/python"
S="$(mktemp -d /tmp/palace_smoke.XXXXXX)"
mkdir -p "$S/config" "$S/data/vault"
sed -E \
  -e "s#^(vault_dir\s*=\s*)\"[^\"]*\"#\1\"$S/data/vault\"#" \
  -e "s#^(lance_dir\s*=\s*)\"[^\"]*\"#\1\"$S/data/lance\"#" \
  -e "s#^(claims_db_path\s*=\s*)\"[^\"]*\"#\1\"$S/data/claims.db\"#" \
  -e "s#^(schema_sql_path\s*=\s*)\"[^\"]*\"#\1\"$REPO/src/palace/stores/schema.sql\"#" \
  -e "s#^(temp_audio_dir\s*=\s*)\"[^\"]*\"#\1\"$S/data/tmp_audio\"#" \
  "$REPO/config/settings.toml" > "$S/config/settings.toml"
[ -d "$REPO/.flashrank_cache" ] && ln -s "$REPO/.flashrank_cache" "$S/.flashrank_cache"
cd "$S"
echo "workspace: $S"

claims_rows() { "$PY" -c "
import os, sqlite3
if not os.path.exists('data/claims.db'): print('no claims.db')
else: print(sqlite3.connect('data/claims.db').execute('select count(*) from claims').fetchone()[0], 'claim rows')"; }

for i in 1 2 3; do
  food=(leaves grass bark)
  printf -- '---\ntitle: Quokka log %s\nstatus: raw\n---\n\n# Quokka log %s\n\nObservation %s: the quokka colony at the harbour eats %s.\n' \
    "$i" "$i" "$i" "${food[$((i-1))]}" > "data/vault/raw-$i.md"
done

echo; echo "== 1. index raw notes";                 "$P" index
echo; echo "== 2. search (expect 3 raw chunks)";    "$P" search "quokka colony diet" | grep -c 'status=raw' | sed 's/^/raw hits: /'
echo; echo "== 3. compile (expect: created + indexed)"; "$P" compile
echo; echo "== 4. re-index (expect: 3 archived skipped)"; "$P" index
echo; echo "== 5. search (expect: compiled hit, no raw hits)"
"$P" search "quokka colony diet" | grep -E 'status=' | sed 's/source=.*\//source=…\//'
echo; echo "== 6. sources preserved + marked (expect processed/compiled_into)"; head -6 data/vault/raw-1.md
echo; echo "== 7. compile again (expect: nothing to do)"; "$P" compile

echo; echo "== 8. chat --dry-run (expect proposals table, 0 claim rows)"
printf 'My name is Bob and I work at MyStorage.\nquit\n' | "$P" chat --dry-run 2>&1 | tail -9
claims_rows

echo; echo "== 9. chat (expect '[claims] saved ...')"
printf 'My name is Bob and I work at MyStorage.\nquit\n' | "$P" chat 2>&1 | tail -5
claims_rows
echo; "$P" claim list
echo; echo "scratch workspace kept at: $S (safe to delete)"
