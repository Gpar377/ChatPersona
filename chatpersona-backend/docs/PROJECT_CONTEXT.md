# Project Context

Last reviewed: 2026-03-22

## Product Snapshot

`chatpersona` is a local-only Python CLI that:

- parses WhatsApp 1:1 `.txt` exports
- extracts reply-conditioned examples for a chosen persona
- stores local profile metadata and retrieval artifacts under `~/.chatpersona/`
- runs persona chat against local Ollama models only

The product goal is privacy-first local experimentation, not hosted chat, sync, or multi-user collaboration.

## Core Flows

1. `chatpersona doctor`
   Checks Python, Ollama reachability, model availability, and profile storage.
2. `chatpersona init`
   Guides the user through export selection, persona selection, model choice, and artifact creation.
3. `chatpersona chat "<profile>"`
   Loads a saved profile, prepares assets, and opens the interactive chat workspace.
4. `chatpersona rebuild "<profile>"`
   Recomputes artifacts when the export changes or the parser evolves.

## Repo Map

- `chatpersona/cli.py`
  CLI commands, onboarding flow, doctor checks, dashboard rendering.
- `chatpersona/corpus.py`
  WhatsApp parsing, turn construction, reply example generation, lexical ranking, embedding index creation.
- `chatpersona/profiles.py`
  Profile persistence, rebuild detection, and artifact build orchestration.
- `chatpersona/runtime.py`
  Interactive chat loop, planner/style prompting, reply cleanup, retry/fallback behavior.
- `chatpersona/ollama.py`
  Local model discovery via Ollama API and CLI.
- `chatpersona/ui.py`
  Shared Rich UI helpers.
- `tests/`
  Unit/integration coverage using synthetic fixtures only.

## Validation Notes

Reviewed locally with the project `venv`:

- `./venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q`
- `./venv/bin/python -m ruff check .`
- `./venv/bin/python -m ruff format --check .`
- `./venv/bin/python -m build --no-isolation`

Notes:

- plain `python` was not on PATH in this environment
- isolated `python -m build` could not fetch build requirements because network access was unavailable
- the no-isolation build succeeded

## Current Design Choices

- Preferred retrieval mode and last built retrieval mode are tracked separately.
- Embedding retrieval can fall back to lexical retrieval without blocking profile creation.
- The runtime uses a two-stage generation flow:
  planner prompt for semantic intent, then style prompt for texting shape.
- The product is intentionally macOS-first and local-only.

## Known Maintenance Opportunities

- `chat` still rebuilds parsing assets on startup even when source content is unchanged, which may feel slow on large exports.
- Tests currently rely on the full runtime dependency set; parser/profile-only smoke tests could be split into a lighter dependency lane.
- Retrieval preference is inferred from model availability rather than chosen explicitly by the user.

## Suggested Working Norms

- Keep real chat exports and generated artifacts out of the repo.
- Prefer synthetic fixtures under `tests/fixtures/` for all new tests.
- Treat UI copy and error guidance as product surface, not just implementation detail.
- When changing retrieval behavior, update both profile persistence and rebuild-status logic together.
