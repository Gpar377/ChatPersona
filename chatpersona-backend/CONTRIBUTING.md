# Contributing

`chatpersona` is a local-only CLI. The repo is safe to share publicly only when it does not contain real chat exports, generated retrieval artifacts, or local profile data.

## Local setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[dev]
```

## Quality checks

Run these before opening a pull request:

```bash
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
python -m build
```

## Data safety rules

- Never commit real WhatsApp exports.
- Never commit generated `reply_examples.jsonl`, Chroma indexes, SQLite files, or anything under `~/.chatpersona/`.
- Keep local exports outside the repo root whenever possible.
- Use only synthetic fixtures under `tests/fixtures/` for tests or examples.

## Preparing a public repo safely

If real chat exports or generated persona artifacts were ever committed in local history, deleting the files in the latest commit is not enough. Rewrite the git history or start from a clean root commit before pushing the repo to GitHub.
