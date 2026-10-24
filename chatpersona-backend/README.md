# chatpersona

`chatpersona` is a local-only CLI for building a conversational persona from your own exported WhatsApp 1:1 chat history and running it against local Ollama models.

It is designed for private local experimentation. Chat exports, profile metadata, and retrieval artifacts stay on disk on your machine. This project is not a hosted service and does not send your chats to a remote API.

## Supported Scope

- WhatsApp 1:1 `.txt` exports only
- Ollama for all model inference
- macOS-first local setup
- Local profile storage under `~/.chatpersona/`

This is not intended for group chats, cloud sync, or hosted inference.

## Privacy And Responsible Use

Use `chatpersona` only with chat exports you are allowed to use. The tool is best treated as a private local archive/persona experiment built from your own data. If you are sharing the repo publicly, do not commit real chat exports, generated profile artifacts, or local profile storage.

## Prerequisites

Before you start, make sure you have:

- Python 3.11 or newer
- Ollama installed locally and running
- at least one local chat model in Ollama
- an exported WhatsApp `.txt` file from a direct 1:1 conversation

Recommended Ollama models:

```bash
ollama pull llama3.2:latest
ollama pull qwen3-embedding:0.6b
```

## Local Installation

The supported setup flow for a clean macOS environment is:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

After installation, both of these entrypoints are supported:

```bash
chatpersona --help
chatpersona --version
python -m chatpersona --help
```

`chatpersona` is the preferred public-facing command once the package is installed.

## Quickstart

1. Check your local setup:

```bash
chatpersona doctor
```

2. Start the onboarding wizard:

```bash
chatpersona init
```

3. See saved profiles:

```bash
chatpersona profiles list
```

4. Start chatting:

```bash
chatpersona chat "Sam - sample_chat"
```

5. Rebuild a profile after the export changes:

```bash
chatpersona rebuild "Sam - sample_chat"
```

## First-Time Setup Example

```text
$ chatpersona doctor
PASS  Python version     Using Python 3.13.2, which meets the >=3.11 requirement.
PASS  Ollama reachable   Ollama responded at http://127.0.0.1:11434.
PASS  Local models       Found 2 local Ollama model(s).
PASS  Chat model         Usable chat models were found. Default choice: `llama3.2:latest`.
PASS  Embedding model    Embedding retrieval can use `qwen3-embedding:0.6b`.
PASS  Profile storage    Local profile storage is writable at `~/.chatpersona/profiles`.

$ chatpersona init
chatpersona setup
1. find the exported WhatsApp `.txt` file
2. choose which person to emulate
3. pick the local Ollama chat model
4. build the local retrieval artifacts

Choose the exported WhatsApp .txt file: ./exports/sample_chat.txt
Choose which participant to emulate: Sam
Profile name: Sam - sample_chat
Choose the chat model: llama3.2:latest

Building profile artifacts ...
Profile Ready
Profile: Sam - sample_chat
Persona: Sam
Partner: Alex
Chat model: llama3.2:latest
Preferred retrieval mode: embedding
Last build retrieval mode: embedding

What happens next:
- `chatpersona chat "Sam - sample_chat"`
- `chatpersona rebuild "Sam - sample_chat"` after the export changes
- `chatpersona doctor` if Ollama or model checks fail
```

## Core Commands

```bash
chatpersona init
chatpersona onboard
chatpersona chat "<profile>"
chatpersona rebuild "<profile>"
chatpersona profiles list
chatpersona models list
chatpersona doctor
chatpersona --version
```

## Where Data Lives

`chatpersona` stores local profile data under:

```text
~/.chatpersona/
```

Each profile gets its own directory containing:

- `profile.json` for profile metadata
- `artifacts/reply_examples.jsonl` for cleaned paired examples
- `artifacts/chroma_db/` when embedding retrieval is available

Your original WhatsApp export stays wherever you keep it. The CLI stores only the path plus the generated local artifacts.

## Repository Layout And Safe-To-Commit Files

Safe to commit:

- `chatpersona/` source code
- `tests/` and synthetic fixtures under `tests/fixtures/`
- `README.md`, `CONTRIBUTING.md`, and `LICENSE`

Keep local only:

- real WhatsApp exports
- generated `reply_examples.jsonl`
- Chroma indexes and SQLite files
- `~/.chatpersona/` profile storage

## Troubleshooting

### Ollama Is Not Reachable

Run:

```bash
chatpersona doctor
```

If the Ollama check fails, make sure Ollama is installed and the local server is running before retrying.

### No Local Models Found

Install at least one chat model:

```bash
ollama pull llama3.2:latest
```

Then rerun:

```bash
chatpersona doctor
```

### Embedding Retrieval Falls Back To Lexical

If the embedding model is missing or initialization fails, `chatpersona` will fall back to lexical retrieval. This is usable, but lower quality than embedding-backed retrieval.

Install the recommended embedding model:

```bash
ollama pull qwen3-embedding:0.6b
```

Then rebuild the profile:

```bash
chatpersona rebuild "<profile>"
```

`chatpersona profiles list` will show both the preferred retrieval mode and the last actual build mode when they differ.

### Rebuild Or Indexing Takes A Long Time

Large chat exports can take a while to parse and embed. `chatpersona rebuild` shows stage-by-stage progress and embedding batch progress while it works. If the build repeatedly times out, confirm Ollama is healthy with:

```bash
chatpersona doctor
```

### Missing Or Invalid WhatsApp Export

Use a direct 1:1 WhatsApp `.txt` export. Group chats and malformed exports are not supported in this version.

## Preparing A Public Fork Safely

If a real chat export or generated artifact was ever committed to your local git history, deleting the file in the latest commit is not enough. Rewrite the history or start from a clean root commit before pushing the repo to GitHub.

## Development And Testing

Run the test suite with:

```bash
venv/bin/python -m unittest discover -s tests -v
```

Contributor checks:

```bash
python -m ruff check .
python -m ruff format --check .
python -m build
```

For command discovery:

```bash
chatpersona --help
```
