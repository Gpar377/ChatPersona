<div align="center">

# 🧠 ChatPersona

### *"In case your friend goes missing and you miss them a lot, just upload your WhatsApp chat export onto our website and have an AI talk like them instead. Who needs a real friend anyway?"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-3.0-lightgrey.svg)](https://flask.palletsprojects.com)
[![Ollama](https://img.shields.io/badge/ollama-local%20LLM-orange.svg)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What is this?

ChatPersona lets you upload a WhatsApp `.txt` chat export, pick the person to emulate, and then actually *talk to them* — except it's an AI that has studied their texting style, vocabulary, reply patterns, emoji usage, and even their Hindi-English code-switching.

Built with:
- 🐍 **Python** backend (`chatpersona` engine) — RAG + local LLM via Ollama
- ⚗️ **Flask** REST API server bridging the brain to the browser
- 🎨 **Vanilla JS/CSS** frontend — dark glassmorphism chat UI
- 🦙 **Ollama** (runs 100% locally — no data leaves your machine)

---

## How it works

```
WhatsApp .txt export
        │
        ▼
  [corpus.py] parse & clean messages
        │
        ▼
  [ChromaDB] embed reply examples (semantic search)
        │
        ▼
  [Ollama LLM] 2-stage generation:
    1. Planner  → decides *what* to say
    2. Styler   → says it *how they would*
        │
        ▼
  Flask /chat → browser chat UI
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- At least one local model: `ollama pull llama3.2:latest`
- Optional (for embeddings): `ollama pull qwen3-embedding:0.6b`

### 2. Clone & install

```bash
git clone https://github.com/Gpar377/ChatPersona.git
cd ChatPersona

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install flask flask-cors
pip install -e ./chatpersona-backend
```

### 3. Run

```bash
python server.py
```

Open **http://localhost:5000** in your browser.

---

## Usage

1. Export a WhatsApp chat: **Chat → ⋮ → More → Export Chat → Without Media**
2. Open the app, upload the `.txt` file
3. Enter the name of the person to emulate
4. Click **Build Persona** (takes 10–30 seconds to index)
5. Start chatting 💬

---

## Project Structure

```
ChatPersona/
├── server.py                  ← Flask API (the bridge)
├── requirements.txt
├── frontend/                  ← Web UI
│   ├── index.html
│   ├── style.css
│   └── app.js
├── chatpersona-backend/       ← Core AI engine (submodule)
│   └── chatpersona/
│       ├── corpus.py          ← WhatsApp parser + RAG indexer
│       ├── runtime.py         ← 2-stage LLM generation
│       ├── profiles.py        ← Profile storage
│       └── ollama.py          ← Model management
├── frontend-src/              ← Original frontend (reference)
└── uploads/                   ← Uploaded .txt files (gitignored)
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/status` | Ollama health + available models |
| `GET`  | `/profiles` | List all saved personas |
| `POST` | `/upload` | Upload `.txt` + build persona |
| `POST` | `/chat` | Send message, get reply bubbles |
| `DELETE` | `/profiles/<slug>` | Delete a persona |
| `DELETE` | `/sessions/<slug>` | Clear chat memory |

---

## Privacy

Everything runs **100% locally**. Your chat export never leaves your machine. No API keys, no cloud, no data collection.

---

## Team

Built at an internal hackathon.

- **Backend engine**: [@OPDhaker](https://github.com/OPDhaker) — `chatpersona` core
- **Frontend v1**: [@tanushkeshri](https://github.com/tanushkeshri) — `internal-hacathon`
- **Integration & polish**: [@Gpar377](https://github.com/Gpar377)

---

> *"Who needs real friends when you have local LLMs?"* 🤖
