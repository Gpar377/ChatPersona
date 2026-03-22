"""
ChatPersona Web Server
Flask REST API that bridges the chatpersona Python backend to the web frontend.

Endpoints:
  POST /upload             - Upload a WhatsApp .txt export and build a persona profile
  POST /chat               - Send a message, get an AI reply as the persona
  GET  /profiles           - List all saved profiles
  DELETE /profiles/<slug>  - Delete a profile
  DELETE /sessions/<slug>  - Clear live chat session memory
  GET  /status             - Health check + Ollama connectivity
"""

from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Any

# ── Make chatpersona importable from sibling folder ──────────────────────────
BACKEND_DIR = Path(__file__).parent / "chatpersona-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# chatpersona internals
from chatpersona.ollama import (
    list_local_models,
    choose_default_chat_model,
    choose_embedding_model,
    filter_chat_models,
)
from chatpersona.profiles import (
    ChatProfile,
    create_profile,
    list_profiles,
    save_profile,
    build_profile_artifacts,
    get_profile_dir,
    get_profile_json_path,
    slugify_name,
    PROFILE_FILENAME,
)
from chatpersona.corpus import MessageTurn
from chatpersona.runtime import (
    generate_reply,
    build_planner_prompt,
    build_style_prompt,
    build_model,
    PRIMARY_TIMEOUT_SECONDS,
    FALLBACK_TIMEOUT_SECONDS,
)

# ── App setup ────────────────────────────────────────────────────────────────
UPLOAD_DIR   = Path(__file__).parent / "uploads"
FRONTEND_DIR = Path(__file__).parent / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

# In-memory: profile_slug -> list[MessageTurn]
_sessions: dict[str, list[MessageTurn]] = {}
# Cached ChatAssets: profile_slug -> ChatAssets
_assets_cache: dict[str, Any] = {}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _err(msg: str, code: int = 400) -> Any:
    return jsonify({"ok": False, "error": msg}), code

def _ok(payload: dict) -> Any:
    return jsonify({"ok": True, **payload})

def _load_profile_from_slug(slug: str) -> ChatProfile:
    path = get_profile_json_path(slug)
    if not path.exists():
        raise FileNotFoundError(f"Profile '{slug}' not found")
    return ChatProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))

def _get_assets(profile: ChatProfile):
    slug = profile.profile_slug
    if slug not in _assets_cache:
        # Assets were built during /upload; rebuild if cache is cold
        _, assets, _ = build_profile_artifacts(profile)
        _assets_cache[slug] = assets
    return _assets_cache[slug]

# ── Static / index ───────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/<path:path>")
def serve_static(path: str):
    return send_from_directory(str(FRONTEND_DIR), path)

# ── Health / discovery ───────────────────────────────────────────────────────
@app.route("/status")
def status():
    try:
        models      = list_local_models()
        chat_models = [m.name for m in filter_chat_models(models)]
        return _ok({"ollama": True, "chat_models": chat_models, "total_models": len(models)})
    except Exception as exc:
        return _ok({"ollama": False, "error": str(exc), "chat_models": []})

# ── Profiles ─────────────────────────────────────────────────────────────────
@app.route("/profiles")
def get_profiles():
    try:
        profiles = list_profiles()
        return _ok({
            "profiles": [
                {
                    "slug":      p.profile_slug,
                    "name":      p.profile_name,
                    "persona":   p.persona_name,
                    "partner":   p.partner_name,
                    "model":     p.chat_model,
                    "retrieval": p.retrieval_mode,
                }
                for p in profiles
            ]
        })
    except Exception as exc:
        return _err(f"Could not load profiles: {exc}")

@app.route("/profiles/<slug>", methods=["DELETE"])
def delete_profile(slug: str):
    import shutil
    profile_dir = get_profile_dir(slug)
    if not profile_dir.exists():
        return _err("Profile not found", 404)
    shutil.rmtree(profile_dir, ignore_errors=True)
    _sessions.pop(slug, None)
    _assets_cache.pop(slug, None)
    for f in UPLOAD_DIR.glob(f"{slug}*"):
        f.unlink(missing_ok=True)
    return _ok({"deleted": slug})

# ── Upload & build ───────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload a WhatsApp .txt export and build a chatpersona profile.
    Form fields: file (required), persona (required), partner (optional), model (optional)
    """
    if "file" not in request.files:
        return _err("No file provided")

    f = request.files["file"]
    if not f.filename or not f.filename.endswith(".txt"):
        return _err("Only WhatsApp .txt exports are supported")

    persona_name = (request.form.get("persona") or "").strip()
    partner_name = (request.form.get("partner") or "").strip()
    model_name   = (request.form.get("model")   or "").strip()

    if not persona_name:
        return _err("'persona' field is required")

    # Detect models
    try:
        models = list_local_models()
    except Exception as exc:
        return _err(f"Ollama is not reachable: {exc}. Is Ollama running?")

    if not model_name:
        model_name = choose_default_chat_model(models)
    embedding_model = choose_embedding_model(models)
    retrieval_mode  = "embedding" if embedding_model else "lexical"

    # Save the uploaded file
    slug        = slugify_name(f"{persona_name}")
    export_path = UPLOAD_DIR / f"{slug}.txt"
    f.save(str(export_path))

    # Create profile metadata
    profile_name = f"{persona_name} - chat"
    try:
        profile = create_profile(
            profile_name    = profile_name,
            chat_export_path= export_path,
            persona_name    = persona_name,
            partner_name    = partner_name or "You",
            chat_model      = model_name,
            embedding_model = embedding_model,
            retrieval_mode  = retrieval_mode,
        )
        save_profile(profile)
    except Exception as exc:
        traceback.print_exc()
        return _err(f"Failed to create profile: {exc}")

    # Parse & index the chat
    try:
        updated_profile, assets, warnings = build_profile_artifacts(profile)
        save_profile(updated_profile)
        _assets_cache[updated_profile.profile_slug] = assets
    except Exception as exc:
        traceback.print_exc()
        return _err(f"Failed to build chat assets: {exc}")

    _sessions[updated_profile.profile_slug] = []

    return _ok({
        "slug":      updated_profile.profile_slug,
        "name":      updated_profile.profile_name,
        "persona":   updated_profile.persona_name,
        "partner":   assets.partner_name or partner_name or "You",
        "model":     model_name,
        "retrieval": retrieval_mode,
        "warnings":  warnings,
    })

# ── Chat ─────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    """
    Send a message; receive bubbles back as the persona.
    JSON body: { message, slug }
    """
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    slug    = (body.get("slug")    or "").strip()

    if not message:
        return _err("'message' is required")
    if not slug:
        return _err("'slug' is required")

    try:
        profile = _load_profile_from_slug(slug)
    except FileNotFoundError as exc:
        return _err(str(exc), 404)
    except Exception as exc:
        return _err(f"Could not load profile: {exc}")

    try:
        assets = _get_assets(profile)
    except Exception as exc:
        traceback.print_exc()
        return _err(f"Could not load chat assets: {exc}")

    session = _sessions.setdefault(slug, [])

    try:
        planner_prompt = build_planner_prompt()
        style_prompt   = build_style_prompt()
        primary_model  = build_model(profile.chat_model, PRIMARY_TIMEOUT_SECONDS,  num_predict=120)
        fallback_model = build_model(profile.chat_model, FALLBACK_TIMEOUT_SECONDS, num_predict=80)
    except Exception as exc:
        return _err(f"Could not initialise model: {exc}")

    try:
        reply_lines = generate_reply(
            planner_prompt,
            style_prompt,
            primary_model,
            fallback_model,
            assets,
            message,
            session,
        )
    except Exception as exc:
        traceback.print_exc()
        return _err(f"Generation failed: {exc}", 500)

    # Persist turns into session
    session.append(MessageTurn(speaker=assets.partner_name, messages=[message],  source="live"))
    session.append(MessageTurn(speaker=assets.friend_name,  messages=reply_lines, source="live"))

    return _ok({
        "reply":   "\n".join(reply_lines),
        "bubbles": reply_lines,
        "persona": assets.friend_name,
        "partner": assets.partner_name,
    })

# ── Session ──────────────────────────────────────────────────────────────────
@app.route("/sessions/<slug>", methods=["DELETE"])
def clear_session(slug: str):
    _sessions.pop(slug, None)
    return _ok({"cleared": slug})

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🧠 ChatPersona Web Server → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=False)
