from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
from typing import Sequence
from urllib import error, request

DEFAULT_CHAT_MODEL = "llama3.2:latest"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
EMBEDDING_MODEL_HINTS = ("embedding", "embed")
PREFERRED_EMBEDDING_MODELS = (
  "qwen3-embedding:0.6b",
  "nomic-embed-text",
  "mxbai-embed-large",
  "all-minilm",
  "granite-embedding",
)


@dataclass(slots=True)
class OllamaModel:
  name: str
  size: str = ""
  modified_at: str = ""
  source: str = ""

  @property
  def kind(self) -> str:
    return "embedding" if is_embedding_model_name(self.name) else "chat"


def humanize_bytes(size_bytes: int | None) -> str:
  if not size_bytes:
    return ""
  size = float(size_bytes)
  for unit in ("B", "KB", "MB", "GB", "TB"):
    if size < 1024.0 or unit == "TB":
      if unit == "B":
        return f"{int(size)} {unit}"
      return f"{size:.1f} {unit}"
    size /= 1024.0
  return ""


def is_embedding_model_name(name: str) -> bool:
  lowered = name.lower()
  return any(hint in lowered for hint in EMBEDDING_MODEL_HINTS)


def filter_chat_models(models: Sequence[OllamaModel]) -> list[OllamaModel]:
  chat_models = [model for model in models if not is_embedding_model_name(model.name)]
  return chat_models or list(models)


def choose_embedding_model(models: Sequence[OllamaModel]) -> str | None:
  lower_to_original = {model.name.lower(): model.name for model in models}
  for preferred in PREFERRED_EMBEDDING_MODELS:
    if preferred.lower() in lower_to_original:
      return lower_to_original[preferred.lower()]

  for model in models:
    if is_embedding_model_name(model.name):
      return model.name
  return None


def choose_default_chat_model(models: Sequence[OllamaModel]) -> str:
  chat_models = filter_chat_models(models)
  names = {model.name for model in chat_models}
  if DEFAULT_CHAT_MODEL in names:
    return DEFAULT_CHAT_MODEL

  for preferred_prefix in ("llama3.2", "qwen", "mistral", "gemma", "llama"):
    for model in chat_models:
      if model.name.startswith(preferred_prefix):
        return model.name

  if chat_models:
    return chat_models[0].name
  return DEFAULT_CHAT_MODEL


def parse_ollama_list_output(output: str) -> list[OllamaModel]:
  lines = [line.rstrip() for line in output.splitlines() if line.strip()]
  if not lines:
    return []

  if lines[0].lower().startswith("name"):
    lines = lines[1:]

  models: list[OllamaModel] = []
  for line in lines:
    if re.match(r"^(warning|error):", line, re.IGNORECASE):
      continue
    parts = re.split(r"\s{2,}", line.strip())
    if not parts:
      continue
    name = parts[0].strip()
    if not name:
      continue
    size = parts[2].strip() if len(parts) > 2 else ""
    modified_at = parts[3].strip() if len(parts) > 3 else ""
    models.append(OllamaModel(name=name, size=size, modified_at=modified_at, source="cli"))

  return models


def parse_ollama_tags_payload(payload: str | bytes) -> list[OllamaModel]:
  raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
  parsed = json.loads(raw)
  models: list[OllamaModel] = []

  for item in parsed.get("models", []):
    if not isinstance(item, dict):
      continue
    name = str(item.get("name", "")).strip()
    if not name:
      continue
    size_value = item.get("size")
    models.append(
      OllamaModel(
        name=name,
        size=humanize_bytes(size_value if isinstance(size_value, int) else None),
        modified_at=str(item.get("modified_at", "")).strip(),
        source="api",
      )
    )

  return models


def list_models_via_api(timeout_seconds: float = 2.0) -> list[OllamaModel]:
  response = request.urlopen(OLLAMA_TAGS_URL, timeout=timeout_seconds)
  payload = response.read()
  return parse_ollama_tags_payload(payload)


def list_models_via_cli(timeout_seconds: float = 5.0) -> list[OllamaModel]:
  result = subprocess.run(
    ["ollama", "list"],
    check=False,
    capture_output=True,
    text=True,
    timeout=timeout_seconds,
  )
  if result.returncode != 0:
    details = result.stderr.strip() or result.stdout.strip() or "unknown error"
    raise RuntimeError(details)
  return parse_ollama_list_output(result.stdout)


def list_local_models(timeout_seconds: float = 2.0) -> list[OllamaModel]:
  errors_seen: list[str] = []

  try:
    models = list_models_via_api(timeout_seconds=timeout_seconds)
    if models:
      return models
    errors_seen.append("Ollama API returned no local models.")
  except (
    error.URLError,
    TimeoutError,
    RuntimeError,
    json.JSONDecodeError,
    OSError,
    ValueError,
  ) as exc:
    errors_seen.append(f"api: {exc}")

  try:
    models = list_models_via_cli(timeout_seconds=max(timeout_seconds, 5.0))
    if models:
      return models
    errors_seen.append("`ollama list` returned no local models.")
  except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
    errors_seen.append(f"cli: {exc}")

  raise RuntimeError(
    "Could not list local Ollama models. Make sure Ollama is installed and the local server is running. "
    + " | ".join(errors_seen)
  )
