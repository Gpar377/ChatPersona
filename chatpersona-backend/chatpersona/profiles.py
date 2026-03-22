from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
from typing import Iterable

from chatpersona.corpus import (
  BuildProgressCallback,
  DATASET_EXPORT_FILENAME,
  DEFAULT_DB_DIRNAME,
  MANIFEST_FILENAME,
  PARSER_VERSION,
  ChatAssets,
  build_chat_assets,
  compute_source_hash,
)

APP_HOME_ENV = "CHATPERSONA_HOME"
DEFAULT_APP_DIRNAME = ".chatpersona"
PROFILE_FILENAME = "profile.json"


@dataclass(slots=True)
class ChatProfile:
  profile_name: str
  profile_slug: str
  chat_export_path: str
  persona_name: str
  partner_name: str
  chat_model: str
  embedding_model: str | None
  retrieval_mode: str
  last_built_retrieval_mode: str
  artifact_dir: str
  source_hash: str
  parser_version: str

  @property
  def export_path(self) -> Path:
    return Path(self.chat_export_path).expanduser()

  @property
  def artifacts_path(self) -> Path:
    return Path(self.artifact_dir).expanduser()

  @property
  def profile_dir(self) -> Path:
    return self.artifacts_path.parent

  def to_dict(self) -> dict[str, object]:
    return {
      "profile_name": self.profile_name,
      "profile_slug": self.profile_slug,
      "chat_export_path": self.chat_export_path,
      "persona_name": self.persona_name,
      "partner_name": self.partner_name,
      "chat_model": self.chat_model,
      "embedding_model": self.embedding_model,
      "retrieval_mode": self.retrieval_mode,
      "last_built_retrieval_mode": self.last_built_retrieval_mode,
      "artifact_dir": self.artifact_dir,
      "source_hash": self.source_hash,
      "parser_version": self.parser_version,
    }

  @classmethod
  def from_dict(cls, payload: dict[str, object]) -> "ChatProfile":
    saved_retrieval_mode = str(payload.get("retrieval_mode", "lexical"))
    embedding_model = str(payload["embedding_model"]) if payload.get("embedding_model") else None
    preferred_retrieval_mode = str(
      payload.get("preferred_retrieval_mode")
      or ("embedding" if embedding_model else saved_retrieval_mode)
    )
    return cls(
      profile_name=str(payload["profile_name"]),
      profile_slug=str(payload.get("profile_slug") or slugify_name(str(payload["profile_name"]))),
      chat_export_path=str(payload["chat_export_path"]),
      persona_name=str(payload["persona_name"]),
      partner_name=str(payload.get("partner_name", "")),
      chat_model=str(payload["chat_model"]),
      embedding_model=embedding_model,
      retrieval_mode=preferred_retrieval_mode,
      last_built_retrieval_mode=str(payload.get("last_built_retrieval_mode", saved_retrieval_mode)),
      artifact_dir=str(payload["artifact_dir"]),
      source_hash=str(payload.get("source_hash", "")),
      parser_version=str(payload.get("parser_version", PARSER_VERSION)),
    )


def slugify_name(value: str) -> str:
  slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
  return slug or "profile"


def get_app_home(base_dir: str | Path | None = None) -> Path:
  if base_dir is not None:
    return Path(base_dir).expanduser().resolve()
  override = os.environ.get(APP_HOME_ENV)
  if override:
    return Path(override).expanduser().resolve()
  return (Path.home() / DEFAULT_APP_DIRNAME).resolve()


def get_profiles_dir(base_dir: str | Path | None = None) -> Path:
  return get_app_home(base_dir) / "profiles"


def get_profile_dir(profile_slug: str, base_dir: str | Path | None = None) -> Path:
  return get_profiles_dir(base_dir) / profile_slug


def get_profile_json_path(profile_slug: str, base_dir: str | Path | None = None) -> Path:
  return get_profile_dir(profile_slug, base_dir) / PROFILE_FILENAME


def create_profile(
  *,
  profile_name: str,
  chat_export_path: str | Path,
  persona_name: str,
  partner_name: str,
  chat_model: str,
  embedding_model: str | None,
  retrieval_mode: str,
  base_dir: str | Path | None = None,
) -> ChatProfile:
  profile_slug = slugify_name(profile_name)
  profile_dir = get_profile_dir(profile_slug, base_dir)
  artifact_dir = profile_dir / "artifacts"
  export_path = Path(chat_export_path).expanduser().resolve()
  return ChatProfile(
    profile_name=profile_name,
    profile_slug=profile_slug,
    chat_export_path=str(export_path),
    persona_name=persona_name,
    partner_name=partner_name,
    chat_model=chat_model,
    embedding_model=embedding_model,
    retrieval_mode=retrieval_mode,
    last_built_retrieval_mode="pending",
    artifact_dir=str(artifact_dir),
    source_hash="",
    parser_version=PARSER_VERSION,
  )


def save_profile(profile: ChatProfile, base_dir: str | Path | None = None) -> Path:
  profile_dir = get_profile_dir(profile.profile_slug, base_dir)
  profile_dir.mkdir(parents=True, exist_ok=True)
  profile.artifacts_path.mkdir(parents=True, exist_ok=True)
  target_path = profile_dir / PROFILE_FILENAME
  target_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
  return target_path


def load_profile(identifier: str, base_dir: str | Path | None = None) -> ChatProfile:
  profiles_dir = get_profiles_dir(base_dir)
  if not profiles_dir.exists():
    raise FileNotFoundError("No saved profiles were found yet.")

  exact_path = get_profile_json_path(slugify_name(identifier), base_dir)
  if exact_path.exists():
    try:
      payload = json.loads(exact_path.read_text(encoding="utf-8"))
      return ChatProfile.from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
      raise ValueError(f"Corrupted profile metadata at `{exact_path}`.") from exc

  for profile in list_profiles(base_dir):
    if profile.profile_name == identifier or profile.profile_slug == identifier:
      return profile

  raise FileNotFoundError(f"No saved profile named `{identifier}` was found.")


def list_profiles(base_dir: str | Path | None = None) -> list[ChatProfile]:
  profiles_dir = get_profiles_dir(base_dir)
  if not profiles_dir.exists():
    return []

  loaded_profiles: list[ChatProfile] = []
  for profile_json in sorted(profiles_dir.glob(f"*/{PROFILE_FILENAME}")):
    try:
      payload = json.loads(profile_json.read_text(encoding="utf-8"))
      loaded_profiles.append(ChatProfile.from_dict(payload))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
      continue
  return sorted(
    loaded_profiles, key=lambda profile: (profile.profile_name.lower(), profile.profile_slug)
  )


def profile_artifact_paths(profile: ChatProfile) -> Iterable[Path]:
  yield profile.artifacts_path / DATASET_EXPORT_FILENAME
  if profile.last_built_retrieval_mode == "embedding":
    yield profile.artifacts_path / DEFAULT_DB_DIRNAME
    yield profile.artifacts_path / MANIFEST_FILENAME


def profile_needs_rebuild(profile: ChatProfile) -> bool:
  if profile.parser_version != PARSER_VERSION:
    return True
  if not profile.export_path.exists():
    return True
  if not profile.source_hash:
    return True
  if compute_source_hash(profile.export_path) != profile.source_hash:
    return True
  return any(not artifact_path.exists() for artifact_path in profile_artifact_paths(profile))


def build_profile_artifacts(
  profile: ChatProfile,
  progress_callback: BuildProgressCallback | None = None,
) -> tuple[ChatProfile, ChatAssets, list[str]]:
  assets, warnings = build_chat_assets(
    profile.chat_export_path,
    profile.persona_name,
    profile.artifact_dir,
    embedding_model=profile.embedding_model,
    retrieval_mode=profile.retrieval_mode,
    progress_callback=progress_callback,
  )
  updated_profile = ChatProfile(
    profile_name=profile.profile_name,
    profile_slug=profile.profile_slug,
    chat_export_path=str(profile.export_path.resolve()),
    persona_name=profile.persona_name,
    partner_name=assets.partner_name,
    chat_model=profile.chat_model,
    embedding_model=profile.embedding_model,
    retrieval_mode=profile.retrieval_mode,
    last_built_retrieval_mode=assets.retrieval_mode,
    artifact_dir=str(profile.artifacts_path),
    source_hash=compute_source_hash(profile.export_path),
    parser_version=PARSER_VERSION,
  )
  return updated_profile, assets, warnings
