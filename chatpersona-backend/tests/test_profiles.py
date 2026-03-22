from __future__ import annotations

import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

from chatpersona.profiles import (
  ChatProfile,
  compute_source_hash,
  build_profile_artifacts,
  create_profile,
  load_profile,
  profile_needs_rebuild,
  save_profile,
)

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "sample_whatsapp_chat.txt"
SAMPLE_CHAT = FIXTURE_PATH.read_text(encoding="utf-8")


class ProfileStorageTests(unittest.TestCase):
  def test_save_and_load_profile_round_trip(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      export_path = Path(temp_dir) / "chat.txt"
      export_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      with mock.patch.dict(os.environ, {"CHATPERSONA_HOME": temp_dir}, clear=False):
        profile = create_profile(
          profile_name="sam-sample",
          chat_export_path=export_path,
          persona_name="Sam",
          partner_name="Alex",
          chat_model="llama3.2:latest",
          embedding_model=None,
          retrieval_mode="lexical",
        )
        save_profile(profile)
        loaded = load_profile("sam-sample")

      self.assertEqual(loaded.profile_name, "sam-sample")
      self.assertEqual(loaded.persona_name, "Sam")
      self.assertEqual(loaded.chat_model, "llama3.2:latest")
      self.assertEqual(loaded.last_built_retrieval_mode, "pending")

  def test_profile_needs_rebuild_after_source_changes(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      export_path = Path(temp_dir) / "chat.txt"
      export_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      with mock.patch.dict(os.environ, {"CHATPERSONA_HOME": temp_dir}, clear=False):
        profile = create_profile(
          profile_name="sam-sample",
          chat_export_path=export_path,
          persona_name="Sam",
          partner_name="Alex",
          chat_model="llama3.2:latest",
          embedding_model=None,
          retrieval_mode="lexical",
        )
        built_profile, _assets, _warnings = build_profile_artifacts(profile)
        save_profile(built_profile)

        self.assertFalse(profile_needs_rebuild(built_profile))
        self.assertEqual(built_profile.retrieval_mode, "lexical")
        self.assertEqual(built_profile.last_built_retrieval_mode, "lexical")

        export_path.write_text(
          SAMPLE_CHAT + "\n22/08/2025, 09:54 - Alex: library tomorrow?", encoding="utf-8"
        )
        reloaded = load_profile("sam-sample")

        self.assertTrue(profile_needs_rebuild(reloaded))

  def test_legacy_profile_with_embedding_model_defaults_back_to_embedding_preference(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      profile_dir = Path(temp_dir) / "profiles" / "legacy"
      profile_dir.mkdir(parents=True, exist_ok=True)
      profile_json = profile_dir / "profile.json"
      profile_json.write_text(
        textwrap.dedent(
          """
          {
            "profile_name": "legacy",
            "profile_slug": "legacy",
            "chat_export_path": "/tmp/chat.txt",
            "persona_name": "Sam",
            "partner_name": "Alex",
            "chat_model": "llama3.2:latest",
            "embedding_model": "qwen3-embedding:0.6b",
            "retrieval_mode": "lexical",
            "artifact_dir": "/tmp/artifacts",
            "source_hash": "",
            "parser_version": "reply-conditioned-cli-v1"
          }
          """
        ).strip(),
        encoding="utf-8",
      )

      loaded = load_profile("legacy", base_dir=temp_dir)

      self.assertEqual(loaded.retrieval_mode, "embedding")
      self.assertEqual(loaded.last_built_retrieval_mode, "lexical")

  def test_embedding_preference_with_lexical_last_build_is_not_marked_stale(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      export_path = Path(temp_dir) / "chat.txt"
      export_path.write_text(SAMPLE_CHAT, encoding="utf-8")
      artifact_dir = Path(temp_dir) / "profiles" / "sam-sample" / "artifacts"
      artifact_dir.mkdir(parents=True, exist_ok=True)
      (artifact_dir / "reply_examples.jsonl").write_text("{}", encoding="utf-8")

      profile = ChatProfile(
        profile_name="sam-sample",
        profile_slug="sam-sample",
        chat_export_path=str(export_path),
        persona_name="Sam",
        partner_name="Alex",
        chat_model="llama3.2:latest",
        embedding_model="qwen3-embedding:0.6b",
        retrieval_mode="embedding",
        last_built_retrieval_mode="lexical",
        artifact_dir=str(artifact_dir),
        source_hash=compute_source_hash(export_path),
        parser_version="reply-conditioned-cli-v1",
      )

      self.assertFalse(profile_needs_rebuild(profile))

  def test_load_profile_raises_clear_error_for_corrupt_profile_json(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      profile_dir = Path(temp_dir) / "profiles" / "broken"
      profile_dir.mkdir(parents=True, exist_ok=True)
      profile_json = profile_dir / "profile.json"
      profile_json.write_text("{not valid json", encoding="utf-8")

      with self.assertRaisesRegex(ValueError, "Corrupted profile metadata"):
        load_profile("broken", base_dir=temp_dir)


if __name__ == "__main__":
  unittest.main()
