from __future__ import annotations

from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rich.console import Console

from chatpersona.corpus import (
  ChatAssets,
  MessageTurn,
  StyleProfile,
  build_reply_examples,
  build_style_example_bundle,
)
from chatpersona.profiles import create_profile
from chatpersona.runtime import (
  build_conversation_guidance,
  build_dialogue_state,
  interactive_chat_session,
  score_local_reply_candidate,
)


class _DummyStatus:
  def __init__(self) -> None:
    self.messages: list[str] = []

  def __enter__(self) -> "_DummyStatus":
    return self

  def __exit__(self, exc_type, exc, tb) -> bool:
    return False

  def update(self, message: str) -> None:
    self.messages.append(message)


class RuntimeWorkspaceTests(unittest.TestCase):
  def make_console(self) -> tuple[Console, StringIO]:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120)
    console.clear = mock.Mock()  # type: ignore[method-assign]
    return console, stream

  def make_profile_and_assets(self):
    profile = create_profile(
      profile_name="sam-sample",
      chat_export_path="chat.txt",
      persona_name="Sam",
      partner_name="Alex",
      chat_model="llama3.2:latest",
      embedding_model="qwen3-embedding:0.6b",
      retrieval_mode="embedding",
      base_dir=Path(tempfile.gettempdir()) / "chatpersona-runtime-tests",
    )
    profile.last_built_retrieval_mode = "embedding"
    assets = ChatAssets(
      friend_name="Sam",
      partner_name="Alex",
      style_profile=StyleProfile(
        avg_reply_burst=2.0,
        median_message_length=24,
        common_tokens=["haa", "kya", "tum"],
        common_phrases=["kya hua", "tum kya kar rahe ho"],
        tone_notes=["Warm and casual, with short back-and-forth replies."],
      ),
      reply_examples=[],
      dialogue_snippets=[],
      retriever=mock.Mock(),
      retrieval_mode="embedding",
      artifact_dir=Path("/tmp/chatpersona-runtime"),
    )
    return profile, assets

  def test_workspace_help_and_profile_commands_render_sections(self) -> None:
    console, stream = self.make_console()
    profile, assets = self.make_profile_and_assets()
    console.input = mock.Mock(side_effect=["/help", "/profile", "/quit"])  # type: ignore[method-assign]
    console.status = mock.Mock(return_value=_DummyStatus())  # type: ignore[method-assign]

    with (
      mock.patch("chatpersona.runtime.build_planner_prompt", return_value=object()),
      mock.patch("chatpersona.runtime.build_style_prompt", return_value=object()),
      mock.patch("chatpersona.runtime.build_model", return_value=object()),
    ):
      interactive_chat_session(profile, assets, console)

    output = stream.getvalue()
    self.assertIn("Help", output)
    self.assertIn("Profile", output)
    self.assertIn("qwen3-embedding:0.6b", output)

  def test_workspace_retry_regenerates_last_reply(self) -> None:
    console, stream = self.make_console()
    profile, assets = self.make_profile_and_assets()
    console.input = mock.Mock(side_effect=["heyy", "/retry", "/quit"])  # type: ignore[method-assign]
    console.status = mock.Mock(return_value=_DummyStatus())  # type: ignore[method-assign]

    with (
      mock.patch("chatpersona.runtime.build_planner_prompt", return_value=object()),
      mock.patch("chatpersona.runtime.build_style_prompt", return_value=object()),
      mock.patch("chatpersona.runtime.build_model", return_value=object()),
      mock.patch(
        "chatpersona.runtime.generate_reply",
        side_effect=[["heyy", "tum kya kar rahe ho"], ["hii", "ab bata"]],
      ) as generate_reply_mock,
    ):
      interactive_chat_session(profile, assets, console)

    output = stream.getvalue()
    self.assertEqual(generate_reply_mock.call_count, 2)
    self.assertIn("Reply regenerated.", output)
    self.assertIn("ab bata", output)

  def test_workspace_clear_resets_live_transcript(self) -> None:
    console, stream = self.make_console()
    profile, assets = self.make_profile_and_assets()
    console.input = mock.Mock(side_effect=["heyy", "/clear", "/quit"])  # type: ignore[method-assign]
    console.status = mock.Mock(return_value=_DummyStatus())  # type: ignore[method-assign]

    with (
      mock.patch("chatpersona.runtime.build_planner_prompt", return_value=object()),
      mock.patch("chatpersona.runtime.build_style_prompt", return_value=object()),
      mock.patch("chatpersona.runtime.build_model", return_value=object()),
      mock.patch(
        "chatpersona.runtime.generate_reply", return_value=["heyy", "tum kya kar rahe ho"]
      ),
    ):
      interactive_chat_session(profile, assets, console)

    output = stream.getvalue()
    self.assertIn("Session cleared", output)
    self.assertIn("Start the conversation.", output)

  def test_dialogue_state_preserves_topic_across_recent_turns(self) -> None:
    state = build_dialogue_state(
      [
        MessageTurn("Alex", ["heyyy"], "t1"),
        MessageTurn("Sam", ["heyy", "kya kar raha hai"], "t2"),
        MessageTurn("Alex", ["coding kar raha hu assignment ke liye"], "t3"),
      ],
      "Alex",
      "Sam",
    )

    self.assertTrue(any(topic in {"coding", "assignment"} for topic in state.active_topics))
    self.assertTrue(state.unresolved_questions)

  def test_local_reply_eval_penalizes_copied_archive_wording(self) -> None:
    turns = [
      MessageTurn("Alex", ["i feel awful today"], "t1"),
      MessageTurn("Sam", ["Oh no", "Are you okay"], "t2"),
      MessageTurn("Alex", ["library tomorrow?"], "t3"),
      MessageTurn("Sam", ["Lets go", "Wait for me"], "t4"),
    ]
    examples, _ = build_reply_examples(turns, "Sam")
    style_bundle = build_style_example_bundle(examples[:1])
    guidance = build_conversation_guidance("i feel awful today")

    copied = score_local_reply_candidate(
      "i feel awful today",
      ["Oh no", "Are you okay"],
      guidance,
      style_bundle,
    )
    fresh = score_local_reply_candidate(
      "i feel awful today",
      ["thats really rough", "kya hua"],
      guidance,
      style_bundle,
    )

    self.assertGreater(copied.archive_copy_rate, fresh.archive_copy_rate)


if __name__ == "__main__":
  unittest.main()
