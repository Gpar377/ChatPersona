from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from vector import (
  MessageTurn,
  batched_documents,
  build_dialogue_snippets,
  build_message_turns,
  build_reply_examples,
  build_style_example_bundle,
  compare_reply_shape,
  parse_whatsapp_messages,
  rank_dialogue_snippets,
  rank_reply_examples,
  rank_style_examples,
)


class VectorParsingTests(unittest.TestCase):
  def test_parse_whatsapp_messages_keeps_continuations(self) -> None:
    sample_chat = textwrap.dedent(
      """
      11/08/2025, 18:25 - Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them.
      22/08/2025, 09:50 - Alex: yooo
      22/08/2025, 09:51 - Sam: Wanna go to the library
      22/08/2025, 09:51 - Sam: Wait for me
      22/08/2025, 09:52 - Alex: i am cooked this sem
      need help
      22/08/2025, 09:53 - Sam: Oh no
      22/08/2025, 09:53 - Sam: You should study
      """
    ).strip()

    with tempfile.TemporaryDirectory() as temp_dir:
      chat_path = Path(temp_dir) / "chat.txt"
      chat_path.write_text(sample_chat, encoding="utf-8")
      messages = parse_whatsapp_messages(chat_path)

    self.assertEqual(messages[3].text, "i am cooked this sem\nneed help")
    self.assertEqual(messages[1].sender, "Sam")

  def test_parse_whatsapp_messages_supports_bracketed_export_format(self) -> None:
    sample_chat = textwrap.dedent(
      """
      \u200e[13/11/25, 1:05:55\u202fAM] Person <3: Sequence and Series.pdf • \u200e31 pages \u200edocument omitted
      [13/11/25, 1:06:08\u202fAM] OPDhaker: heyy
      [13/11/25, 1:06:58\u202fAM] Person <3: I forgot my GitHub wala thing 😭
      [13/11/25, 1:08:01\u202fAM] Person <3: Like
      I'll get back on all the stuff
      [13/11/25, 1:08:01\u202fAM] OPDhaker: bro, i was talking to a few founders
      """
    ).strip()

    with tempfile.TemporaryDirectory() as temp_dir:
      chat_path = Path(temp_dir) / "chat.txt"
      chat_path.write_text(sample_chat, encoding="utf-8")
      messages = parse_whatsapp_messages(chat_path)

    self.assertEqual(len(messages), 5)
    self.assertEqual(messages[0].timestamp, "13/11/25 1:05:55 AM")
    self.assertEqual(messages[0].sender, "Person <3")
    self.assertEqual(messages[3].text, "Like\nI'll get back on all the stuff")

  def test_build_message_turns_filters_noise_and_pathological_bursts(self) -> None:
    sample_chat = textwrap.dedent(
      """
      22/08/2025, 09:50 - Alex: yooo
      22/08/2025, 09:51 - Sam: Wanna go to the library
      22/08/2025, 09:51 - Sam: Wait for me
      22/08/2025, 09:52 - Alex: i am cooked this sem
      need help
      22/08/2025, 09:53 - Sam: Oh no
      22/08/2025, 09:53 - Sam: You should study
      22/08/2025, 09:54 - Sam: <Media omitted>
      22/08/2025, 09:55 - Sam: Guys, here are the WhatsApp group links
      1. Software - https://example.com/a
      2. Mechanical - https://example.com/b
      22/08/2025, 09:56 - Sam: h
      22/08/2025, 09:56 - Sam: h
      22/08/2025, 09:56 - Sam: h
      22/08/2025, 09:56 - Sam: h
      22/08/2025, 09:56 - Sam: h
      22/08/2025, 09:56 - Sam: h
      22/08/2025, 09:57 - Alex: library tomorrow?
      22/08/2025, 09:58 - Sam: Lets go
      22/08/2025, 09:58 - Sam: Wait for me
      """
    ).strip()

    with tempfile.TemporaryDirectory() as temp_dir:
      chat_path = Path(temp_dir) / "chat.txt"
      chat_path.write_text(sample_chat, encoding="utf-8")
      messages = parse_whatsapp_messages(chat_path)
      turns = build_message_turns(messages)

    flattened_turns = [turn.flattened() for turn in turns]
    self.assertEqual(
      flattened_turns,
      [
        "yooo",
        "Wanna go to the library | Wait for me",
        "i am cooked this sem\nneed help",
        "Oh no | You should study",
        "library tomorrow?",
        "Lets go | Wait for me",
      ],
    )

  def test_build_reply_examples_returns_real_reply_pairs(self) -> None:
    turns = [
      MessageTurn("Alex", ["yooo"], "t1"),
      MessageTurn("Sam", ["Wanna go to the library", "Wait for me"], "t2"),
      MessageTurn("Alex", ["i am cooked this sem", "need help"], "t3"),
      MessageTurn("Sam", ["Oh no", "You should study"], "t4"),
      MessageTurn("Alex", ["library tomorrow?"], "t5"),
      MessageTurn("Sam", ["Lets go", "Wait for me"], "t6"),
    ]

    examples, partner_name = build_reply_examples(turns, "Sam")

    self.assertEqual(partner_name, "Alex")
    self.assertEqual(len(examples), 3)
    self.assertEqual(examples[1].user_turn.messages, ["i am cooked this sem", "need help"])
    self.assertEqual(examples[1].reply_turn.messages, ["Oh no", "You should study"])


class RetrievalRankingTests(unittest.TestCase):
  def setUp(self) -> None:
    turns = [
      MessageTurn("Alex", ["i feel awful today"], "t1"),
      MessageTurn("Sam", ["Oh no", "Are you okay"], "t2"),
      MessageTurn("Alex", ["wanna go to the library tomorrow"], "t3"),
      MessageTurn("Sam", ["Lets go", "Wait for me"], "t4"),
      MessageTurn("Alex", ["you are so cute hehe"], "t5"),
      MessageTurn("Sam", ["Hehe", "Sus"], "t6"),
      MessageTurn("Alex", ["maths pdf bhej na"], "t7"),
      MessageTurn("Sam", ["Sure sure", "Also send me the graph one"], "t8"),
    ]
    self.examples, _ = build_reply_examples(turns, "Sam")

  def test_supportive_queries_rank_supportive_example_first(self) -> None:
    ranked = rank_reply_examples("i feel so awful and tired rn", [], self.examples)
    self.assertIn("supportive", ranked[0].tags)
    self.assertEqual(ranked[0].reply_turn.messages, ["Oh no", "Are you okay"])

  def test_planning_queries_rank_planning_example_first(self) -> None:
    ranked = rank_reply_examples("library tomorrow after class?", [], self.examples)
    self.assertIn("planning", ranked[0].tags)
    self.assertEqual(ranked[0].reply_turn.messages, ["Lets go", "Wait for me"])

  def test_playful_queries_rank_playful_example_first(self) -> None:
    ranked = rank_reply_examples("hehe you are cute", [], self.examples)
    self.assertIn("playful", ranked[0].tags)
    self.assertEqual(ranked[0].reply_turn.messages, ["Hehe", "Sus"])

  def test_style_queries_prefer_short_shape_examples(self) -> None:
    ranked = rank_style_examples("i feel awful and tired", [], self.examples)

    self.assertIn("supportive", ranked[0].tags)
    self.assertLessEqual(ranked[0].reply_burst_count, 3)

  def test_dialogue_snippets_keep_contiguous_setup_and_follow_through(self) -> None:
    turns = [
      MessageTurn("Alex", ["heyyy"], "t1"),
      MessageTurn("Sam", ["heyy", "kya kar raha hai"], "t2"),
      MessageTurn("Alex", ["nothing much was about to go code"], "t3"),
      MessageTurn("Sam", ["haa sure"], "t4"),
      MessageTurn("Alex", ["you going library later?"], "t5"),
      MessageTurn("Sam", ["maybe", "let me finish this first"], "t6"),
    ]

    snippets, partner_name = build_dialogue_snippets(turns, "Sam")

    self.assertEqual(partner_name, "Alex")
    self.assertEqual(len(snippets), 3)
    self.assertEqual(snippets[1].lead_context[-1].flattened(), "heyy | kya kar raha hai")
    self.assertEqual(snippets[1].partner_turn.flattened(), "nothing much was about to go code")
    self.assertEqual(snippets[1].reply_turn.flattened(), "haa sure")
    self.assertEqual(snippets[1].trailing_context[0].flattened(), "you going library later?")

  def test_dialogue_ranking_prefers_broader_exchange_over_isolated_short_reply(self) -> None:
    turns = [
      MessageTurn("Alex", ["heyyy"], "t1"),
      MessageTurn("Sam", ["heyy", "kya kar raha hai"], "t2"),
      MessageTurn("Alex", ["nothing much was about to go code"], "t3"),
      MessageTurn("Sam", ["haa sure"], "t4"),
      MessageTurn("Alex", ["library tomorrow?"], "t5"),
      MessageTurn("Sam", ["lets go", "wait for me"], "t6"),
    ]
    snippets, _ = build_dialogue_snippets(turns, "Sam")

    ranked = rank_dialogue_snippets(
      "coding kar raha hu abhi",
      [
        MessageTurn("Alex", ["heyyy"], "live-1"),
        MessageTurn("Sam", ["heyy", "kya kar raha hai"], "live-2"),
      ],
      snippets,
      friend_name="Sam",
      partner_name="Alex",
    )

    self.assertEqual(ranked[0].partner_turn.flattened(), "nothing much was about to go code")
    self.assertEqual(ranked[0].lead_context[-1].flattened(), "heyy | kya kar raha hai")
    self.assertNotEqual(ranked[0].reply_turn.flattened(), "lets go | wait for me")


class ReplyShapeTests(unittest.TestCase):
  def test_compare_reply_shape_reports_basic_deltas(self) -> None:
    reference_turn = MessageTurn("Sam", ["Haa okayy", "kal milte"], "t1")
    comparison = compare_reply_shape(["Haa okayy", "kal milte"], reference_turn)

    self.assertEqual(comparison["burst_delta"], 0)
    self.assertEqual(comparison["character_delta"], 0)
    self.assertEqual(comparison["code_switch_delta"], 0.0)

  def test_batched_documents_splits_large_inputs(self) -> None:
    documents = list(range(130))

    batches = batched_documents(documents, batch_size=64)

    self.assertEqual([len(batch) for batch in batches], [64, 64, 2])

  def test_build_style_example_bundle_summarizes_rhythm_without_context_dump(self) -> None:
    turns = [
      MessageTurn("Alex", ["i feel awful today"], "t1"),
      MessageTurn("Sam", ["Oh no", "Are you okay"], "t2"),
      MessageTurn("Alex", ["library tomorrow?"], "t3"),
      MessageTurn("Sam", ["Lets go", "Wait for me"], "t4"),
    ]
    examples, _ = build_reply_examples(turns, "Sam")

    bundle = build_style_example_bundle(examples)
    bundle_text = bundle.to_prompt_text()

    self.assertIn("Similar replies average", bundle_text)
    self.assertIn("Style evidence", bundle_text)
    self.assertNotIn("Alex:", bundle_text)


if __name__ == "__main__":
  unittest.main()
