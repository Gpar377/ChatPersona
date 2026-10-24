from __future__ import annotations

import unittest

from main import (
  build_dialogue_state,
  build_conversation_guidance,
  build_last_resort_plan,
  build_last_resort_reply,
  build_session_summary,
  evaluate_generation_quality,
  is_low_substance_reply,
  score_local_reply_candidate,
)
from vector import MessageTurn, build_reply_examples, build_style_example_bundle


class ConversationGuidanceTests(unittest.TestCase):
  def test_greeting_guidance_requires_engagement(self) -> None:
    guidance = build_conversation_guidance("heyy")

    self.assertIn("greeting", guidance.tags)
    self.assertGreaterEqual(guidance.minimum_bursts, 2)
    self.assertTrue(guidance.prefer_question)

  def test_supportive_messages_flag_dry_replies(self) -> None:
    guidance = build_conversation_guidance("i had a really bad day today")

    self.assertIn("supportive", guidance.tags)
    self.assertTrue(is_low_substance_reply("i had a really bad day today", ["yeah"], guidance))
    self.assertFalse(
      is_low_substance_reply("i had a really bad day today", ["oh no", "kya hua"], guidance)
    )

  def test_status_messages_need_direct_answer(self) -> None:
    guidance = build_conversation_guidance("what are you doing")

    self.assertIn("status", guidance.tags)
    self.assertTrue(is_low_substance_reply("what are you doing", ["Haa"], guidance))
    self.assertFalse(
      is_low_substance_reply(
        "what are you doing", ["nothing much", "tum kya kar rahe ho?"], guidance
      )
    )

  def test_planning_fallback_is_actionable(self) -> None:
    guidance = build_conversation_guidance("should we go library tomorrow?")

    self.assertEqual(
      build_last_resort_reply("should we go library tomorrow?", guidance),
      ["haa lets go", "kab chale?"],
    )

  def test_last_resort_plan_stays_generic_for_unknown_status_facts(self) -> None:
    guidance = build_conversation_guidance("where are you rn")

    plan = build_last_resort_plan("where are you rn", guidance)

    self.assertEqual(plan.fact_confidence, "generic")
    self.assertEqual(plan.reply_act, "answer_and_bounce_back")
    self.assertFalse(
      any(word in plan.direct_answer.lower() for word in ("hostel", "library", "classroom"))
    )

  def test_session_summary_keeps_older_live_context(self) -> None:
    session_history = [
      MessageTurn("Alex", ["i had a bad day"], "t1"),
      MessageTurn("Sam", ["oh no", "kya hua"], "t2"),
      MessageTurn("Alex", ["attendance bhi kharab ho gayi"], "t3"),
      MessageTurn("Sam", ["thats so annoying"], "t4"),
      MessageTurn("Alex", ["abhi bhi bura lag raha hai"], "t5"),
      MessageTurn("Sam", ["im here"], "t6"),
      MessageTurn("Alex", ["thanks"], "t7"),
      MessageTurn("Sam", ["hmm"], "t8"),
      MessageTurn("Alex", ["you there?"], "t9"),
      MessageTurn("Sam", ["haa"], "t10"),
    ]

    summary = build_session_summary(
      session_history, "Alex", "Sam", recent_turn_window=4, max_lines=3
    )

    self.assertTrue(summary.lines)
    self.assertTrue(any("attendance" in line for line in summary.lines))

  def test_generation_quality_flags_archive_copy(self) -> None:
    turns = [
      MessageTurn("Alex", ["i feel awful today"], "t1"),
      MessageTurn("Sam", ["Oh no", "Are you okay"], "t2"),
      MessageTurn("Alex", ["library tomorrow?"], "t3"),
      MessageTurn("Sam", ["Lets go", "Wait for me"], "t4"),
    ]
    examples, _ = build_reply_examples(turns, "Sam")
    style_bundle = build_style_example_bundle(examples[:1])
    guidance = build_conversation_guidance("i feel awful today")
    plan = build_last_resort_plan("i feel awful today", guidance)

    report = evaluate_generation_quality(
      "i feel awful today",
      ["Oh no", "Are you okay"],
      guidance,
      plan,
      style_bundle,
    )

    self.assertFalse(report.passes)
    self.assertIn("archive_derivative", report.reasons)

  def test_dialogue_state_keeps_recent_topic_and_open_thread(self) -> None:
    session_history = [
      MessageTurn("Alex", ["heyyy"], "t1"),
      MessageTurn("Sam", ["heyy", "kya kar raha hai"], "t2"),
      MessageTurn("Alex", ["coding kar raha hu assignment ke liye"], "t3"),
    ]

    state = build_dialogue_state(session_history, "Alex", "Sam")

    self.assertTrue(any(topic in {"coding", "assignment"} for topic in state.active_topics))
    self.assertTrue(state.unresolved_questions)
    self.assertIn("coding kar raha hu assignment ke liye", state.unresolved_questions[0])

  def test_local_reply_eval_prefers_contextual_reply_over_generic_one(self) -> None:
    archive_turns = [
      MessageTurn("Alex", ["i feel awful today"], "t1"),
      MessageTurn("Sam", ["Oh no", "Are you okay"], "t2"),
      MessageTurn("Alex", ["library tomorrow?"], "t3"),
      MessageTurn("Sam", ["Lets go", "Wait for me"], "t4"),
    ]
    examples, _ = build_reply_examples(archive_turns, "Sam")
    style_bundle = build_style_example_bundle(examples)
    live_history = [
      MessageTurn("Alex", ["heyyy"], "live-1"),
      MessageTurn("Sam", ["heyy", "kya kar raha hai"], "live-2"),
      MessageTurn("Alex", ["coding kar raha hu assignment ke liye"], "live-3"),
    ]
    guidance = build_conversation_guidance("coding kar raha hu assignment ke liye")
    state = build_dialogue_state(live_history, "Alex", "Sam")

    coherent = score_local_reply_candidate(
      "coding kar raha hu assignment ke liye",
      ["ohh nice", "kya bana raha hai?"],
      guidance,
      style_bundle,
      recent_turns=live_history[:-1],
      dialogue_state=state,
    )
    generic = score_local_reply_candidate(
      "coding kar raha hu assignment ke liye",
      ["hmm"],
      guidance,
      style_bundle,
      recent_turns=live_history[:-1],
      dialogue_state=state,
    )

    self.assertGreater(coherent.overall, generic.overall)
    self.assertGreater(coherent.continuity, generic.continuity)
    self.assertGreater(coherent.directness, generic.directness)

  def test_small_local_eval_set_scores_natural_candidates_higher(self) -> None:
    archive_turns = [
      MessageTurn("Alex", ["i feel awful today"], "t1"),
      MessageTurn("Sam", ["Oh no", "Are you okay"], "t2"),
      MessageTurn("Alex", ["library tomorrow?"], "t3"),
      MessageTurn("Sam", ["Lets go", "Wait for me"], "t4"),
      MessageTurn("Alex", ["what are you doing"], "t5"),
      MessageTurn("Sam", ["nothing much", "tum kya kar rahe ho?"], "t6"),
    ]
    examples, _ = build_reply_examples(archive_turns, "Sam")
    style_bundle = build_style_example_bundle(examples)
    eval_cases = [
      {
        "question": "i had a really bad day today",
        "good": ["oh no", "kya hua"],
        "bad": ["hmm"],
      },
      {
        "question": "what are you doing",
        "good": ["nothing much", "tum kya kar rahe ho?"],
        "bad": ["haa"],
      },
      {
        "question": "library tomorrow?",
        "good": ["haa lets go", "kab chale?"],
        "bad": ["okay"],
      },
    ]

    for case in eval_cases:
      guidance = build_conversation_guidance(case["question"])
      better = score_local_reply_candidate(
        case["question"],
        case["good"],
        guidance,
        style_bundle,
      )
      worse = score_local_reply_candidate(
        case["question"],
        case["bad"],
        guidance,
        style_bundle,
      )
      self.assertGreater(
        better.overall,
        worse.overall,
        msg=f"Expected better reply to win for {case['question']!r}",
      )


if __name__ == "__main__":
  unittest.main()
