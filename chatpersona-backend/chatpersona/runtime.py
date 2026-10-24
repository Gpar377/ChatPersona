from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rich.console import Console, Group
from rich.text import Text

from chatpersona.corpus import (
  ChatAssets,
  DialogueSnippet,
  MessageTurn,
  StyleExampleBundle,
  build_style_example_bundle,
  clip_text,
  compare_reply_shape,
  format_recent_chat,
  tokenize_text,
)
from chatpersona.profiles import ChatProfile
from chatpersona.ui import (
  PERSONA_COLOR,
  SUCCESS,
  USER_COLOR,
  WARNING,
  message_panel,
  notice_line,
  print_screen,
  screen_title,
  section_rule,
  shortcut_footer,
  status_line,
)

PRIMARY_TIMEOUT_SECONDS = 60.0
FALLBACK_TIMEOUT_SECONDS = 180.0
MODEL_KEEP_ALIVE = "15m"

SUPPORTIVE_MARKERS = (
  "awful",
  "bad day",
  "cooked",
  "depressed",
  "drained",
  "hate",
  "hurt",
  "lonely",
  "not okay",
  "rough",
  "sad",
  "stressed",
  "terrible",
  "tired",
  "upset",
)
GREETING_MARKERS = (
  "good morning",
  "good night",
  "hello",
  "hey",
  "heyy",
  "heyyy",
  "hi",
  "hii",
  "yo",
  "yoo",
)
STATUS_MARKERS = (
  "how are you",
  "how was your day",
  "what are you doing",
  "what are u doing",
  "what's up",
  "whatre you doing",
  "where are you",
  "wyd",
)
AFFECTION_MARKERS = (
  "cuddle",
  "hug",
  "kiss",
  "love you",
  "miss you",
)
PLANNING_MARKERS = (
  "come",
  "go ",
  "kab",
  "library",
  "meet",
  "milte",
  "should we",
  "time",
  "tomorrow",
  "when",
)
ACADEMIC_MARKERS = (
  "assignment",
  "attendance",
  "bio",
  "chem",
  "class",
  "coding",
  "exam",
  "graph",
  "lab",
  "maths",
  "pdf",
  "req",
  "table",
)
SUPPORTIVE_REPLY_MARKERS = (
  "are you okay",
  "its okay",
  "it's okay",
  "kya hua",
  "oh no",
  "take care",
  "tell me",
  "what happened",
)
AFFECTION_REPLY_MARKERS = (
  "awh",
  "aww",
  "cute",
  "love you",
  "love you too",
  "miss you",
  "miss you too",
  "🥹",
)
PLANNING_REPLY_MARKERS = (
  "come",
  "kab",
  "lets go",
  "let's go",
  "library",
  "milte",
  "time",
  "tomorrow",
  "wait",
  "when",
)
STATUS_REPLY_MARKERS = (
  "at ",
  "bas",
  "home",
  "i am",
  "i'm",
  "im ",
  "in ",
  "nothing much",
  "party",
  "room",
  "sleeping",
  "studying",
)
FOLLOW_UP_MARKERS = (
  "bol na",
  "kab",
  "kya hua",
  "kya kar",
  "tum kya",
  "tu bata",
  "what happened",
)
GENERIC_ARCHIVE_PHRASES = frozenset(
  {
    "awh",
    "aww",
    "bol na",
    "dekhte hai",
    "haan",
    "haan send kar",
    "haa",
    "haa lets go",
    "haa sure",
    "heyy",
    "hmm",
    "kab",
    "kab chale",
    "kya hua",
    "lets go",
    "love you too",
    "miss you too",
    "nothing much",
    "oh no",
    "take care",
    "tell me",
    "tum kya kar rahe ho",
    "wait for me",
  }
)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
GenerationStatusCallback = Callable[[str], None]
IGNORED_TOPIC_TOKENS = frozenset(
  {
    "about",
    "again",
    "also",
    "coding",
    "going",
    "gonna",
    "just",
    "like",
    "much",
    "nothing",
    "really",
    "still",
    "sure",
    "that",
    "thing",
    "today",
    "tomorrow",
    "wanna",
  }
)
COMMITMENT_MARKERS = (
  "call",
  "come",
  "later",
  "let's",
  "lets",
  "meet",
  "send",
  "tomorrow",
  "wait",
  "will",
)


@dataclass(slots=True)
class ConversationGuidance:
  tags: list[str]
  goals: list[str]
  minimum_bursts: int
  minimum_length: int
  prefer_question: bool

  def to_prompt_text(self) -> str:
    lines = [
      "Detected conversation mode: " + ", ".join(self.tags) + ".",
      *[f"- {goal}" for goal in self.goals],
    ]
    if self.prefer_question:
      lines.append(
        "- Keep the conversation moving with one small follow-up question if it feels natural."
      )
    return "\n".join(lines)


@dataclass(slots=True)
class SessionSummary:
  lines: list[str]

  def to_prompt_text(self) -> str:
    if not self.lines:
      return "No earlier live session summary yet."
    return "\n".join(f"- {line}" for line in self.lines)


@dataclass(slots=True)
class DialogueState:
  active_topics: list[str]
  unresolved_questions: list[str]
  emotional_mode: str
  recent_commitments: list[str]
  last_assistant_stance: str
  continuity_notes: list[str]
  last_partner_message: str
  last_assistant_message: str

  def to_prompt_text(self) -> str:
    lines = [
      "- Emotional mode: " + self.emotional_mode + ".",
      "- Active topics: " + (", ".join(self.active_topics) if self.active_topics else "none") + ".",
      "- Open questions: "
      + (", ".join(self.unresolved_questions) if self.unresolved_questions else "none")
      + ".",
      "- Recent commitments/plans: "
      + (", ".join(self.recent_commitments) if self.recent_commitments else "none")
      + ".",
      "- Last assistant stance: " + self.last_assistant_stance + ".",
    ]
    if self.last_partner_message:
      lines.append("- Last partner message: " + self.last_partner_message + ".")
    if self.last_assistant_message:
      lines.append("- Last assistant message: " + self.last_assistant_message + ".")
    lines.extend(f"- {note}" for note in self.continuity_notes)
    return "\n".join(lines)


@dataclass(slots=True)
class SimilarSituationBundle:
  snippets: list[DialogueSnippet]

  def to_prompt_text(self, friend_name: str, partner_name: str) -> str:
    if not self.snippets:
      return "No similar archive situations were available."

    lines: list[str] = []
    for index, snippet in enumerate(self.snippets, start=1):
      lines.append(f"Situation {index}:")
      if snippet.lead_context:
        setup = " / ".join(clip_text(turn.flattened(), 70) for turn in snippet.lead_context[-2:])
        lines.append(f"- Setup: {setup}")
      lines.append(f"- Partner move: {clip_text(snippet.partner_turn.flattened(), 90)}")
      lines.append(f"- Persona behavior: {snippet.interaction_pattern}")
      lines.append(
        f"- Reply shape: {snippet.reply_burst_count} short bubble(s), about {snippet.reply_character_count} characters."
      )
      if snippet.trailing_context:
        lines.append("- Follow-through: " + clip_text(snippet.trailing_context[0].flattened(), 80))
    return "\n".join(lines)


@dataclass(slots=True)
class SemanticReplyPlan:
  intent: str
  reply_act: str
  direct_answer: str
  emotional_acknowledgment: str
  supporting_thought: str
  follow_up: str
  initiative_level: str
  fact_confidence: str

  def to_prompt_text(self) -> str:
    return "\n".join(
      [
        f"- Intent: {self.intent}",
        f"- Reply act: {self.reply_act}",
        f"- Emotional acknowledgment: {self.emotional_acknowledgment or '-'}",
        f"- Direct answer: {self.direct_answer or '-'}",
        f"- Supporting thought: {self.supporting_thought or '-'}",
        f"- Follow-up: {self.follow_up or '-'}",
        f"- Initiative level: {self.initiative_level}",
        f"- Fact confidence: {self.fact_confidence}",
      ]
    )


@dataclass(slots=True)
class GenerationQualityReport:
  passes: bool
  reasons: list[str]


@dataclass(slots=True)
class LocalReplyEvalScore:
  continuity: float
  directness: float
  tone_fidelity: float
  archive_copy_rate: float
  perceived_naturalness: float

  @property
  def overall(self) -> float:
    return round(
      (
        self.continuity
        + self.directness
        + self.tone_fidelity
        + (1.0 - self.archive_copy_rate)
        + self.perceived_naturalness
      )
      / 5,
      2,
    )


PLANNER_SYSTEM_TEMPLATE = """
You are planning the meaning of the next text reply in a private chat.

Do not write in the person's texting style yet.
Decide what the reply should actually say as a real conversational partner.
Use the live conversation and dialogue state as the main source of truth.
Use similar archive situations only as behavioral context, not as a source of specific facts to repeat.
If a concrete fact is unknown, stay plausible and generic instead of inventing specifics.
Prefer continuity: answer what is open, stay on the active topic, and only then add one relevant thought or one small follow-up.

Return one strict JSON object only with these keys:
- intent
- reply_act
- direct_answer
- emotional_acknowledgment
- supporting_thought
- follow_up
- initiative_level
- fact_confidence

Rules:
- direct_answer should not be blank when the other person asked something direct.
- emotional_acknowledgment should be blank unless the moment needs emotional reaction.
- supporting_thought should add one useful, warm, or playful beat.
- follow_up should be blank or one short natural follow-up.
- fact_confidence must be one of: live_session, generic, unsure.
- Never mention archives, AI, assistants, or roleplay.
""".strip()

PLANNER_USER_TEMPLATE = """
Recent live conversation:
{recent_chat}

Earlier live session summary:
{session_summary}

Dialogue state:
{dialogue_state}

Similar archive situations:
{similar_situations}

Latest message from {partner_name}:
{question}

Conversation goals:
{conversation_guidance}

Default behavior:
- initiative_level: moderately_proactive
- factual_behavior: plausible_generic
""".strip()

STYLE_SYSTEM_TEMPLATE = """
You are {friend_name} in a real private text conversation.

Stay fully in character.
Do not mention being an AI, assistant, model, or roleplay.
Do not narrate, explain, summarize, or add speaker labels.
Do not sound polished, therapeutic, or generic.

Use the semantic reply plan for meaning.
Use the dialogue state for continuity and the style profile/tone evidence only for texting rhythm, bubble count, wording texture, and code-switching.
Do not copy archive lines verbatim unless the phrase is extremely generic.
Do not introduce new concrete facts that are not already in the semantic plan or the live session.

Output only the final reply text, with one bubble per line.
""".strip()

STYLE_USER_TEMPLATE = """
Recent live conversation:
{recent_chat}

Dialogue state:
{dialogue_state}

Latest message from {partner_name}:
{question}

Semantic reply plan:
{semantic_reply_plan}

Style profile:
{style_profile}

Tone evidence:
{style_examples}

Reply rules:
- Prefer {target_burst_count} short text bubbles, each on its own line.
- Keep the total reply around {target_length} characters unless the moment clearly needs a little more.
- Keep it text-like and natural.
- Answer the person first, then keep the conversation moving.
- Use Hindi/English mix only when it fits.
{strict_mode}
""".strip()

META_PATTERN = re.compile(
  r"\b(?:as [\w\s]+|assistant|ai|reply:|response:|text:|i would reply|here'?s a reply|bot:)\b",
  re.IGNORECASE,
)
SPEAKER_PREFIX_PATTERN = re.compile(
  r"^(?:assistant|reply|response|text|bot)\s*[:\-]\s*", re.IGNORECASE
)


def split_user_messages(text: str) -> list[str]:
  pieces = [piece.strip() for piece in text.split("|")]
  return [piece for piece in pieces if piece] or [text.strip()]


def text_contains_any(text: str, phrases: tuple[str, ...]) -> bool:
  return any(phrase in text for phrase in phrases)


def detect_conversation_tags(question: str) -> list[str]:
  lowered = question.lower().strip()
  tags: list[str] = []

  if text_contains_any(lowered, GREETING_MARKERS) or lowered in {
    "h",
    "hey",
    "heyy",
    "hi",
    "hii",
    "yo",
    "yoo",
  }:
    tags.append("greeting")
  if text_contains_any(lowered, STATUS_MARKERS):
    tags.append("status")
  if text_contains_any(lowered, AFFECTION_MARKERS):
    tags.append("affection")
  if text_contains_any(lowered, SUPPORTIVE_MARKERS):
    tags.append("supportive")
  if text_contains_any(lowered, PLANNING_MARKERS):
    tags.append("planning")
  if text_contains_any(lowered, ACADEMIC_MARKERS):
    tags.append("academic")
  if "?" in lowered and "status" not in tags and "planning" not in tags:
    tags.append("question")
  if not tags:
    tags.append("casual")

  return tags


def build_conversation_guidance(question: str) -> ConversationGuidance:
  tags = detect_conversation_tags(question)
  goals: list[str] = []
  minimum_bursts = 1
  minimum_length = 20 if len(question.strip()) <= 12 else 30
  prefer_question = False

  if "greeting" in tags:
    goals.extend(
      [
        "Respond warmly instead of abruptly or dismissively.",
        "Reopen the chat with a small bounce-back question or inviting line.",
      ]
    )
    minimum_bursts = max(minimum_bursts, 2)
    minimum_length = max(minimum_length, 22)
    prefer_question = True

  if "status" in tags:
    goals.extend(
      [
        "Answer what you are doing or how you are first.",
        "Add one small relevant detail or bounce the question back.",
      ]
    )
    minimum_bursts = max(minimum_bursts, 2)
    minimum_length = max(minimum_length, 32)
    prefer_question = True

  if "supportive" in tags:
    goals.extend(
      [
        "Acknowledge the feeling first before changing topic.",
        "Sound caring and ask what happened or what went wrong.",
      ]
    )
    minimum_bursts = max(minimum_bursts, 2)
    minimum_length = max(minimum_length, 44)
    prefer_question = True

  if "affection" in tags:
    goals.extend(
      [
        "Reciprocate the affection naturally.",
        "Add one soft reassuring or affectionate follow-up line.",
      ]
    )
    minimum_bursts = max(minimum_bursts, 2)
    minimum_length = max(minimum_length, 30)

  if "planning" in tags:
    goals.extend(
      [
        "Be concrete about the plan instead of vague.",
        "Confirm or ask the next useful detail like time or place.",
      ]
    )
    minimum_bursts = max(minimum_bursts, 2)
    minimum_length = max(minimum_length, 34)
    prefer_question = True

  if "academic" in tags:
    goals.extend(
      [
        "Be practically helpful and directly relevant.",
        "Answer the task/question before adding anything extra.",
      ]
    )
    minimum_bursts = max(minimum_bursts, 2 if len(question) > 25 else 1)
    minimum_length = max(minimum_length, 26)

  if "question" in tags and "status" not in tags:
    goals.append("Answer the direct question before anything else.")
    minimum_length = max(minimum_length, 24)

  if len(question) > 90 and "supportive" not in tags:
    goals.append("React to the main point of the message, not just one tiny fragment.")
    minimum_bursts = max(minimum_bursts, 2)
    minimum_length = max(minimum_length, 48)

  deduped_goals = list(dict.fromkeys(goals)) or [
    "React to what they actually said and keep the conversation moving naturally.",
  ]

  return ConversationGuidance(
    tags=tags,
    goals=deduped_goals,
    minimum_bursts=minimum_bursts,
    minimum_length=minimum_length,
    prefer_question=prefer_question,
  )


def choose_target_burst_count(
  question: str, retrieved_examples, guidance: ConversationGuidance
) -> int:
  average_burst = round(
    sum(example.reply_burst_count for example in retrieved_examples)
    / max(len(retrieved_examples), 1)
  )
  if len(question) <= 20:
    return max(1, guidance.minimum_bursts)
  if "?" in question or len(question) >= 80:
    return max(guidance.minimum_bursts, min(3, average_burst + 1))
  return max(guidance.minimum_bursts, min(3, average_burst or 2))


def choose_target_length(question: str, retrieved_examples, guidance: ConversationGuidance) -> int:
  if not retrieved_examples:
    return max(80, guidance.minimum_length)
  average_length = round(
    sum(example.reply_character_count for example in retrieved_examples) / len(retrieved_examples)
  )
  if len(question) > 120:
    average_length += 30
  return max(guidance.minimum_length, min(180, average_length))


def clean_generated_reply(text: str, target_burst_count: int, target_length: int) -> list[str]:
  compact = text.replace("\r", "").strip().strip("\"'")
  if not compact:
    return []

  lines: list[str] = []
  for raw_line in compact.splitlines():
    line = raw_line.strip().strip("\"'")
    line = re.sub(r"^\d+[.)]\s*", "", line)
    line = SPEAKER_PREFIX_PATTERN.sub("", line).strip()
    if not line:
      continue
    if META_PATTERN.search(line) and len(line.split()) > 4:
      continue
    lines.extend(piece.strip() for piece in line.split("|") if piece.strip())

  if not lines:
    stripped = SPEAKER_PREFIX_PATTERN.sub("", compact).strip()
    if stripped and not META_PATTERN.search(stripped):
      lines = [stripped]

  cleaned: list[str] = []
  total_chars = 0
  max_total_chars = max(target_length * 2, 220)
  for line in lines[: max(target_burst_count + 1, 3)]:
    short_line = " ".join(line.split()).strip()
    if not short_line:
      continue
    if len(short_line) > 120:
      trimmed = short_line[:120].rsplit(" ", 1)[0].strip()
      short_line = trimmed or short_line[:120]
    if cleaned and total_chars + len(short_line) > max_total_chars:
      break
    cleaned.append(short_line)
    total_chars += len(short_line)

  return cleaned[: max(target_burst_count, 1)]


def needs_retry(lines: list[str], target_burst_count: int, target_length: int) -> bool:
  joined = " ".join(lines)
  return (
    not lines
    or len(lines) > max(target_burst_count, 3)
    or len(joined) > max(target_length * 2, 220)
    or META_PATTERN.search(joined) is not None
  )


def is_low_substance_reply(
  question: str,
  lines: list[str],
  guidance: ConversationGuidance,
) -> bool:
  joined = " ".join(lines).lower().strip()
  has_follow_up_signal = "?" in joined or text_contains_any(joined, FOLLOW_UP_MARKERS)
  has_engaged_phrase = text_contains_any(
    joined,
    SUPPORTIVE_REPLY_MARKERS
    + AFFECTION_REPLY_MARKERS
    + PLANNING_REPLY_MARKERS
    + STATUS_REPLY_MARKERS,
  )
  if not joined:
    return True
  if len(joined) < max(8, guidance.minimum_length // 3) and not (
    has_follow_up_signal or has_engaged_phrase
  ):
    return True
  if guidance.minimum_bursts >= 2 and len(lines) < 2 and len(joined) < guidance.minimum_length:
    return True
  if (
    guidance.prefer_question and not has_follow_up_signal and len(joined) < guidance.minimum_length
  ):
    return True
  if "supportive" in guidance.tags and not (
    text_contains_any(joined, SUPPORTIVE_REPLY_MARKERS) or has_follow_up_signal
  ):
    return True
  if "affection" in guidance.tags and not text_contains_any(joined, AFFECTION_REPLY_MARKERS):
    return True
  if "planning" in guidance.tags and not (
    text_contains_any(joined, PLANNING_REPLY_MARKERS) or has_follow_up_signal
  ):
    return True
  if "status" in guidance.tags and not (
    text_contains_any(joined, STATUS_REPLY_MARKERS) or has_follow_up_signal
  ):
    return True
  if ("status" in guidance.tags or "planning" in guidance.tags) and " / " in joined:
    return True
  if "greeting" in guidance.tags and len(joined) < 10:
    return True
  if len(question) > 40 and len(joined) < 16 and not (has_follow_up_signal or has_engaged_phrase):
    return True
  return False


def is_timeout_error(error: Exception) -> bool:
  lowered = str(error).lower()
  return "timed out" in lowered or "timeout" in lowered


def build_planner_prompt() -> ChatPromptTemplate:
  return ChatPromptTemplate.from_messages(
    [
      ("system", PLANNER_SYSTEM_TEMPLATE),
      ("human", PLANNER_USER_TEMPLATE),
    ]
  )


def build_style_prompt() -> ChatPromptTemplate:
  return ChatPromptTemplate.from_messages(
    [
      ("system", STYLE_SYSTEM_TEMPLATE),
      ("human", STYLE_USER_TEMPLATE),
    ]
  )


def build_prompt() -> ChatPromptTemplate:
  return build_style_prompt()


def build_model(model_name: str, timeout_seconds: float, num_predict: int) -> ChatOllama:
  return ChatOllama(
    model=model_name,
    temperature=0.9,
    num_predict=num_predict,
    keep_alive=MODEL_KEEP_ALIVE,
    sync_client_kwargs={"timeout": timeout_seconds},
  )


def compact_style_profile(style_profile_text: str) -> str:
  lines = [line.strip() for line in style_profile_text.splitlines() if line.strip()]
  return "\n".join(lines[:4])


def build_session_summary(
  session_history: list[MessageTurn],
  partner_name: str,
  friend_name: str,
  recent_turn_window: int = 8,
  max_lines: int = 4,
) -> SessionSummary:
  older_turns = (
    session_history[:-recent_turn_window] if len(session_history) > recent_turn_window else []
  )
  if not older_turns:
    return SessionSummary([])

  indexed_turns = list(enumerate(older_turns))
  selected_pairs = [pair for pair in indexed_turns if pair[1].speaker == partner_name][-max_lines:]
  if len(selected_pairs) < max_lines:
    remaining = max_lines - len(selected_pairs)
    selected_indices = {index for index, _turn in selected_pairs}
    filler_pairs = [pair for pair in indexed_turns if pair[0] not in selected_indices][-remaining:]
    selected_pairs.extend(filler_pairs)

  selected_turns = [turn for _index, turn in sorted(selected_pairs, key=lambda pair: pair[0])]
  lines = []
  for turn in selected_turns:
    label = partner_name if turn.speaker == partner_name else friend_name
    lines.append(f"{label} said: {turn.flattened()}")
  return SessionSummary(lines)


def clamp_score(value: float) -> float:
  return round(max(0.0, min(1.0, value)), 2)


def extract_active_topics(
  session_history: list[MessageTurn],
  partner_name: str,
  friend_name: str,
  max_topics: int = 4,
) -> list[str]:
  recent_turns = session_history[-6:]
  if not recent_turns:
    return []

  ignored_tokens = (
    set(IGNORED_TOPIC_TOKENS) | set(tokenize_text(partner_name)) | set(tokenize_text(friend_name))
  )
  topic_counts: Counter[str] = Counter()

  for index, turn in enumerate(recent_turns, start=1):
    weight = 1.0
    if turn.speaker == partner_name:
      weight += 0.4
    if index == len(recent_turns):
      weight += 0.5

    for token in tokenize_text(turn.flattened()):
      if token in ignored_tokens:
        continue
      topic_counts[token] += weight

  return [token for token, _count in topic_counts.most_common(max_topics)]


def message_needs_response(text: str) -> bool:
  lowered = text.lower().strip()
  tags = set(detect_conversation_tags(lowered))
  return (
    "?" in lowered
    or bool(tags & {"status", "planning", "supportive", "academic", "question"})
    or any(
      phrase in lowered
      for phrase in (
        "tell me",
        "you there",
        "bol na",
        "kya hua",
        "kya kar",
        "tu bata",
      )
    )
  )


def extract_unresolved_questions(
  session_history: list[MessageTurn],
  partner_name: str,
  friend_name: str,
  limit: int = 2,
) -> list[str]:
  unresolved: list[str] = []
  if not session_history:
    return unresolved

  for turn in reversed(session_history[-6:]):
    if turn.speaker == friend_name and unresolved:
      break
    if turn.speaker != partner_name:
      continue
    flattened = turn.flattened()
    if not flattened or not message_needs_response(flattened):
      continue
    clipped = clip_text(flattened, 80)
    if clipped not in unresolved:
      unresolved.append(clipped)
    if len(unresolved) >= limit:
      break

  unresolved.reverse()
  return unresolved


def infer_dialogue_emotional_mode(session_history: list[MessageTurn]) -> str:
  if not session_history:
    return "casual and open"

  tag_counts: Counter[str] = Counter()
  for turn in session_history[-6:]:
    tag_counts.update(detect_conversation_tags(turn.flattened()))

  if tag_counts["supportive"]:
    return "tender and attentive"
  if tag_counts["planning"] and tag_counts["status"]:
    return "casual but coordinating"
  if tag_counts["planning"]:
    return "practical and forward-moving"
  if tag_counts["affection"]:
    return "soft and affectionate"
  if tag_counts["academic"]:
    return "practical and task-focused"
  if tag_counts["greeting"] and len(session_history) <= 3:
    return "warm opening energy"
  if tag_counts["status"]:
    return "light everyday catch-up"
  if tag_counts["question"]:
    return "engaged and responsive"
  return "easygoing and conversational"


def extract_recent_commitments(session_history: list[MessageTurn], limit: int = 3) -> list[str]:
  commitments: list[str] = []
  for turn in reversed(session_history[-8:]):
    flattened = " ".join(turn.flattened().split())
    lowered = flattened.lower()
    if not flattened or not any(marker in lowered for marker in COMMITMENT_MARKERS):
      continue
    clipped = clip_text(flattened, 72)
    if clipped not in commitments:
      commitments.append(clipped)
    if len(commitments) >= limit:
      break
  commitments.reverse()
  return commitments


def infer_last_assistant_stance(
  session_history: list[MessageTurn],
  friend_name: str,
) -> str:
  last_assistant_turn = next(
    (turn for turn in reversed(session_history) if turn.speaker == friend_name),
    None,
  )
  if last_assistant_turn is None:
    return "no assistant reply yet"

  reply_text = last_assistant_turn.flattened().lower()
  tags = set(detect_conversation_tags(reply_text))
  if "supportive" in tags:
    return "warm and checking in"
  if "planning" in tags:
    return "coordinating the next step"
  if "affection" in tags:
    return "soft and affectionate"
  if "academic" in tags:
    return "practical and helpful"
  if "status" in tags:
    return "answering directly and bouncing the chat back"
  if "?" in reply_text:
    return "keeping the chat moving with a light follow-up"
  if len(last_assistant_turn.messages) >= 2:
    return "short, bursty, and conversational"
  return "compact and natural"


def build_dialogue_state(
  session_history: list[MessageTurn],
  partner_name: str,
  friend_name: str,
) -> DialogueState:
  last_partner_turn = next(
    (turn for turn in reversed(session_history) if turn.speaker == partner_name),
    None,
  )
  last_assistant_turn = next(
    (turn for turn in reversed(session_history) if turn.speaker == friend_name),
    None,
  )
  active_topics = extract_active_topics(session_history, partner_name, friend_name)
  unresolved_questions = extract_unresolved_questions(session_history, partner_name, friend_name)
  emotional_mode = infer_dialogue_emotional_mode(session_history)
  recent_commitments = extract_recent_commitments(session_history)
  last_assistant_stance = infer_last_assistant_stance(session_history, friend_name)

  continuity_notes: list[str] = []
  if unresolved_questions:
    continuity_notes.append("Answer the open question before changing topic.")
  if active_topics:
    continuity_notes.append("Stay anchored to the active topic unless the partner shifts it.")
  if recent_commitments:
    continuity_notes.append("Keep recent plans and commitments consistent.")
  if emotional_mode == "tender and attentive":
    continuity_notes.append("Match the feeling first, then move the chat forward gently.")
  if last_assistant_turn is not None and last_partner_turn is not None:
    continuity_notes.append("Do not repeat the previous assistant line; add the next beat.")

  return DialogueState(
    active_topics=active_topics,
    unresolved_questions=unresolved_questions,
    emotional_mode=emotional_mode,
    recent_commitments=recent_commitments,
    last_assistant_stance=last_assistant_stance,
    continuity_notes=list(dict.fromkeys(continuity_notes)),
    last_partner_message=clip_text(last_partner_turn.flattened(), 90) if last_partner_turn else "",
    last_assistant_message=(
      clip_text(last_assistant_turn.flattened(), 90) if last_assistant_turn else ""
    ),
  )


def build_similar_situation_bundle(snippets: list[DialogueSnippet]) -> SimilarSituationBundle:
  return SimilarSituationBundle(snippets=snippets[:4])


def build_planner_prompt_variables(
  *,
  question: str,
  recent_turns: list[MessageTurn],
  session_summary: SessionSummary,
  dialogue_state: DialogueState,
  similar_situations: SimilarSituationBundle,
  conversation_guidance: ConversationGuidance,
  partner_name: str,
  friend_name: str,
  compact: bool,
) -> dict[str, object]:
  capped_recent_turns = recent_turns[-(3 if compact else 6) :]
  snippet_bundle = SimilarSituationBundle(similar_situations.snippets[: (2 if compact else 4)])
  return {
    "recent_chat": format_recent_chat(capped_recent_turns, friend_name, partner_name),
    "session_summary": session_summary.to_prompt_text(),
    "dialogue_state": dialogue_state.to_prompt_text(),
    "similar_situations": snippet_bundle.to_prompt_text(friend_name, partner_name),
    "conversation_guidance": conversation_guidance.to_prompt_text(),
    "partner_name": partner_name,
    "question": question,
  }


def build_style_prompt_variables(
  *,
  assets: ChatAssets,
  question: str,
  recent_turns: list[MessageTurn],
  dialogue_state: DialogueState,
  semantic_reply_plan: SemanticReplyPlan,
  style_examples: StyleExampleBundle,
  target_burst_count: int,
  target_length: int,
  strict_mode: str,
  compact: bool,
) -> dict[str, object]:
  capped_recent_turns = recent_turns[-(3 if compact else 6) :]
  style_profile = assets.style_profile.to_prompt_text()
  if compact:
    style_profile = compact_style_profile(style_profile)

  return {
    "friend_name": assets.friend_name,
    "style_profile": style_profile,
    "recent_chat": format_recent_chat(capped_recent_turns, assets.friend_name, assets.partner_name),
    "dialogue_state": dialogue_state.to_prompt_text(),
    "style_examples": style_examples.to_prompt_text(),
    "semantic_reply_plan": semantic_reply_plan.to_prompt_text(),
    "target_burst_count": target_burst_count,
    "target_length": target_length,
    "partner_name": assets.partner_name,
    "question": question,
    "strict_mode": strict_mode,
  }


def invoke_prompt(
  prompt: ChatPromptTemplate, model: ChatOllama, variables: dict[str, object]
) -> str:
  prompt_value = prompt.invoke(variables)
  response = model.invoke(prompt_value)
  if hasattr(response, "content"):
    return str(response.content)
  return str(response)


def extract_json_object(text: str) -> dict[str, object] | None:
  compact = text.strip()
  candidates = [compact]
  match = JSON_OBJECT_PATTERN.search(compact)
  if match:
    candidates.append(match.group(0))

  for candidate in candidates:
    try:
      payload = json.loads(candidate)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      return payload
  return None


def clean_plan_value(value: object) -> str:
  if value is None:
    return ""
  cleaned = " ".join(str(value).replace("\r", " ").split()).strip()
  if cleaned.lower() in {"none", "null", "n/a", "-"}:
    return ""
  return cleaned


def parse_semantic_reply_plan(
  raw_text: str,
  question: str,
  guidance: ConversationGuidance,
) -> SemanticReplyPlan | None:
  payload = extract_json_object(raw_text)
  if payload is None:
    return None

  direct_answer = clean_plan_value(payload.get("direct_answer"))
  emotional_acknowledgment = clean_plan_value(payload.get("emotional_acknowledgment"))
  supporting_thought = clean_plan_value(payload.get("supporting_thought"))
  follow_up = clean_plan_value(payload.get("follow_up"))

  if not direct_answer and not emotional_acknowledgment and not supporting_thought:
    return None

  intent = clean_plan_value(payload.get("intent")) or (
    guidance.tags[0] if guidance.tags else "casual"
  )
  reply_act = clean_plan_value(payload.get("reply_act")) or "answer_and_continue"
  initiative_level = clean_plan_value(payload.get("initiative_level")) or "moderately_proactive"
  fact_confidence = clean_plan_value(payload.get("fact_confidence")) or "generic"

  return SemanticReplyPlan(
    intent=intent,
    reply_act=reply_act,
    direct_answer=direct_answer,
    emotional_acknowledgment=emotional_acknowledgment,
    supporting_thought=supporting_thought,
    follow_up=follow_up,
    initiative_level=initiative_level,
    fact_confidence=fact_confidence,
  )


def build_last_resort_plan(
  question: str,
  guidance: ConversationGuidance,
  dialogue_state: DialogueState | None = None,
) -> SemanticReplyPlan:
  lowered = question.lower()
  active_topics = set(dialogue_state.active_topics if dialogue_state is not None else [])
  work_topics = {"coding", "code", "assignment", "class", "exam"}

  if "supportive" in guidance.tags:
    return SemanticReplyPlan(
      intent="supportive",
      reply_act="comfort_and_check_in",
      direct_answer="that sounds really rough",
      emotional_acknowledgment="oh no",
      supporting_thought="im here",
      follow_up="kya hua",
      initiative_level="moderately_proactive",
      fact_confidence="generic",
    )
  if "affection" in guidance.tags:
    return SemanticReplyPlan(
      intent="affection",
      reply_act="reciprocate_and_reassure",
      direct_answer="miss you too",
      emotional_acknowledgment="awh",
      supporting_thought="thats cute",
      follow_up="",
      initiative_level="moderately_proactive",
      fact_confidence="generic",
    )
  if "planning" in guidance.tags:
    if "library" in lowered:
      return SemanticReplyPlan(
        intent="planning",
        reply_act="confirm_and_next_step",
        direct_answer="haa lets go",
        emotional_acknowledgment="",
        supporting_thought="",
        follow_up="kab chale?",
        initiative_level="moderately_proactive",
        fact_confidence="generic",
      )
    return SemanticReplyPlan(
      intent="planning",
      reply_act="confirm_and_next_step",
      direct_answer="haa sure",
      emotional_acknowledgment="",
      supporting_thought="",
      follow_up="kab?",
      initiative_level="moderately_proactive",
      fact_confidence="generic",
    )
  if "status" in guidance.tags:
    direct_answer = "nothing much"
    if work_topics & active_topics or any(token in lowered for token in work_topics):
      direct_answer = "nothing much just doing some work"
    return SemanticReplyPlan(
      intent="status",
      reply_act="answer_and_bounce_back",
      direct_answer=direct_answer,
      emotional_acknowledgment="",
      supporting_thought="just here rn",
      follow_up="tum kya kar rahe ho?",
      initiative_level="moderately_proactive",
      fact_confidence="generic",
    )
  if "academic" in guidance.tags:
    return SemanticReplyPlan(
      intent="academic",
      reply_act="practical_help",
      direct_answer="haan send kar",
      emotional_acknowledgment="",
      supporting_thought="dekhte hai",
      follow_up="",
      initiative_level="moderately_proactive",
      fact_confidence="generic",
    )
  if "greeting" in guidance.tags:
    return SemanticReplyPlan(
      intent="greeting",
      reply_act="warm_open_and_bounce_back",
      direct_answer="heyy",
      emotional_acknowledgment="",
      supporting_thought="",
      follow_up="kya kar raha hai",
      initiative_level="moderately_proactive",
      fact_confidence="generic",
    )
  if "question" in guidance.tags:
    return SemanticReplyPlan(
      intent="question",
      reply_act="answer_and_invite",
      direct_answer="haan",
      emotional_acknowledgment="",
      supporting_thought="",
      follow_up="bol na",
      initiative_level="moderately_proactive",
      fact_confidence="generic",
    )
  return SemanticReplyPlan(
    intent="casual",
    reply_act="react_and_continue",
    direct_answer=(
      "ohh nice"
      if work_topics & active_topics or any(token in lowered for token in work_topics)
      else "hmm"
    ),
    emotional_acknowledgment="",
    supporting_thought=(
      "sounds productive"
      if work_topics & active_topics or any(token in lowered for token in work_topics)
      else "im listening"
    ),
    follow_up=(
      "what are you working on?"
      if "coding" in active_topics or "code" in active_topics
      else "tell me"
    ),
    initiative_level="moderately_proactive",
    fact_confidence="generic",
  )


def build_last_resort_reply(
  question: str,
  guidance: ConversationGuidance,
  dialogue_state: DialogueState | None = None,
) -> list[str]:
  fallback_plan = build_last_resort_plan(question, guidance, dialogue_state)
  pieces = [
    fallback_plan.emotional_acknowledgment,
    fallback_plan.direct_answer,
    fallback_plan.supporting_thought,
    fallback_plan.follow_up,
  ]
  desired_bursts = max(guidance.minimum_bursts, 1)
  raw_lines = [piece for piece in pieces if piece][: max(desired_bursts + 1, 3)]
  return clean_generated_reply(
    "\n".join(raw_lines), desired_bursts, max(guidance.minimum_length, 50)
  )


def normalize_reply_phrase(text: str) -> str:
  normalized = re.sub(r"[^a-z0-9?']+", " ", text.lower()).strip()
  return " ".join(normalized.split())


def content_overlap_ratio(source_text: str, rendered_text: str) -> float:
  source_tokens = tokenize_text(source_text)
  rendered_tokens = tokenize_text(rendered_text)
  if not source_tokens or not rendered_tokens:
    return 0.0
  return len(source_tokens & rendered_tokens) / len(source_tokens)


def is_generic_archive_phrase(text: str) -> bool:
  normalized = normalize_reply_phrase(text)
  return normalized in GENERIC_ARCHIVE_PHRASES or len(normalized.split()) <= 1


def is_archive_derivative_reply(lines: list[str], style_examples: StyleExampleBundle) -> bool:
  archived_lines = [normalize_reply_phrase(line) for line in style_examples.archived_reply_lines()]
  archived_lines = [line for line in archived_lines if line and not is_generic_archive_phrase(line)]
  if not archived_lines:
    return False

  for line in lines:
    normalized_line = normalize_reply_phrase(line)
    if not normalized_line or is_generic_archive_phrase(normalized_line):
      continue
    line_tokens = tokenize_text(normalized_line)
    for archived_line in archived_lines:
      if normalized_line == archived_line:
        return True
      archived_tokens = tokenize_text(archived_line)
      if len(line_tokens) >= 4 and archived_tokens:
        overlap = len(line_tokens & archived_tokens) / max(len(line_tokens), 1)
        if overlap >= 0.8:
          return True
  return False


def archive_copy_rate(lines: list[str], style_examples: StyleExampleBundle) -> float:
  archived_lines = [normalize_reply_phrase(line) for line in style_examples.archived_reply_lines()]
  archived_lines = [line for line in archived_lines if line and not is_generic_archive_phrase(line)]
  if not archived_lines:
    return 0.0

  max_overlap = 0.0
  for line in lines:
    normalized_line = normalize_reply_phrase(line)
    if not normalized_line or is_generic_archive_phrase(normalized_line):
      continue
    line_tokens = tokenize_text(normalized_line)
    for archived_line in archived_lines:
      archived_tokens = tokenize_text(archived_line)
      if normalized_line == archived_line:
        return 1.0
      if line_tokens and archived_tokens:
        max_overlap = max(
          max_overlap,
          len(line_tokens & archived_tokens) / max(len(line_tokens), 1),
        )
  return round(max_overlap, 2)


def reply_engages_recent_context(
  question: str,
  lines: list[str],
  guidance: ConversationGuidance,
  recent_turns: list[MessageTurn],
  dialogue_state: DialogueState,
) -> bool:
  reply_text = " ".join(lines).strip().lower()
  if not reply_text:
    return False

  reply_tokens = tokenize_text(reply_text)
  context_tokens = (
    tokenize_text(question)
    | tokenize_text(dialogue_state.last_partner_message)
    | tokenize_text(" ".join(dialogue_state.active_topics))
    | tokenize_text(" ".join(dialogue_state.unresolved_questions))
    | tokenize_text(" ".join(turn.flattened() for turn in recent_turns[-3:]))
  )
  reply_has_question = "?" in reply_text

  if len(reply_tokens & context_tokens) >= 1:
    return True
  if "supportive" in guidance.tags and (
    text_contains_any(reply_text, SUPPORTIVE_REPLY_MARKERS) or reply_has_question
  ):
    return True
  if "planning" in guidance.tags and (
    text_contains_any(reply_text, PLANNING_REPLY_MARKERS) or reply_has_question
  ):
    return True
  if "status" in guidance.tags and (
    text_contains_any(reply_text, STATUS_REPLY_MARKERS) or reply_has_question
  ):
    return True
  if "academic" in guidance.tags and text_contains_any(
    reply_text,
    ("send", "class", "graph", "pdf", "table", "code", "coding", "assignment"),
  ):
    return True
  if "greeting" in guidance.tags:
    return len(reply_text) >= 6
  if "question" in guidance.tags and not reply_has_question:
    return bool(reply_tokens)
  return not context_tokens and len(reply_tokens) >= 2


def is_overly_generic_reply(
  question: str,
  lines: list[str],
  guidance: ConversationGuidance,
  recent_turns: list[MessageTurn],
  dialogue_state: DialogueState,
) -> bool:
  reply_text = " ".join(lines).strip().lower()
  if not reply_text:
    return True
  if normalize_reply_phrase(reply_text) in GENERIC_ARCHIVE_PHRASES and len(question.strip()) > 8:
    return True
  if len(tokenize_text(reply_text)) <= 1 and guidance.minimum_length >= 24:
    return True
  if dialogue_state.unresolved_questions and not reply_engages_recent_context(
    question, lines, guidance, recent_turns, dialogue_state
  ):
    return True
  return False


def score_tone_fidelity(lines: list[str], style_examples: StyleExampleBundle) -> float:
  if not lines:
    return 0.0
  if not style_examples.examples:
    total_chars = sum(len(line) for line in lines)
    compact_bonus = 0.15 if len(lines) <= 3 and total_chars <= 180 else 0.0
    return clamp_score(0.7 + compact_bonus)

  shape_report = compare_reply_shape(lines, style_examples.examples[0].reply_turn)
  burst_penalty = min(float(shape_report["burst_delta"]) / 3, 1.0)
  character_penalty = min(float(shape_report["character_delta"]) / 120, 1.0)
  code_penalty = min(float(shape_report["code_switch_delta"]), 1.0)
  return clamp_score(
    1.0 - ((burst_penalty * 0.45) + (character_penalty * 0.35) + (code_penalty * 0.2))
  )


def score_directness(
  question: str,
  lines: list[str],
  guidance: ConversationGuidance,
  recent_turns: list[MessageTurn],
  dialogue_state: DialogueState,
) -> float:
  if not lines:
    return 0.0
  if reply_engages_recent_context(question, lines, guidance, recent_turns, dialogue_state):
    return 1.0
  if "?" in question and "?" in " ".join(lines):
    return 0.55
  if guidance.tags == ["casual"] and len(" ".join(lines)) >= 12:
    return 0.7
  return 0.3


def score_perceived_naturalness(
  question: str,
  lines: list[str],
  guidance: ConversationGuidance,
  recent_turns: list[MessageTurn],
  dialogue_state: DialogueState,
) -> float:
  joined = " ".join(lines)
  score = 1.0
  if META_PATTERN.search(joined):
    score -= 0.5
  if len(lines) > 3:
    score -= 0.2
  if len(joined) > 220:
    score -= 0.2
  if is_overly_generic_reply(question, lines, guidance, recent_turns, dialogue_state):
    score -= 0.25
  return clamp_score(score)


def score_local_reply_candidate(
  question: str,
  lines: list[str],
  guidance: ConversationGuidance,
  style_examples: StyleExampleBundle,
  recent_turns: list[MessageTurn] | None = None,
  dialogue_state: DialogueState | None = None,
) -> LocalReplyEvalScore:
  recent_turns = list(recent_turns or [])
  if dialogue_state is None:
    dialogue_state = DialogueState(
      active_topics=[],
      unresolved_questions=[clip_text(question, 80)] if message_needs_response(question) else [],
      emotional_mode="easygoing and conversational",
      recent_commitments=[],
      last_assistant_stance="no assistant reply yet",
      continuity_notes=[],
      last_partner_message=clip_text(question, 90),
      last_assistant_message="",
    )

  continuity = (
    1.0
    if reply_engages_recent_context(question, lines, guidance, recent_turns, dialogue_state)
    else 0.35
  )
  return LocalReplyEvalScore(
    continuity=continuity,
    directness=score_directness(question, lines, guidance, recent_turns, dialogue_state),
    tone_fidelity=score_tone_fidelity(lines, style_examples),
    archive_copy_rate=archive_copy_rate(lines, style_examples),
    perceived_naturalness=score_perceived_naturalness(
      question, lines, guidance, recent_turns, dialogue_state
    ),
  )


def reply_addresses_plan(
  lines: list[str],
  plan: SemanticReplyPlan,
  guidance: ConversationGuidance,
) -> bool:
  rendered_text = " ".join(lines)
  required_segments = []
  if "supportive" in guidance.tags and plan.emotional_acknowledgment:
    required_segments.append(plan.emotional_acknowledgment)
  if plan.direct_answer:
    required_segments.append(plan.direct_answer)

  if (
    "status" in guidance.tags
    or "planning" in guidance.tags
    or "academic" in guidance.tags
    or "question" in guidance.tags
  ) and not plan.direct_answer:
    return False

  meaningful_segments = [
    segment
    for segment in required_segments
    if len(tokenize_text(segment)) >= 2 and not is_generic_archive_phrase(segment)
  ]
  if not meaningful_segments:
    return True

  return any(
    content_overlap_ratio(segment, rendered_text) >= 0.25 for segment in meaningful_segments
  )


def evaluate_generation_quality(
  question: str,
  lines: list[str],
  guidance: ConversationGuidance,
  plan: SemanticReplyPlan,
  style_examples: StyleExampleBundle,
  recent_turns: list[MessageTurn] | None = None,
  dialogue_state: DialogueState | None = None,
) -> GenerationQualityReport:
  recent_turns = list(recent_turns or [])
  if dialogue_state is None:
    dialogue_state = DialogueState(
      active_topics=[],
      unresolved_questions=[clip_text(question, 80)] if message_needs_response(question) else [],
      emotional_mode="easygoing and conversational",
      recent_commitments=[],
      last_assistant_stance="no assistant reply yet",
      continuity_notes=[],
      last_partner_message=clip_text(question, 90),
      last_assistant_message="",
    )
  reasons: list[str] = []
  if is_low_substance_reply(question, lines, guidance):
    reasons.append("low_substance")
  if is_archive_derivative_reply(lines, style_examples):
    reasons.append("archive_derivative")
  if not reply_addresses_plan(lines, plan, guidance):
    reasons.append("does_not_answer")
  if not reply_engages_recent_context(question, lines, guidance, recent_turns, dialogue_state):
    reasons.append("misses_recent_context")
  if is_overly_generic_reply(question, lines, guidance, recent_turns, dialogue_state):
    reasons.append("too_generic")
  return GenerationQualityReport(not reasons, reasons)


def generate_semantic_reply_plan(
  prompt: ChatPromptTemplate,
  primary_model: ChatOllama,
  fallback_model: ChatOllama,
  assets: ChatAssets,
  question: str,
  recent_turns: list[MessageTurn],
  session_summary: SessionSummary,
  dialogue_state: DialogueState,
  similar_situations: SimilarSituationBundle,
  guidance: ConversationGuidance,
  status_callback: GenerationStatusCallback | None = None,
) -> SemanticReplyPlan:
  attempt_configs = [
    {"model": primary_model, "compact": False},
    {"model": fallback_model, "compact": True},
  ]

  for attempt_index, config in enumerate(attempt_configs):
    try:
      if status_callback is not None:
        status_callback("Planning reply")
      raw_text = invoke_prompt(
        prompt,
        config["model"],
        build_planner_prompt_variables(
          question=question,
          recent_turns=recent_turns,
          session_summary=session_summary,
          dialogue_state=dialogue_state,
          similar_situations=similar_situations,
          conversation_guidance=guidance,
          partner_name=assets.partner_name,
          friend_name=assets.friend_name,
          compact=config["compact"],
        ),
      )
      parsed_plan = parse_semantic_reply_plan(raw_text, question, guidance)
      if parsed_plan is not None:
        return parsed_plan
    except Exception as error:
      if (
        status_callback is not None
        and is_timeout_error(error)
        and attempt_index < len(attempt_configs) - 1
      ):
        status_callback("Retrying planner with a smaller prompt")
      if not is_timeout_error(error) or attempt_index == len(attempt_configs) - 1:
        break

  return build_last_resort_plan(question, guidance, dialogue_state)


def render_reply_from_plan(
  prompt: ChatPromptTemplate,
  primary_model: ChatOllama,
  fallback_model: ChatOllama,
  assets: ChatAssets,
  question: str,
  recent_turns: list[MessageTurn],
  dialogue_state: DialogueState,
  plan: SemanticReplyPlan,
  style_examples: StyleExampleBundle,
  target_burst_count: int,
  target_length: int,
  guidance: ConversationGuidance,
  status_callback: GenerationStatusCallback | None = None,
) -> list[str]:
  attempt_configs = [
    {"model": primary_model, "compact": False, "strict_mode": ""},
    {
      "model": primary_model,
      "compact": False,
      "strict_mode": (
        "Do not reuse archive lines or highly specific archive facts. "
        "Keep the meaning from the semantic reply plan, but phrase it more freshly and naturally."
      ),
    },
    {
      "model": fallback_model,
      "compact": True,
      "strict_mode": (
        "Keep the reply compact, text-like, and natural. "
        "Do not copy archive lines. Use the plan for meaning and the archive only for style."
      ),
    },
  ]
  last_timeout: Exception | None = None

  for attempt_index, config in enumerate(attempt_configs):
    try:
      if status_callback is not None:
        status_callback(
          "Collecting style evidence" if attempt_index == 0 else "Retrying style render"
        )
      raw_reply = invoke_prompt(
        prompt,
        config["model"],
        build_style_prompt_variables(
          assets=assets,
          question=question,
          recent_turns=recent_turns,
          dialogue_state=dialogue_state,
          semantic_reply_plan=plan,
          style_examples=style_examples,
          target_burst_count=max(1, min(target_burst_count, 2 if config["compact"] else 3)),
          target_length=min(target_length, 130 if config["compact"] else target_length),
          strict_mode=config["strict_mode"],
          compact=config["compact"],
        ),
      )
      cleaned_reply = clean_generated_reply(raw_reply, target_burst_count, target_length)
      report = evaluate_generation_quality(
        question,
        cleaned_reply,
        guidance,
        plan,
        style_examples,
        recent_turns=recent_turns,
        dialogue_state=dialogue_state,
      )
      if report.passes:
        return cleaned_reply
      if status_callback is not None:
        status_callback("Retrying style render")
    except Exception as error:
      if is_timeout_error(error):
        last_timeout = error
        continue
      break

  if last_timeout is not None:
    raise RuntimeError(
      f"timed out talking to Ollama model `{primary_model.model}`. "
      + f"Try running `ollama run {primary_model.model}` once to warm it up, then retry."
    )
  return build_last_resort_reply(question, guidance, dialogue_state)


def generate_reply(
  planner_prompt: ChatPromptTemplate,
  style_prompt: ChatPromptTemplate,
  primary_model: ChatOllama,
  fallback_model: ChatOllama,
  assets: ChatAssets,
  question: str,
  session_history: list[MessageTurn],
  status_callback: GenerationStatusCallback | None = None,
) -> list[str]:
  recent_turns = session_history[:-1][-6:]
  guidance = build_conversation_guidance(question)
  dialogue_state = build_dialogue_state(
    session_history,
    partner_name=assets.partner_name,
    friend_name=assets.friend_name,
  )
  similar_situations = build_similar_situation_bundle(
    assets.retriever.find_dialogue_snippets(question, recent_turns, k=4)
  )
  style_examples = build_style_example_bundle(
    assets.retriever.find_style_examples(question, recent_turns, k=4),
    assets.style_profile,
  )
  target_burst_count = choose_target_burst_count(question, style_examples.examples, guidance)
  target_length = choose_target_length(question, style_examples.examples, guidance)
  session_summary = build_session_summary(
    session_history[:-1], assets.partner_name, assets.friend_name
  )

  semantic_plan = generate_semantic_reply_plan(
    planner_prompt,
    primary_model,
    fallback_model,
    assets,
    question,
    recent_turns,
    session_summary,
    dialogue_state,
    similar_situations,
    guidance,
    status_callback=status_callback,
  )
  return render_reply_from_plan(
    style_prompt,
    primary_model,
    fallback_model,
    assets,
    question,
    recent_turns,
    dialogue_state,
    semantic_plan,
    style_examples,
    target_burst_count,
    target_length,
    guidance,
    status_callback=status_callback,
  )


def format_session_state(session_history: list[MessageTurn]) -> str:
  if not session_history:
    return "Fresh session"
  exchange_count = max(1, len(session_history) // 2)
  return f"{exchange_count} live exchange(s)"


def render_workspace_header(
  profile: ChatProfile, assets: ChatAssets, session_history: list[MessageTurn]
) -> Group:
  return Group(
    screen_title("chatpersona", profile.profile_name),
    status_line(
      "Chat",
      f"{assets.partner_name} -> {assets.friend_name} · {profile.chat_model} · {assets.retrieval_mode}",
    ),
    status_line("Session", format_session_state(session_history)),
  )


def render_turn_lines(turn: MessageTurn, assets: ChatAssets) -> list[Text]:
  is_user = turn.speaker == assets.partner_name
  title = "You" if is_user else assets.friend_name
  tone = USER_COLOR if is_user else PERSONA_COLOR
  lines: list[Text] = []

  for index, message in enumerate(turn.messages):
    text = Text()
    if index == 0:
      text.append(title, style=f"bold {tone}")
      text.append("  ")
    else:
      text.append(" " * (len(title) + 2))
    text.append(message, style="white")
    lines.append(text)

  return lines


def render_transcript(session_history: list[MessageTurn], assets: ChatAssets) -> Group:
  if not session_history:
    return Group(notice_line("Start the conversation."))

  recent_turns = session_history[-10:]
  entries: list[Text] = []
  if len(session_history) > len(recent_turns):
    entries.append(
      Text("Earlier live turns are hidden so the workspace stays compact.", style="grey70")
    )
    entries.append(Text(""))
  for turn in recent_turns:
    entries.extend(render_turn_lines(turn, assets))
    entries.append(Text(""))
  return Group(*entries[:-1] if entries else entries)


def render_chat_help_section() -> Group:
  return Group(
    section_rule("Help"),
    notice_line("/help  show chat shortcuts"),
    notice_line("/profile  show active profile details"),
    notice_line("/clear  reset the live transcript and session memory"),
    notice_line("/retry  regenerate the last assistant reply"),
    notice_line("/quit  exit the chat"),
  )


def render_profile_section(profile: ChatProfile, assets: ChatAssets) -> Group:
  return Group(
    section_rule("Profile"),
    status_line("Profile", profile.profile_name),
    status_line("Persona", profile.persona_name),
    status_line("Partner", assets.partner_name),
    status_line("Chat model", profile.chat_model),
    status_line("Preferred retrieval", profile.retrieval_mode),
    status_line("Last build", profile.last_built_retrieval_mode),
    status_line("Embedding model", profile.embedding_model or "-"),
  )


def render_workspace(
  console: Console,
  profile: ChatProfile,
  assets: ChatAssets,
  session_history: list[MessageTurn],
  focus_panel=None,
) -> None:
  renderables = [
    render_workspace_header(profile, assets, session_history),
    section_rule("Transcript"),
    render_transcript(session_history, assets),
  ]
  if focus_panel is not None:
    renderables.append(Text(""))
    renderables.append(focus_panel)
  renderables.append(Text(""))
  renderables.append(
    shortcut_footer("/help  /profile  /clear  /retry  /quit   | for multi-bubble input")
  )
  print_screen(console, *renderables, clear=True)


def interactive_chat_session(profile: ChatProfile, assets: ChatAssets, console: Console) -> None:
  planner_prompt = build_planner_prompt()
  style_prompt = build_style_prompt()
  primary_model = build_model(profile.chat_model, PRIMARY_TIMEOUT_SECONDS, num_predict=120)
  fallback_model = build_model(profile.chat_model, FALLBACK_TIMEOUT_SECONDS, num_predict=80)
  session_history: list[MessageTurn] = []
  focus_panel = notice_line("Ready.")

  while True:
    render_workspace(console, profile, assets, session_history, focus_panel)
    focus_panel = None

    try:
      user_input = console.input(
        f"[bold {USER_COLOR}]{assets.partner_name}[/bold {USER_COLOR}] > "
      ).strip()
    except (EOFError, KeyboardInterrupt):
      break

    if user_input in {"/q", "q", "quit", "exit", "/quit"}:
      break
    if not user_input:
      continue
    if user_input == "/help":
      focus_panel = render_chat_help_section()
      continue
    if user_input == "/profile":
      focus_panel = render_profile_section(profile, assets)
      continue
    if user_input == "/clear":
      session_history.clear()
      focus_panel = message_panel(
        "Session cleared", "The live transcript and session memory were reset.", SUCCESS
      )
      continue
    if user_input == "/retry":
      removed_assistant_turn = None
      if session_history and session_history[-1].speaker == assets.friend_name:
        removed_assistant_turn = session_history.pop()

      if not session_history or session_history[-1].speaker != assets.partner_name:
        if removed_assistant_turn is not None:
          session_history.append(removed_assistant_turn)
        focus_panel = notice_line("Nothing to retry yet.", WARNING)
        continue

      question = "\n".join(session_history[-1].messages)
      try:
        with console.status("Planning reply", spinner="dots") as status:
          reply_lines = generate_reply(
            planner_prompt,
            style_prompt,
            primary_model,
            fallback_model,
            assets,
            question,
            session_history,
            status_callback=status.update,
          )
      except Exception as error:
        if removed_assistant_turn is not None:
          session_history.append(removed_assistant_turn)
        focus_panel = message_panel(
          "Could not generate a reply",
          f"{error}\nLikely cause: the local model is cold or unavailable.\nNext: warm Ollama and try `/retry` again.",
          WARNING,
        )
        continue

      session_history.append(MessageTurn(assets.friend_name, reply_lines, "live"))
      focus_panel = notice_line("Reply regenerated.", SUCCESS)
      continue

    user_messages = split_user_messages(user_input)
    question = "\n".join(user_messages)
    session_history.append(MessageTurn(assets.partner_name, user_messages, "live"))

    try:
      with console.status("Planning reply", spinner="dots") as status:
        reply_lines = generate_reply(
          planner_prompt,
          style_prompt,
          primary_model,
          fallback_model,
          assets,
          question,
          session_history,
          status_callback=status.update,
        )
    except Exception as error:
      session_history.pop()
      focus_panel = message_panel(
        "Could not generate a reply",
        f"{error}\nLikely cause: the local model is cold or unavailable.\nNext: warm Ollama or run `chatpersona doctor`.",
        WARNING,
      )
      continue

    session_history.append(MessageTurn(assets.friend_name, reply_lines, "live"))
