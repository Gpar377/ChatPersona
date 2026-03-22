from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re
import shutil
from typing import Callable, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

DEFAULT_PERSONA_NAME = "Persona"
DEFAULT_PARTNER_NAME = "You"
DEFAULT_ARTIFACTS_DIR = Path(".chatpersona_artifacts")
DEFAULT_DB_DIRNAME = "chroma_db"
DATASET_EXPORT_FILENAME = "reply_examples.jsonl"
MANIFEST_FILENAME = "index_manifest.json"
PARSER_VERSION = "reply-conditioned-cli-v1"
EMBEDDING_TIMEOUT_SECONDS = 120.0
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_WARMUP_TEXT = "warm up embedding model"

WHATSAPP_CLASSIC_LINE_PATTERN = re.compile(
  r"^(?P<date>\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})),\s"
  r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?)\s-\s"
  r"(?P<sender>[^:]+?):\s?(?P<text>.*)$"
)
WHATSAPP_BRACKETED_LINE_PATTERN = re.compile(
  r"^\[(?P<date>\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})),\s"
  r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?)\]\s"
  r"(?P<sender>[^:]+?):\s?(?P<text>.*)$"
)
WHATSAPP_INVISIBLE_TRANSLATION = str.maketrans(
  {
    "\ufeff": None,
    "\u200e": None,
    "\u200f": None,
    "\u202a": None,
    "\u202b": None,
    "\u202c": None,
    "\u202d": None,
    "\u202e": None,
    "\u2066": None,
    "\u2067": None,
    "\u2068": None,
    "\u2069": None,
    "\u202f": " ",
    "\xa0": " ",
  }
)
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[a-zA-Z']+")

SKIP_TEXT_FRAGMENTS = (
  "messages and calls are end-to-end encrypted",
  "<media omitted>",
  "missed voice call",
  "missed video call",
  "video call",
  "audio omitted",
  "voice call",
  "document omitted",
  "deleted this message",
  "gif omitted",
  "image omitted",
  "null",
  "sticker omitted",
)

STOPWORDS = {
  "about",
  "after",
  "again",
  "all",
  "also",
  "and",
  "are",
  "because",
  "been",
  "before",
  "being",
  "but",
  "dont",
  "from",
  "have",
  "just",
  "like",
  "make",
  "more",
  "much",
  "only",
  "really",
  "same",
  "that",
  "than",
  "then",
  "they",
  "this",
  "what",
  "when",
  "where",
  "which",
  "will",
  "with",
  "would",
  "your",
  "you",
  "youre",
  "yeah",
  "okay",
  "okayy",
}

HINDI_MARKERS = {
  "acha",
  "accha",
  "arey",
  "arrey",
  "bas",
  "bcz",
  "bhi",
  "haan",
  "haa",
  "hai",
  "ho",
  "hua",
  "kar",
  "kya",
  "kyu",
  "main",
  "mat",
  "nahi",
  "pls",
  "toh",
  "tum",
  "yaar",
  "yrr",
}

INTENT_KEYWORDS = {
  "supportive": (
    "anxious",
    "awful",
    "bad",
    "bad day",
    "bored",
    "break",
    "cooked",
    "cry",
    "depressed",
    "drained",
    "feel low",
    "feel off",
    "fuck",
    "health",
    "hate",
    "hurt",
    "lonely",
    "loser",
    "not okay",
    "overwhelmed",
    "rough",
    "sad",
    "shit",
    "stressed",
    "terrible",
    "tired",
    "upset",
    "vent",
    "worst",
    "worse",
  ),
  "planning": (
    "campus",
    "come",
    "going",
    "kab",
    "library",
    "let's",
    "meet",
    "milte",
    "orientation",
    "register",
    "room",
    "should we",
    "stairs",
    "time",
    "tomorrow",
    "wait",
    "where",
    "wanna go",
    "lets go",
    "when",
  ),
  "greeting": (
    "gm",
    "gn",
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
  ),
  "status": (
    "how are you",
    "how was your day",
    "what are you doing",
    "what are u doing",
    "what's up",
    "whatre you doing",
    "where are you",
    "wyd",
  ),
  "affection": (
    "cuddle",
    "cute",
    "hug",
    "kiss",
    "love you",
    "miss you",
    "romantic",
  ),
  "playful": (
    "cute",
    "hehe",
    "heheh",
    "hot",
    "love",
    "miss you",
    "romantic",
    "sensual",
    "sus",
    "bad girl",
    "bad boy",
    "byee",
    "hahaha",
    "lol",
  ),
  "check_in": (
    "everything okay",
    "what happened",
    "wyd",
    "where are you",
    "you okay",
    "you there",
  ),
  "academic": (
    "academic",
    "assignment",
    "attendance",
    "bio",
    "chem",
    "class",
    "coding",
    "graph",
    "lab",
    "maths",
    "pdf",
    "req",
    "sem",
    "table",
  ),
}


@dataclass(slots=True)
class ChatMessage:
  timestamp: str
  sender: str
  text: str


@dataclass(slots=True)
class MessageTurn:
  speaker: str
  messages: list[str]
  started_at: str

  def flattened(self, separator: str = " | ") -> str:
    return separator.join(message for message in self.messages if message.strip())


@dataclass(slots=True)
class ReplyExample:
  example_id: str
  context: list[MessageTurn]
  user_turn: MessageTurn
  reply_turn: MessageTurn
  tags: list[str]
  search_text: str
  search_tokens: frozenset[str]

  @property
  def reply_burst_count(self) -> int:
    return len(self.reply_turn.messages)

  @property
  def reply_character_count(self) -> int:
    return sum(len(message) for message in self.reply_turn.messages)

  @property
  def dialogue_signature(self) -> str:
    return build_dialogue_signature(self.user_turn, self.reply_turn)

  def prompt_block(self, friend_name: str, partner_name: str) -> str:
    parts: list[str] = []
    if self.context:
      context_lines = [format_turn(turn, friend_name, partner_name) for turn in self.context[-2:]]
      parts.append("Recent context:\n" + "\n".join(context_lines))
    parts.append(f"{partner_name}: " + clip_text(self.user_turn.flattened()))
    parts.append(f"{friend_name}: " + clip_text(" / ".join(self.reply_turn.messages)))
    parts.append("Tags: " + ", ".join(self.tags))
    return "\n".join(parts)


@dataclass(slots=True)
class DialogueSnippet:
  snippet_id: str
  lead_context: list[MessageTurn]
  partner_turn: MessageTurn
  reply_turn: MessageTurn
  trailing_context: list[MessageTurn]
  tags: list[str]
  interaction_pattern: str
  search_text: str
  search_tokens: frozenset[str]

  @property
  def reply_burst_count(self) -> int:
    return len(self.reply_turn.messages)

  @property
  def reply_character_count(self) -> int:
    return sum(len(message) for message in self.reply_turn.messages)

  @property
  def dialogue_signature(self) -> str:
    return build_dialogue_signature(self.partner_turn, self.reply_turn)

  def prompt_block(self, friend_name: str, partner_name: str) -> str:
    parts: list[str] = []
    if self.lead_context:
      setup_lines = [
        format_turn(turn, friend_name, partner_name) for turn in self.lead_context[-2:]
      ]
      parts.append("Recent setup:\n" + "\n".join(setup_lines))
    parts.append("Latest partner turn: " + clip_text(self.partner_turn.flattened()))
    parts.append("Behavior pattern: " + self.interaction_pattern)
    parts.append(
      f"Reply shape: {self.reply_burst_count} short bubble(s), about {self.reply_character_count} characters."
    )
    if self.trailing_context:
      follow_through = format_turn(self.trailing_context[0], friend_name, partner_name)
      parts.append("What happened next: " + clip_text(follow_through))
    parts.append("Tags: " + ", ".join(self.tags))
    return "\n".join(parts)


@dataclass(slots=True)
class StyleProfile:
  avg_reply_burst: float
  median_message_length: int
  common_tokens: list[str]
  common_phrases: list[str]
  tone_notes: list[str]

  def to_prompt_text(self) -> str:
    lines = [
      f"- Average reply burst: {self.avg_reply_burst:.1f} short messages.",
      f"- Typical message length: about {self.median_message_length} characters.",
      "- Common tokens: " + ", ".join(self.common_tokens) + ".",
      "- Frequent short phrases: " + ", ".join(self.common_phrases) + ".",
    ]
    lines.extend(f"- {note}" for note in self.tone_notes)
    return "\n".join(lines)


@dataclass(slots=True)
class StyleExampleBundle:
  examples: list[ReplyExample]
  shape_notes: list[str]
  phrase_notes: list[str]
  rhythm_examples: list[str]

  def to_prompt_text(self) -> str:
    if not self.examples:
      return "No style examples were available."

    lines = [*self.shape_notes]
    if self.phrase_notes:
      lines.append("- Common short phrase shapes: " + ", ".join(self.phrase_notes) + ".")
    if self.rhythm_examples:
      lines.append("- Style evidence:")
      lines.extend(f"  {example}" for example in self.rhythm_examples)
    return "\n".join(lines)

  def archived_reply_lines(self) -> list[str]:
    return [
      message.strip()
      for example in self.examples
      for message in example.reply_turn.messages
      if message.strip()
    ]


@dataclass(slots=True)
class ChatAssets:
  friend_name: str
  partner_name: str
  style_profile: StyleProfile
  reply_examples: list[ReplyExample]
  dialogue_snippets: list[DialogueSnippet]
  retriever: "ConversationRetriever"
  retrieval_mode: str
  artifact_dir: Path


@dataclass(slots=True)
class BuildProgressUpdate:
  stage: str
  message: str
  completed: int | None = None
  total: int | None = None


BuildProgressCallback = Callable[[BuildProgressUpdate], None]


def emit_progress(
  progress_callback: BuildProgressCallback | None,
  stage: str,
  message: str,
  *,
  completed: int | None = None,
  total: int | None = None,
) -> None:
  if progress_callback is None:
    return
  progress_callback(
    BuildProgressUpdate(
      stage=stage,
      message=message,
      completed=completed,
      total=total,
    )
  )


def clip_text(text: str, limit: int = 220) -> str:
  compact = " ".join(text.split())
  if len(compact) <= limit:
    return compact
  clipped = compact[:limit].rsplit(" ", 1)[0].strip()
  return clipped or compact[:limit]


def tokenize_text(text: str) -> frozenset[str]:
  tokens = {
    token
    for token in WORD_PATTERN.findall(text.lower())
    if len(token) > 2 and token not in STOPWORDS
  }
  return frozenset(tokens)


def detect_code_switching(text: str) -> bool:
  lowered = text.lower()
  return any(marker in lowered for marker in HINDI_MARKERS) and bool(WORD_PATTERN.search(lowered))


def infer_intent_tags(text: str) -> list[str]:
  lowered = text.lower()
  tags = [
    tag
    for tag, keywords in INTENT_KEYWORDS.items()
    if any(keyword in lowered for keyword in keywords)
  ]
  return tags or ["casual"]


def text_contains_any(text: str, phrases: Sequence[str]) -> bool:
  return any(phrase in text for phrase in phrases)


def parse_whatsapp_messages(filepath: str | Path) -> list[ChatMessage]:
  messages: list[ChatMessage] = []
  current_message: ChatMessage | None = None

  with Path(filepath).open("r", encoding="utf-8") as handle:
    for raw_line in handle:
      line = normalize_whatsapp_export_line(raw_line.rstrip("\n"))
      match = match_whatsapp_message_line(line)
      if match:
        date = match.group("date")
        time_text = " ".join(match.group("time").split())
        sender = match.group("sender")
        text = match.group("text")
        current_message = ChatMessage(
          timestamp=f"{date} {time_text}",
          sender=sender.strip(),
          text=text.strip(),
        )
        messages.append(current_message)
        continue

      continuation = line.strip()
      if continuation and current_message is not None:
        current_message.text = "\n".join(
          part for part in (current_message.text, continuation) if part
        )

  return messages


def normalize_whatsapp_export_line(line: str) -> str:
  return line.translate(WHATSAPP_INVISIBLE_TRANSLATION)


def match_whatsapp_message_line(line: str) -> re.Match[str] | None:
  return WHATSAPP_CLASSIC_LINE_PATTERN.match(line) or WHATSAPP_BRACKETED_LINE_PATTERN.match(line)


def detect_participants(messages: Sequence[ChatMessage]) -> list[str]:
  counts = Counter(message.sender for message in messages if message.sender.strip())
  return [name for name, _ in counts.most_common()]


def ensure_two_party_chat(messages: Sequence[ChatMessage]) -> list[str]:
  participants = detect_participants(messages)
  if len(participants) < 2:
    raise ValueError("Could not find two participants in this WhatsApp export.")
  if len(participants) > 2:
    raise ValueError("Only 1:1 WhatsApp exports are supported right now.")
  return participants


def count_urls(text: str) -> int:
  return len(URL_PATTERN.findall(text))


def is_broadcast_style_message(text: str) -> bool:
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  numbered_lines = sum(1 for line in lines if re.match(r"^\d+[.)]", line))
  url_count = count_urls(text)
  return (
    url_count >= 2
    or (len(lines) >= 4 and url_count >= 1)
    or numbered_lines >= 2
    or ("group link" in text.lower() and url_count >= 1)
  )


def should_skip_message(text: str) -> bool:
  stripped = text.strip()
  lowered = stripped.lower()
  return (
    not stripped
    or any(fragment in lowered for fragment in SKIP_TEXT_FRAGMENTS)
    or stripped.lower() == "this message was deleted"
    or is_broadcast_style_message(stripped)
  )


def is_pathological_turn(turn: MessageTurn) -> bool:
  if len(turn.messages) < 6:
    return False

  normalized = []
  for message in turn.messages:
    compact = re.sub(r"[^a-z0-9]+", "", message.lower())
    if compact:
      normalized.append(compact)

  return (
    bool(normalized) and all(len(token) <= 2 for token in normalized) and len(set(normalized)) <= 2
  )


def trim_pathological_suffix(turn: MessageTurn) -> MessageTurn | None:
  suffix_tokens: list[str] = []
  suffix_length = 0

  for message in reversed(turn.messages):
    compact = re.sub(r"[^a-z0-9]+", "", message.lower())
    if compact and len(compact) <= 2:
      suffix_tokens.append(compact)
      suffix_length += 1
      continue
    break

  if suffix_length >= 6 and len(set(suffix_tokens)) <= 2:
    kept_messages = turn.messages[:-suffix_length]
  else:
    kept_messages = list(turn.messages)

  if not kept_messages:
    return None

  candidate = MessageTurn(turn.speaker, kept_messages, turn.started_at)
  if is_pathological_turn(candidate):
    return None
  return candidate


def merge_adjacent_turns(turns: Sequence[MessageTurn]) -> list[MessageTurn]:
  merged: list[MessageTurn] = []
  for turn in turns:
    if merged and merged[-1].speaker == turn.speaker:
      merged[-1].messages.extend(turn.messages)
    else:
      merged.append(MessageTurn(turn.speaker, list(turn.messages), turn.started_at))
  return merged


def build_message_turns(messages: Sequence[ChatMessage]) -> list[MessageTurn]:
  turns: list[MessageTurn] = []

  for message in messages:
    if should_skip_message(message.text):
      continue

    if turns and turns[-1].speaker == message.sender:
      turns[-1].messages.append(message.text.strip())
      continue

    turns.append(MessageTurn(message.sender, [message.text.strip()], message.timestamp))

  filtered_turns = []
  for turn in turns:
    trimmed_turn = trim_pathological_suffix(turn)
    if trimmed_turn is not None:
      filtered_turns.append(trimmed_turn)
  return merge_adjacent_turns(filtered_turns)


def copy_turn(turn: MessageTurn) -> MessageTurn:
  return MessageTurn(turn.speaker, list(turn.messages), turn.started_at)


def build_dialogue_signature(partner_turn: MessageTurn, reply_turn: MessageTurn) -> str:
  return (
    f"{partner_turn.started_at}|{partner_turn.speaker}|{reply_turn.started_at}|{reply_turn.speaker}"
  )


def is_overlong_turn(turn: MessageTurn, max_messages: int, max_characters: int) -> bool:
  return len(turn.messages) > max_messages or len(turn.flattened()) > max_characters


def infer_partner_name(turns: Sequence[MessageTurn], friend_name: str) -> str:
  speaker_counts = Counter(turn.speaker for turn in turns if turn.speaker != friend_name)
  if not speaker_counts:
    raise ValueError("Could not infer the other chat participant from the WhatsApp export.")
  if len(speaker_counts) > 1:
    raise ValueError("Only 1:1 WhatsApp exports are supported right now.")
  return speaker_counts.most_common(1)[0][0]


def build_reply_examples(
  turns: Sequence[MessageTurn],
  friend_name: str,
  context_window: int = 2,
) -> tuple[list[ReplyExample], str]:
  partner_name = infer_partner_name(turns, friend_name)
  examples: list[ReplyExample] = []

  for index in range(1, len(turns)):
    reply_turn = turns[index]
    user_turn = turns[index - 1]
    if reply_turn.speaker != friend_name or user_turn.speaker != partner_name:
      continue

    context_start = max(0, index - 1 - context_window)
    context = [copy_turn(turn) for turn in turns[context_start : index - 1]]
    context_text = " ".join(turn.flattened() for turn in context)
    user_text = user_turn.flattened()
    reply_text = reply_turn.flattened()
    tags = sorted(set(infer_intent_tags(" ".join((user_text, reply_text)))))
    search_text = " ".join(part for part in (context_text, user_text, reply_text) if part).strip()

    if not user_text or not reply_text:
      continue
    if is_overlong_turn(user_turn, max_messages=12, max_characters=700):
      continue
    if is_overlong_turn(reply_turn, max_messages=8, max_characters=280):
      continue

    examples.append(
      ReplyExample(
        example_id=f"reply-{len(examples)}",
        context=context,
        user_turn=copy_turn(user_turn),
        reply_turn=copy_turn(reply_turn),
        tags=tags,
        search_text=search_text,
        search_tokens=tokenize_text(search_text),
      )
    )

  return examples, partner_name


def infer_interaction_pattern(
  partner_turn: MessageTurn,
  reply_turn: MessageTurn,
  tags: Sequence[str],
) -> str:
  reply_text = reply_turn.flattened().lower()
  has_follow_up = "?" in reply_text
  multi_bubble = len(reply_turn.messages) >= 2

  if "supportive" in tags:
    pattern = "acknowledges the feeling before checking in"
  elif "planning" in tags:
    pattern = "confirms the plan and moves it one step forward"
  elif "status" in tags:
    pattern = "answers directly, then lightly bounces the chat back"
  elif "greeting" in tags:
    pattern = "returns the greeting warmly and keeps the chat open"
  elif "academic" in tags:
    pattern = "answers practically and stays on task"
  elif "affection" in tags:
    pattern = "reciprocates softly without overexplaining"
  elif "question" in tags:
    pattern = "answers first, then adds one natural continuation beat"
  else:
    pattern = "gives a short natural response that keeps the conversation moving"

  if multi_bubble:
    pattern += " across multiple short bubbles"
  else:
    pattern += " in one compact bubble"
  if has_follow_up:
    pattern += " and ends with a light follow-up question"
  elif len(partner_turn.messages) >= 2:
    pattern += " while staying anchored to the partner's thread"

  return pattern


def build_dialogue_snippets(
  turns: Sequence[MessageTurn],
  friend_name: str,
  context_window: int = 3,
  trailing_window: int = 1,
) -> tuple[list[DialogueSnippet], str]:
  partner_name = infer_partner_name(turns, friend_name)
  snippets: list[DialogueSnippet] = []

  for index in range(1, len(turns)):
    reply_turn = turns[index]
    partner_turn = turns[index - 1]
    if reply_turn.speaker != friend_name or partner_turn.speaker != partner_name:
      continue

    if not partner_turn.flattened() or not reply_turn.flattened():
      continue
    if is_overlong_turn(partner_turn, max_messages=12, max_characters=700):
      continue
    if is_overlong_turn(reply_turn, max_messages=8, max_characters=280):
      continue

    context_start = max(0, index - 1 - context_window)
    lead_context = [copy_turn(turn) for turn in turns[context_start : index - 1]]
    trailing_context = [
      copy_turn(turn)
      for turn in turns[index + 1 : index + 1 + trailing_window]
      if turn.speaker == partner_name
    ]
    snippet_tags = sorted(
      set(
        infer_intent_tags(
          " ".join(
            [
              " ".join(turn.flattened() for turn in lead_context),
              partner_turn.flattened(),
              reply_turn.flattened(),
              " ".join(turn.flattened() for turn in trailing_context),
            ]
          )
        )
      )
    )
    interaction_pattern = infer_interaction_pattern(partner_turn, reply_turn, snippet_tags)
    search_text = " ".join(
      part
      for part in (
        " ".join(turn.flattened() for turn in lead_context),
        partner_turn.flattened(),
        " ".join(turn.flattened() for turn in trailing_context),
        interaction_pattern,
        " ".join(snippet_tags),
      )
      if part
    ).strip()

    snippets.append(
      DialogueSnippet(
        snippet_id=f"snippet-{len(snippets)}",
        lead_context=lead_context,
        partner_turn=copy_turn(partner_turn),
        reply_turn=copy_turn(reply_turn),
        trailing_context=trailing_context,
        tags=snippet_tags,
        interaction_pattern=interaction_pattern,
        search_text=search_text,
        search_tokens=tokenize_text(search_text),
      )
    )

  return snippets, partner_name


def build_style_profile(reply_examples: Sequence[ReplyExample]) -> StyleProfile:
  if not reply_examples:
    raise ValueError("No usable reply examples were found in the chat export.")

  all_reply_messages = [
    message for example in reply_examples for message in example.reply_turn.messages
  ]
  burst_sizes = [example.reply_burst_count for example in reply_examples]
  message_lengths = sorted(len(message) for message in all_reply_messages if message.strip())

  token_counts = Counter()
  for message in all_reply_messages:
    token_counts.update(tokenize_text(message))

  phrase_counts = Counter(
    message.strip()
    for message in all_reply_messages
    if 2 <= len(message.strip()) <= 28 and count_urls(message) == 0
  )

  code_switch_ratio = sum(
    1 for message in all_reply_messages if detect_code_switching(message)
  ) / max(len(all_reply_messages), 1)
  supportive_ratio = sum(1 for example in reply_examples if "supportive" in example.tags) / max(
    len(reply_examples), 1
  )

  tone_notes = [
    "Replies are usually short, bursty, and split across 1-3 bubbles instead of one polished paragraph.",
    "Tone swings naturally between warm check-ins, teasing, and practical planning depending on what the other person says.",
    "Follow-up questions are small and casual rather than formal or fully explained.",
  ]
  if code_switch_ratio >= 0.15:
    tone_notes.append("Hindi and English are mixed naturally when the moment calls for it.")
  if supportive_ratio >= 0.10:
    tone_notes.append(
      "When the other person is down, replies become gently concerned and supportive without sounding preachy."
    )

  common_tokens = [token for token, _ in token_counts.most_common(8)] or ["okayy", "haa", "yeahh"]
  common_phrases = [phrase for phrase, _ in phrase_counts.most_common(6)] or [
    "okayy",
    "oh no",
    "you should",
  ]

  middle_index = len(message_lengths) // 2
  median_message_length = message_lengths[middle_index] if message_lengths else 0

  return StyleProfile(
    avg_reply_burst=round(sum(burst_sizes) / len(burst_sizes), 2),
    median_message_length=median_message_length,
    common_tokens=common_tokens,
    common_phrases=common_phrases,
    tone_notes=tone_notes,
  )


def format_turn(turn: MessageTurn, friend_name: str, partner_name: str) -> str:
  label = friend_name if turn.speaker == friend_name else partner_name
  return f"{label}: {clip_text(turn.flattened())}"


def format_recent_chat(
  turns: Sequence[MessageTurn],
  friend_name: str,
  partner_name: str,
  max_turns: int = 8,
) -> str:
  if not turns:
    return "No live session yet."
  return "\n".join(format_turn(turn, friend_name, partner_name) for turn in turns[-max_turns:])


def format_retrieved_examples(
  examples: Sequence[ReplyExample],
  friend_name: str,
  partner_name: str,
) -> str:
  if not examples:
    return "No retrieved examples."
  blocks = [
    f"Example {index}\n{example.prompt_block(friend_name, partner_name)}"
    for index, example in enumerate(examples, start=1)
  ]
  return "\n\n".join(blocks)


def build_style_example_bundle(
  examples: Sequence[ReplyExample],
  style_profile: StyleProfile | None = None,
) -> StyleExampleBundle:
  example_list = list(examples)
  if not example_list:
    return StyleExampleBundle(
      examples=[],
      shape_notes=["- Keep replies short and text-like."],
      phrase_notes=style_profile.common_phrases[:4] if style_profile else [],
      rhythm_examples=[],
    )

  burst_sizes = [example.reply_burst_count for example in example_list]
  avg_burst = round(sum(burst_sizes) / len(burst_sizes), 1)
  multi_bubble_ratio = sum(1 for size in burst_sizes if size >= 2) / len(burst_sizes)
  code_switched_ratio = sum(
    1
    for example in example_list
    for message in example.reply_turn.messages
    if detect_code_switching(message)
  ) / max(sum(len(example.reply_turn.messages) for example in example_list), 1)

  phrase_counts = Counter(
    message.strip()
    for example in example_list
    for message in example.reply_turn.messages
    if 2 <= len(message.strip()) <= 30 and count_urls(message) == 0
  )

  shape_notes = [
    f"- Similar replies average about {avg_burst:.1f} short bubbles.",
    "- Replies should feel like fast texts, not polished paragraphs.",
  ]
  if multi_bubble_ratio >= 0.5:
    shape_notes.append("- Multi-bubble replies are common when there is something to respond to.")
  if code_switched_ratio >= 0.15:
    shape_notes.append("- Hindi and English can be mixed naturally when it fits the moment.")

  phrase_notes = [phrase for phrase, _ in phrase_counts.most_common(6)]
  if not phrase_notes and style_profile is not None:
    phrase_notes = style_profile.common_phrases[:4]

  rhythm_examples: list[str] = []
  for example in example_list[:4]:
    tags = ", ".join(example.tags[:2]) if example.tags else "casual"
    pattern = infer_interaction_pattern(example.user_turn, example.reply_turn, example.tags)
    rhythm_examples.append(f"{example.reply_burst_count} bubble(s), {tags}: {pattern}")

  return StyleExampleBundle(
    examples=example_list,
    shape_notes=shape_notes,
    phrase_notes=phrase_notes,
    rhythm_examples=rhythm_examples,
  )


def build_search_query(
  user_text: str,
  recent_turns: Sequence[MessageTurn],
  friend_name: str,
  partner_name: str,
) -> str:
  recent_chat = format_recent_chat(recent_turns, friend_name, partner_name, max_turns=4)
  return "\n".join(
    [
      "Recent chat:",
      recent_chat,
      "",
      "Latest message:",
      user_text.strip(),
    ]
  ).strip()


def build_dialogue_search_query(
  user_text: str,
  recent_turns: Sequence[MessageTurn],
  friend_name: str,
  partner_name: str,
) -> str:
  recent_chat = format_recent_chat(recent_turns, friend_name, partner_name, max_turns=6)
  return "\n".join(
    [
      "Live dialogue state:",
      recent_chat,
      "",
      "Newest partner message:",
      user_text.strip(),
    ]
  ).strip()


def score_dialogue_snippet_match(
  user_text: str,
  recent_turns: Sequence[MessageTurn],
  snippet: DialogueSnippet,
  friend_name: str,
  partner_name: str,
) -> float:
  live_context = format_recent_chat(recent_turns, friend_name, partner_name, max_turns=4)
  query_text = " ".join((live_context, user_text)).strip()
  query_tokens = tokenize_text(query_text)
  lead_tokens = tokenize_text(" ".join(turn.flattened() for turn in snippet.lead_context))
  partner_tokens = tokenize_text(snippet.partner_turn.flattened())
  trailing_tokens = tokenize_text(" ".join(turn.flattened() for turn in snippet.trailing_context))
  pattern_tokens = tokenize_text(snippet.interaction_pattern)

  partner_overlap = len(query_tokens & partner_tokens)
  lead_overlap = len(query_tokens & lead_tokens)
  trailing_overlap = len(query_tokens & trailing_tokens)
  pattern_overlap = len(query_tokens & pattern_tokens)
  overlap_ratio = len(query_tokens & snippet.search_tokens) / max(len(query_tokens), 1)

  query_tags = set(infer_intent_tags(query_text))
  tag_overlap = len(query_tags & set(snippet.tags))

  return (
    (partner_overlap * 2.8)
    + (lead_overlap * 1.1)
    + (trailing_overlap * 0.5)
    + (pattern_overlap * 0.8)
    + (overlap_ratio * 4.5)
    + (tag_overlap * 3.5)
    + (0.4 if "?" in user_text and "?" in snippet.partner_turn.flattened() else 0.0)
  )


def rank_dialogue_snippets(
  user_text: str,
  recent_turns: Sequence[MessageTurn],
  candidates: Sequence[DialogueSnippet],
  friend_name: str = DEFAULT_PERSONA_NAME,
  partner_name: str = DEFAULT_PARTNER_NAME,
) -> list[DialogueSnippet]:
  query_tags = set(
    infer_intent_tags(
      " ".join(
        (format_recent_chat(recent_turns, friend_name, partner_name, max_turns=4), user_text)
      )
    )
  )
  return sorted(
    candidates,
    key=lambda snippet: (
      score_dialogue_snippet_match(user_text, recent_turns, snippet, friend_name, partner_name),
      len(query_tags & set(snippet.tags)),
      snippet.reply_burst_count,
      -snippet.reply_character_count,
    ),
    reverse=True,
  )


def score_example_match(
  user_text: str,
  recent_turns: Sequence[MessageTurn],
  example: ReplyExample,
  friend_name: str,
  partner_name: str,
) -> float:
  query_text = " ".join(
    (format_recent_chat(recent_turns, friend_name, partner_name, max_turns=3), user_text)
  )
  query_tokens = tokenize_text(query_text)
  user_tokens = tokenize_text(example.user_turn.flattened())
  reply_tokens = tokenize_text(example.reply_turn.flattened())
  context_tokens = tokenize_text(" ".join(turn.flattened() for turn in example.context))

  user_overlap = len(query_tokens & user_tokens)
  reply_overlap = len(query_tokens & reply_tokens)
  context_overlap = len(query_tokens & context_tokens)
  overlap_ratio = user_overlap / max(len(query_tokens), 1)

  query_tags = set(infer_intent_tags(query_text))
  tag_overlap = len(query_tags & set(example.tags))

  question_bonus = (
    0.5
    if "?" in user_text and any("?" in message for message in example.user_turn.messages)
    else 0.0
  )
  recent_overlap = 0.0
  if recent_turns:
    recent_overlap = (
      len(query_tokens & tokenize_text(" ".join(turn.flattened() for turn in recent_turns[-2:])))
      * 0.05
    )

  return (
    (user_overlap * 2.4)
    + (reply_overlap * 0.6)
    + (context_overlap * 0.3)
    + (overlap_ratio * 4.0)
    + (tag_overlap * 3.5)
    + question_bonus
    + recent_overlap
  )


def engagement_score_for_query(
  example: ReplyExample,
  query_tags: set[str],
) -> float:
  reply_text = example.reply_turn.flattened().lower()
  reply_has_question = "?" in reply_text
  score = 0.0

  if "supportive" in query_tags:
    if text_contains_any(
      reply_text,
      ("oh no", "take care", "are you okay", "kya hua", "its okay", "okayy", "talk to me"),
    ):
      score += 2.2
    if reply_has_question:
      score += 0.9

  if "planning" in query_tags:
    if text_contains_any(
      reply_text,
      (
        "lets go",
        "let's go",
        "come",
        "wait",
        "when",
        "tomorrow",
        "time",
        "kab",
        "milte",
        "library",
      ),
    ):
      score += 1.8
    if reply_has_question:
      score += 0.7

  if "status" in query_tags:
    if text_contains_any(
      reply_text,
      (
        "i'm",
        "im ",
        "i am",
        "nothing much",
        "room",
        "party",
        "studying",
        "sleeping",
        "home",
        "bas",
      ),
    ):
      score += 1.6
    if reply_has_question:
      score += 0.6

  if "affection" in query_tags:
    if text_contains_any(
      reply_text, ("miss you too", "love you too", "awh", "aww", "cute", "miss you", "🥹")
    ):
      score += 1.8

  if "greeting" in query_tags and (reply_has_question or len(example.reply_turn.messages) >= 2):
    score += 0.8

  if "academic" in query_tags and text_contains_any(
    reply_text,
    ("send", "pdf", "graph", "table", "class", "lab", "likh", "kar"),
  ):
    score += 1.0

  return score


def style_specificity_penalty(example: ReplyExample) -> float:
  reply_text = example.reply_turn.flattened().lower()
  penalty = 0.0
  penalty += count_urls(reply_text) * 4.0
  penalty += sum(1 for token in tokenize_text(reply_text) if len(token) >= 10) * 0.2
  if any(char.isdigit() for char in reply_text):
    penalty += 0.5
  if len(reply_text) > 120:
    penalty += 0.4
  return penalty


def rank_reply_examples(
  user_text: str,
  recent_turns: Sequence[MessageTurn],
  candidates: Sequence[ReplyExample],
  friend_name: str = DEFAULT_PERSONA_NAME,
  partner_name: str = DEFAULT_PARTNER_NAME,
) -> list[ReplyExample]:
  query_tags = set(
    infer_intent_tags(
      " ".join(
        (format_recent_chat(recent_turns, friend_name, partner_name, max_turns=3), user_text)
      )
    )
  )
  return sorted(
    candidates,
    key=lambda example: (
      score_example_match(user_text, recent_turns, example, friend_name, partner_name)
      + engagement_score_for_query(example, query_tags),
      example.reply_burst_count,
      -example.reply_character_count,
    ),
    reverse=True,
  )


def rank_style_examples(
  user_text: str,
  recent_turns: Sequence[MessageTurn],
  candidates: Sequence[ReplyExample],
  friend_name: str = DEFAULT_PERSONA_NAME,
  partner_name: str = DEFAULT_PARTNER_NAME,
) -> list[ReplyExample]:
  query_tags = set(
    infer_intent_tags(
      " ".join(
        (format_recent_chat(recent_turns, friend_name, partner_name, max_turns=3), user_text)
      )
    )
  )
  return sorted(
    candidates,
    key=lambda example: (
      engagement_score_for_query(example, query_tags) * 1.6
      + score_example_match(user_text, recent_turns, example, friend_name, partner_name) * 0.7
      + (0.3 if example.reply_burst_count <= 3 else -0.5)
      - style_specificity_penalty(example)
    ),
    reverse=True,
  )


def reply_shape_metrics(messages: Sequence[str]) -> dict[str, float]:
  cleaned = [message.strip() for message in messages if message.strip()]
  if not cleaned:
    return {
      "burst_count": 0,
      "total_chars": 0,
      "avg_chars": 0.0,
      "code_switch_ratio": 0.0,
    }

  total_chars = sum(len(message) for message in cleaned)
  code_switched = sum(1 for message in cleaned if detect_code_switching(message))
  return {
    "burst_count": len(cleaned),
    "total_chars": total_chars,
    "avg_chars": round(total_chars / len(cleaned), 2),
    "code_switch_ratio": round(code_switched / len(cleaned), 2),
  }


def compare_reply_shape(
  candidate_messages: Sequence[str], reference_turn: MessageTurn
) -> dict[str, object]:
  candidate = reply_shape_metrics(candidate_messages)
  reference = reply_shape_metrics(reference_turn.messages)
  return {
    "candidate": candidate,
    "reference": reference,
    "burst_delta": abs(candidate["burst_count"] - reference["burst_count"]),
    "character_delta": abs(candidate["total_chars"] - reference["total_chars"]),
    "code_switch_delta": round(
      abs(candidate["code_switch_ratio"] - reference["code_switch_ratio"]), 2
    ),
  }


def reply_example_to_document(
  example: ReplyExample, friend_name: str, partner_name: str
) -> Document:
  return Document(
    page_content=example.prompt_block(friend_name, partner_name),
    metadata={
      "example_id": example.example_id,
      "tags": ",".join(example.tags),
      "reply_burst_count": example.reply_burst_count,
    },
  )


def dialogue_snippet_to_document(
  snippet: DialogueSnippet, friend_name: str, partner_name: str
) -> Document:
  return Document(
    page_content=snippet.prompt_block(friend_name, partner_name),
    metadata={
      "snippet_id": snippet.snippet_id,
      "tags": ",".join(snippet.tags),
      "reply_burst_count": snippet.reply_burst_count,
    },
  )


def export_reply_examples(
  reply_examples: Sequence[ReplyExample],
  partner_name: str,
  friend_name: str,
  output_path: str | Path,
) -> None:
  target_path = Path(output_path)
  target_path.parent.mkdir(parents=True, exist_ok=True)

  with target_path.open("w", encoding="utf-8") as handle:
    for example in reply_examples:
      prompt_messages = [
        {
          "role": "user" if turn.speaker == partner_name else "assistant",
          "content": turn.flattened("\n"),
        }
        for turn in example.context
      ]
      prompt_messages.append({"role": "user", "content": example.user_turn.flattened("\n")})
      prompt_messages.append({"role": "assistant", "content": example.reply_turn.flattened("\n")})

      record = {
        "id": example.example_id,
        "tags": example.tags,
        "prompt": "\n".join(
          [
            *[format_turn(turn, friend_name, partner_name) for turn in example.context],
            f"{partner_name}: " + example.user_turn.flattened("\n"),
          ]
        ).strip(),
        "completion": example.reply_turn.flattened("\n"),
        "messages": prompt_messages,
      }
      handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_source_hash(source_path: str | Path) -> str:
  return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()


def compute_index_manifest(
  source_path: Path, friend_name: str, partner_name: str
) -> dict[str, object]:
  source_bytes = source_path.read_bytes()
  return {
    "parser_version": PARSER_VERSION,
    "friend_name": friend_name,
    "partner_name": partner_name,
    "source_path": str(source_path),
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "source_size": len(source_bytes),
  }


def load_existing_manifest(path: Path) -> dict[str, object] | None:
  if not path.exists():
    return None
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return None


def batched_documents(
  documents: Sequence[Document],
  batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[Document]]:
  return [
    list(documents[index : index + batch_size]) for index in range(0, len(documents), batch_size)
  ]


def build_vector_store(
  documents: Sequence[Document],
  manifest: dict[str, object],
  artifact_dir: str | Path,
  embedding_model: str,
  collection_name: str,
  progress_callback: BuildProgressCallback | None = None,
) -> tuple[Chroma | None, str | None]:
  artifact_path = Path(artifact_dir)
  manifest_path = artifact_path / MANIFEST_FILENAME
  db_location = artifact_path / DEFAULT_DB_DIRNAME

  artifact_path.mkdir(parents=True, exist_ok=True)
  should_rebuild = load_existing_manifest(manifest_path) != manifest or not db_location.exists()

  if should_rebuild and db_location.exists():
    emit_progress(progress_callback, "vector", "Removing stale vector index")
    shutil.rmtree(db_location)

  try:
    emit_progress(progress_callback, "vector", f"Loading embedding model `{embedding_model}`")
    embeddings = OllamaEmbeddings(
      model=embedding_model,
      sync_client_kwargs={"timeout": EMBEDDING_TIMEOUT_SECONDS},
    )
    embeddings.embed_query(EMBEDDING_WARMUP_TEXT)
    emit_progress(progress_callback, "vector", f"Embedding model `{embedding_model}` is ready")
    vector_store = Chroma(
      collection_name=collection_name,
      persist_directory=str(db_location),
      embedding_function=embeddings,
    )
    if should_rebuild:
      document_batches = batched_documents(documents)
      total_batches = max(len(document_batches), 1)
      emit_progress(
        progress_callback,
        "index",
        f"Indexing reply examples (0/{total_batches} batches)",
        completed=0,
        total=total_batches,
      )
      for batch_index, document_batch in enumerate(document_batches, start=1):
        vector_store.add_documents(
          documents=document_batch,
          ids=[
            str(
              document.metadata.get("snippet_id")
              or document.metadata.get("example_id")
              or f"document-{batch_index}-{offset}"
            )
            for offset, document in enumerate(document_batch)
          ],
        )
        emit_progress(
          progress_callback,
          "index",
          f"Indexing reply examples ({batch_index}/{total_batches} batches)",
          completed=batch_index,
          total=total_batches,
        )
      manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
      emit_progress(
        progress_callback,
        "vector",
        "Saved vector index manifest",
        completed=total_batches,
        total=total_batches,
      )
    else:
      emit_progress(progress_callback, "vector", "Using existing vector index")
    return vector_store, None
  except Exception as exc:
    error_name = type(exc).__name__
    return None, f"{error_name}: {exc}"


class ConversationRetriever:
  def __init__(
    self,
    reply_examples: Sequence[ReplyExample],
    dialogue_snippets: Sequence[DialogueSnippet],
    friend_name: str,
    partner_name: str,
    vector_store: Chroma | None = None,
  ) -> None:
    self.reply_examples = list(reply_examples)
    self.dialogue_snippets = list(dialogue_snippets)
    self.friend_name = friend_name
    self.partner_name = partner_name
    self.vector_store = vector_store
    self.examples_by_id = {example.example_id: example for example in self.reply_examples}
    self.snippets_by_id = {snippet.snippet_id: snippet for snippet in self.dialogue_snippets}

  def find_dialogue_snippets(
    self,
    user_text: str,
    recent_turns: Sequence[MessageTurn] | None = None,
    k: int = 4,
  ) -> list[DialogueSnippet]:
    recent_turns = list(recent_turns or [])
    candidate_snippets: list[DialogueSnippet] = []
    query_tags = set(
      infer_intent_tags(
        " ".join(
          (
            format_recent_chat(recent_turns, self.friend_name, self.partner_name, max_turns=4),
            user_text,
          )
        )
      )
    )

    if self.vector_store is not None:
      try:
        query = build_dialogue_search_query(
          user_text, recent_turns, self.friend_name, self.partner_name
        )
        documents = self.vector_store.similarity_search(query, k=max(k * 3, 10))
        candidate_snippets = [
          self.snippets_by_id[document.metadata["snippet_id"]]
          for document in documents
          if document.metadata.get("snippet_id") in self.snippets_by_id
        ]
      except Exception:
        candidate_snippets = []

    if not candidate_snippets:
      candidate_snippets = self.dialogue_snippets

    meaningful_tags = query_tags - {"casual"}
    if meaningful_tags:
      tag_matched_snippets = [
        snippet for snippet in candidate_snippets if meaningful_tags & set(snippet.tags)
      ]
      if len(tag_matched_snippets) >= max(k * 2, 6):
        candidate_snippets = tag_matched_snippets

    ranked_snippets = rank_dialogue_snippets(
      user_text,
      recent_turns,
      candidate_snippets,
      friend_name=self.friend_name,
      partner_name=self.partner_name,
    )
    unique_snippets: list[DialogueSnippet] = []
    seen_ids: set[str] = set()
    for snippet in ranked_snippets:
      if snippet.snippet_id in seen_ids:
        continue
      unique_snippets.append(snippet)
      seen_ids.add(snippet.snippet_id)
      if len(unique_snippets) >= k:
        break

    return unique_snippets

  def find_examples(
    self,
    user_text: str,
    recent_turns: Sequence[MessageTurn] | None = None,
    k: int = 4,
  ) -> list[DialogueSnippet]:
    return self.find_dialogue_snippets(user_text, recent_turns, k=k)

  def find_style_examples(
    self,
    user_text: str,
    recent_turns: Sequence[MessageTurn] | None = None,
    k: int = 4,
  ) -> list[ReplyExample]:
    recent_turns = list(recent_turns or [])
    candidate_examples = self.reply_examples
    matched_snippets = self.find_dialogue_snippets(user_text, recent_turns, k=max(k * 2, 6))
    if matched_snippets:
      matched_reply_ids = {snippet.dialogue_signature for snippet in matched_snippets}
      filtered_examples = [
        example
        for example in self.reply_examples
        if example.dialogue_signature in matched_reply_ids
      ]
      if filtered_examples:
        candidate_examples = filtered_examples

    ranked_examples = rank_style_examples(
      user_text,
      recent_turns,
      candidate_examples,
      friend_name=self.friend_name,
      partner_name=self.partner_name,
    )
    return ranked_examples[:k]


def collection_suffix(text: str) -> str:
  lowered = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
  return lowered or "profile"


def build_chat_assets(
  chat_export_path: str | Path,
  friend_name: str,
  artifact_dir: str | Path,
  embedding_model: str | None = None,
  retrieval_mode: str = "lexical",
  progress_callback: BuildProgressCallback | None = None,
) -> tuple[ChatAssets, list[str]]:
  source_path = Path(chat_export_path).expanduser().resolve()
  emit_progress(progress_callback, "parse", f"Parsing WhatsApp export `{source_path.name}`")
  messages = parse_whatsapp_messages(source_path)
  ensure_two_party_chat(messages)
  if not messages:
    raise ValueError("The WhatsApp export did not contain any parseable messages.")

  emit_progress(progress_callback, "turns", f"Parsed {len(messages)} messages, building turns")
  turns = build_message_turns(messages)
  emit_progress(progress_callback, "examples", "Building reply-conditioned examples")
  reply_examples, partner_name = build_reply_examples(turns, friend_name)
  dialogue_snippets, _partner_name = build_dialogue_snippets(turns, friend_name)
  style_profile = build_style_profile(reply_examples)
  emit_progress(
    progress_callback,
    "examples",
    f"Built {len(reply_examples)} reply examples and {len(dialogue_snippets)} dialogue snippets for {friend_name}",
  )

  artifact_path = Path(artifact_dir).expanduser()
  artifact_path.mkdir(parents=True, exist_ok=True)
  emit_progress(progress_callback, "export", "Exporting cleaned reply examples")
  export_reply_examples(
    reply_examples,
    partner_name,
    friend_name,
    artifact_path / DATASET_EXPORT_FILENAME,
  )

  manifest = compute_index_manifest(source_path, friend_name, partner_name)
  vector_store = None
  actual_mode = "lexical"
  warnings: list[str] = []

  if retrieval_mode == "embedding" and embedding_model:
    emit_progress(
      progress_callback, "retrieval", f"Preparing embedding retrieval with `{embedding_model}`"
    )
    vector_store, vector_error = build_vector_store(
      [
        dialogue_snippet_to_document(snippet, friend_name, partner_name)
        for snippet in dialogue_snippets
      ],
      manifest,
      artifact_path,
      embedding_model,
      collection_name=f"dialogue_snippets_{collection_suffix(friend_name)}",
      progress_callback=progress_callback,
    )
    if vector_store is None:
      warnings.append(
        f"Embedding retrieval could not be initialized with `{embedding_model}`, so this profile is using lexical retrieval."
        + (f" Reason: {vector_error}" if vector_error else "")
      )
      emit_progress(progress_callback, "retrieval", "Falling back to lexical retrieval")
    else:
      actual_mode = "embedding"
      emit_progress(
        progress_callback, "retrieval", f"Embedding retrieval is ready with `{embedding_model}`"
      )
  else:
    emit_progress(progress_callback, "retrieval", "Using lexical retrieval")

  emit_progress(progress_callback, "done", f"Profile artifacts ready with {actual_mode} retrieval")

  retriever = ConversationRetriever(
    reply_examples,
    dialogue_snippets,
    friend_name,
    partner_name,
    vector_store,
  )
  return (
    ChatAssets(
      friend_name=friend_name,
      partner_name=partner_name,
      style_profile=style_profile,
      reply_examples=reply_examples,
      dialogue_snippets=dialogue_snippets,
      retriever=retriever,
      retrieval_mode=actual_mode,
      artifact_dir=artifact_path,
    ),
    warnings,
  )


@lru_cache(maxsize=4)
def get_chat_assets(
  chat_export_path: str | Path,
  friend_name: str,
  artifact_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
  *,
  embedding_model: str | None = None,
  retrieval_mode: str = "lexical",
) -> ChatAssets:
  assets, _warnings = build_chat_assets(
    chat_export_path,
    friend_name,
    artifact_dir,
    embedding_model=embedding_model,
    retrieval_mode=retrieval_mode,
  )
  return assets
