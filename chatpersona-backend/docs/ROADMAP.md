# Roadmap

Last updated: 2026-03-22

## Highest-Value Upgrades

### 1. Make retrieval choice explicit

Let the user choose:

- `embedding`
- `lexical`
- `auto`

Why it matters:

- removes ambiguity when an embedding model exists but the user wants faster startup
- makes profile state easier to reason about
- creates a clean path for future benchmarking

### 2. Add cached startup mode

Goal:

- avoid reparsing the export on every `chat` launch when source hash and parser version are unchanged

Why it matters:

- noticeably faster startup on larger exports
- clearer separation between build-time and chat-time responsibilities

### 3. Add a lightweight evaluation harness

Start with local metrics such as:

- reply length distribution
- burst-count similarity
- intent-tag retrieval hit rate
- fallback frequency

Why it matters:

- gives you a safe way to tune prompts and retrieval without guessing

## Product Features Worth Adding

### Profile management

- `chatpersona profiles show "<profile>"`
- `chatpersona profiles delete "<profile>"`
- `chatpersona profiles rename "<profile>"`
- profile export/import for synthetic demo bundles

### Better onboarding

- direct validation for unsupported group exports
- preview of detected participants before selection
- optional default model persistence
- clearer explanation when embedding retrieval falls back to lexical

### Chat runtime improvements

- `/save-summary` or automatic local session summaries
- `/mode` for dry, warm, playful, supportive reply bias
- `/sources` to inspect the top retrieved examples for the last reply
- configurable response temperature and reply length targets

### Data and retrieval

- time-aware turn splitting so long gaps do not always merge into one turn
- optional recency weighting in lexical ranking
- profile rebuild diff summary showing new messages/examples added
- duplicate-example pruning for repetitive chats

### Quality and safety

- stronger tests around parser edge cases like BOMs, AM/PM timestamps, and unusual sender names
- smoke test lane that runs without LangChain/Chroma installed
- package-content test to verify fixtures and docs are shipped correctly
- structured error classes instead of broad `Exception` handling in runtime generation paths

## Longer-Term Ideas

- support multiple style presets per profile
- local fine-tuning/export pipeline from `reply_examples.jsonl`
- read-only conversation analytics dashboard
- multi-export persona fusion with clear weighting rules

## Recommended Build Order

1. Explicit retrieval choice
2. Cached startup path
3. Retrieval/source inspection command
4. Evaluation harness
5. Richer profile management commands
