from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
import tempfile

import typer
from rich.console import Console
from rich.progress import (
  BarColumn,
  Progress,
  SpinnerColumn,
  TaskProgressColumn,
  TextColumn,
  TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.text import Text
from rich.tree import Tree

from chatpersona.corpus import (
  BuildProgressUpdate,
  detect_participants,
  ensure_two_party_chat,
  parse_whatsapp_messages,
)
from chatpersona import __version__
from chatpersona.ollama import (
  DEFAULT_CHAT_MODEL,
  OLLAMA_TAGS_URL,
  OllamaModel,
  choose_default_chat_model,
  choose_embedding_model,
  filter_chat_models,
  is_embedding_model_name,
  list_local_models,
  list_models_via_api,
)
from chatpersona.profiles import (
  ChatProfile,
  build_profile_artifacts,
  create_profile,
  get_profiles_dir,
  list_profiles,
  load_profile,
  profile_needs_rebuild,
  save_profile,
)
from chatpersona.runtime import interactive_chat_session
from chatpersona.ui import (
  ACCENT,
  ERROR,
  SUCCESS,
  WARNING,
  compact_table,
  message_panel,
  notice_line,
  print_screen,
  screen_title,
  section_rule,
  shortcut_footer,
  status_line,
)

app = typer.Typer(
  help="Reusable local CLI for building and chatting with WhatsApp-based personas.",
  no_args_is_help=False,
)
profiles_app = typer.Typer(help="Inspect saved persona profiles.")
models_app = typer.Typer(help="Inspect local Ollama models.")
app.add_typer(profiles_app, name="profiles")
app.add_typer(models_app, name="models")

console = Console()


def fail(message: str, exit_code: int = 1) -> None:
  console.print(message_panel("Something Needs Attention", message, ERROR))
  raise typer.Exit(code=exit_code)


def version_callback(value: bool) -> None:
  if not value:
    return
  console.print(f"chatpersona {__version__}")
  raise typer.Exit()


@dataclass(slots=True)
class DoctorCheckResult:
  name: str
  passed: bool
  detail: str
  next_step: str | None = None
  critical: bool = True


@dataclass(slots=True)
class ExportBrowserOption:
  label: str
  kind: str
  path: Path | None = None


def normalize_path_input(raw_value: str) -> Path:
  return Path(raw_value.strip().strip("\"'")).expanduser()


def is_accessible_export_directory(path: Path) -> bool:
  try:
    return path.is_dir() and os.access(path, os.R_OK | os.X_OK)
  except OSError:
    return False


def is_accessible_export_file(path: Path) -> bool:
  try:
    return path.is_file() and path.suffix.lower() == ".txt" and os.access(path, os.R_OK)
  except OSError:
    return False


def list_export_browser_paths(current_dir: Path) -> tuple[list[Path], str | None]:
  try:
    children = list(current_dir.iterdir())
  except OSError as exc:
    return [], compact_message(str(exc))

  directories: list[Path] = []
  files: list[Path] = []

  for child in children:
    if child.name.startswith("."):
      continue
    if is_accessible_export_directory(child):
      directories.append(child)
      continue
    if is_accessible_export_file(child):
      files.append(child)

  directories.sort(key=lambda path: path.name.lower())
  files.sort(key=lambda path: path.name.lower())
  return directories + files, None


def build_export_jump_targets(start_dir: Path) -> list[tuple[str, Path]]:
  targets = [
    ("Jump to Start Directory", start_dir),
    ("Jump to Downloads", Path.home() / "Downloads"),
    ("Jump to Desktop", Path.home() / "Desktop"),
    ("Jump to Home", Path.home()),
  ]
  deduped_targets: list[tuple[str, Path]] = []
  seen_paths: set[Path] = set()

  for label, target in targets:
    resolved_target = target.expanduser().resolve()
    if resolved_target in seen_paths or not is_accessible_export_directory(resolved_target):
      continue
    deduped_targets.append((label, resolved_target))
    seen_paths.add(resolved_target)

  return deduped_targets


def build_export_browser_options(
  current_dir: Path, start_dir: Path
) -> tuple[list[ExportBrowserOption], str | None]:
  visible_paths, browse_error = list_export_browser_paths(current_dir)
  options: list[ExportBrowserOption] = []

  for path in visible_paths:
    if is_accessible_export_directory(path):
      options.append(ExportBrowserOption(label=f"{path.name}/", kind="directory", path=path))
    else:
      options.append(ExportBrowserOption(label=path.name, kind="file", path=path))

  options.append(ExportBrowserOption(label="Back", kind="back", path=current_dir.parent))
  options.extend(
    ExportBrowserOption(label=label, kind="jump", path=target)
    for label, target in build_export_jump_targets(start_dir)
  )
  options.append(ExportBrowserOption(label="Enter path manually", kind="manual"))
  return options, browse_error


def build_export_browser_tree(current_dir: Path, options: list[ExportBrowserOption]) -> Tree:
  tree = Tree(Text(str(current_dir), style=f"bold {ACCENT}"))
  visible_entries = [option for option in options if option.kind in {"directory", "file"}]

  if not visible_entries:
    tree.add(Text("(no folders or .txt files here)", style="grey70"))
    return tree

  for index, option in enumerate(visible_entries, start=1):
    line = Text(f"{index}. ", style=f"bold {ACCENT}")
    if option.kind == "directory":
      line.append(option.label, style=f"bold {ACCENT}")
    else:
      line.append(option.label, style="white")
    tree.add(line)

  return tree


def render_export_browser_actions(options: list[ExportBrowserOption]):
  table = compact_table(None, ["#", "Action"])
  for index, option in enumerate(options, start=1):
    if option.kind in {"directory", "file"}:
      continue
    table.add_row(str(index), option.label)
  return table


def prompt_for_manual_export_path(console_obj: Console) -> Path:
  while True:
    raw_value = Prompt.ask("Path to the WhatsApp export")
    candidate = normalize_path_input(raw_value)
    if is_accessible_export_file(candidate):
      return candidate.resolve()
    console_obj.print(
      "[yellow]That path does not point to a readable `.txt` export. Try again.[/yellow]"
    )


def render_export_browser(
  console_obj: Console,
  current_dir: Path,
  options: list[ExportBrowserOption],
  browse_error: str | None = None,
  feedback_message: tuple[str, str] | None = None,
) -> None:
  renderables = [
    screen_title("Choose WhatsApp Export", "Browse folders and select a `.txt` chat export."),
    status_line("Folder", str(current_dir)),
    section_rule("Folder View"),
    build_export_browser_tree(current_dir, options),
  ]

  visible_entries = [option for option in options if option.kind in {"directory", "file"}]
  if not visible_entries:
    renderables.append(Text(""))
    renderables.append(notice_line("No folders or `.txt` files are available here."))

  if browse_error:
    renderables.append(Text(""))
    renderables.append(notice_line(f"Could not fully list this folder: {browse_error}", WARNING))

  if feedback_message is not None:
    message, tone = feedback_message
    renderables.append(Text(""))
    renderables.append(notice_line(message, tone))

  renderables.extend(
    [
      Text(""),
      section_rule("Actions"),
      render_export_browser_actions(options),
      shortcut_footer(
        "Directories open. `.txt` files select. Hidden entries stay hidden. "
        "Manual path is always available."
      ),
    ]
  )
  print_screen(console_obj, *renderables, clear=True)


def show_onboarding_intro(console_obj: Console) -> None:
  print_screen(
    console_obj,
    screen_title(
      "chatpersona setup",
      "Create a local persona from your own WhatsApp export.",
    ),
    notice_line("WhatsApp 1:1 `.txt` exports only."),
    notice_line("Everything stays on this machine."),
    shortcut_footer("Run `chatpersona doctor` first if local Ollama or model setup looks off."),
    clear=True,
  )


def prompt_for_export_path(console_obj: Console) -> Path:
  start_dir = Path.cwd().resolve()
  current_dir = start_dir
  feedback_message: tuple[str, str] | None = None

  while True:
    options, browse_error = build_export_browser_options(current_dir, start_dir)
    render_export_browser(
      console_obj,
      current_dir,
      options,
      browse_error=browse_error,
      feedback_message=feedback_message,
    )
    feedback_message = None

    selected_index = prompt_for_number_choice(
      console_obj, "Choose a number", len(options), default="1"
    )
    selected_option = options[selected_index]

    if selected_option.kind == "directory" and selected_option.path is not None:
      current_dir = selected_option.path.resolve()
      continue
    if selected_option.kind == "file" and selected_option.path is not None:
      return selected_option.path.resolve()
    if selected_option.kind == "back":
      parent_dir = current_dir.parent
      if parent_dir == current_dir:
        feedback_message = ("Already at the filesystem root.", WARNING)
      else:
        current_dir = parent_dir
      continue
    if selected_option.kind == "jump" and selected_option.path is not None:
      current_dir = selected_option.path.resolve()
      continue
    if selected_option.kind == "manual":
      return prompt_for_manual_export_path(console_obj)


def prompt_for_profile_name(console_obj: Console, default_value: str) -> str:
  return Prompt.ask("Profile name", default=default_value).strip() or default_value


def compact_message(message: str, limit: int = 220) -> str:
  compact = " ".join(message.split())
  lowered = compact.lower()

  if "sigabrt" in lowered or "uncaught exception" in lowered or "using native backtrace" in lowered:
    compact = "the local `ollama` command crashed while checking installed models"
  elif "operation not permitted" in lowered or "connection refused" in lowered:
    compact = f"could not reach the local Ollama server at {OLLAMA_TAGS_URL}"
  elif "failed to establish a new connection" in lowered:
    compact = f"the local Ollama server at {OLLAMA_TAGS_URL} did not respond"
  elif "timeout" in lowered or "timed out" in lowered:
    compact = "a local Ollama request timed out"

  if len(compact) <= limit:
    return compact
  return compact[: limit - 3].rstrip() + "..."


def format_command_error(command_name: str, error: Exception) -> str:
  summary = compact_message(str(error))
  lowered = summary.lower()

  if isinstance(error, FileNotFoundError) and "profile" in lowered:
    return (
      "Profile not found.\n"
      "Likely cause: that name does not match a saved profile.\n"
      "Next: `chatpersona profiles list`."
    )

  if "profile metadata" in lowered or "corrupted profile" in lowered:
    return (
      "Saved profile metadata could not be read.\n"
      "Likely cause: the local `profile.json` is invalid or incomplete.\n"
      "Next: recreate the profile with `chatpersona init`."
    )

  if "1:1 whatsapp" in lowered or "whatsapp export" in lowered or "parseable messages" in lowered:
    return (
      "Could not read a valid WhatsApp export.\n"
      "Likely cause: the file is not a parseable 1:1 WhatsApp `.txt` export.\n"
      f"Next: re-run `chatpersona {command_name}` with a direct 1:1 export."
    )

  if (
    "ollama" in lowered
    or "local models" in lowered
    or "embedding retrieval" in lowered
    or "timed out" in lowered
  ):
    return (
      f"{summary}.\n"
      "Likely cause: Ollama is not running locally or the required model is missing.\n"
      "Next: `chatpersona doctor`."
    )

  return (
    f"{summary}.\n"
    "Likely cause: local setup is incomplete or the export/profile needs attention.\n"
    "Next: `chatpersona doctor`."
  )


def choose_from_models(
  console_obj: Console, models: list[OllamaModel], prompt_text: str, default_name: str
) -> str:
  if not models:
    return DEFAULT_CHAT_MODEL

  console_obj.print(section_rule("Chat models"))
  table = compact_table(None, ["#", "Model", "Size", "Source"])
  default_index = 1

  for index, model in enumerate(models, start=1):
    if model.name == default_name:
      default_index = index
    table.add_row(str(index), model.name, model.size or "-", model.source or "-")

  console_obj.print(table)
  while True:
    raw_value = Prompt.ask(prompt_text, default=str(default_index)).strip()
    if raw_value.isdigit():
      selected_index = int(raw_value) - 1
      if 0 <= selected_index < len(models):
        return models[selected_index].name
    console_obj.print("[yellow]Pick one of the numbered models above.[/yellow]")


def prompt_for_number_choice(
  console_obj: Console, prompt_text: str, total_options: int, default: str = "1"
) -> int:
  while True:
    raw_value = Prompt.ask(prompt_text, default=default).strip()
    if raw_value.isdigit():
      selected_index = int(raw_value) - 1
      if 0 <= selected_index < total_options:
        return selected_index
    console_obj.print("[yellow]Pick one of the numbered options above.[/yellow]")


def choose_from_options(console_obj: Console, title: str, options: list[str]) -> str:
  console_obj.print(section_rule(title))
  table = compact_table(None, ["#", "Option"])
  for index, option in enumerate(options, start=1):
    table.add_row(str(index), option)
  console_obj.print(table)

  selected_index = prompt_for_number_choice(
    console_obj, "Choose a number", len(options), default="1"
  )
  return options[selected_index]


def choose_persona(console_obj: Console, participants: list[str]) -> str:
  return choose_from_options(console_obj, "Choose Which Participant To Emulate", participants)


def choose_chat_model(console_obj: Console, available_models: list[OllamaModel]) -> str:
  chat_models = filter_chat_models(available_models)
  default_name = choose_default_chat_model(chat_models)
  return choose_from_models(console_obj, chat_models, "Choose the chat model", default_name)


def list_models_or_fallback() -> tuple[list[OllamaModel], str | None]:
  try:
    models = list_local_models()
    return models, None
  except RuntimeError as exc:
    return [], compact_message(str(exc))


def build_and_save_profile(
  profile: ChatProfile,
  progress_callback=None,
) -> tuple[ChatProfile, list[str]]:
  updated_profile, _assets, warnings = build_profile_artifacts(
    profile, progress_callback=progress_callback
  )
  save_profile(updated_profile)
  return updated_profile, warnings


def format_retrieval_summary(profile: ChatProfile) -> str:
  if (
    profile.last_built_retrieval_mode
    and profile.last_built_retrieval_mode != profile.retrieval_mode
  ):
    return f"{profile.retrieval_mode} (last build used: {profile.last_built_retrieval_mode})"
  return profile.retrieval_mode


def format_profile_ready_message(profile: ChatProfile) -> str:
  return (
    f"Profile: {profile.profile_name}\n"
    f"Persona: {profile.persona_name}\n"
    f"Partner: {profile.partner_name}\n"
    f"Chat model: {profile.chat_model}\n"
    f"Preferred retrieval mode: {profile.retrieval_mode}\n"
    f"Last build retrieval mode: {profile.last_built_retrieval_mode}\n"
    f"Embedding model: {profile.embedding_model or '-'}\n"
    f"Artifacts: {profile.artifact_dir}\n\n"
    "What happens next:\n"
    f'- `chatpersona chat "{profile.profile_name}"`\n'
    f'- `chatpersona rebuild "{profile.profile_name}"` after the export changes\n'
    "- `chatpersona doctor` if Ollama or model checks fail"
  )


def check_python_version() -> DoctorCheckResult:
  version_text = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
  if sys.version_info >= (3, 11):
    return DoctorCheckResult(
      name="Python version",
      passed=True,
      detail=f"Using Python {version_text}, which meets the >=3.11 requirement.",
    )
  return DoctorCheckResult(
    name="Python version",
    passed=False,
    detail=f"Using Python {version_text}, but chatpersona requires Python 3.11 or newer.",
    next_step="Create a Python 3.11+ virtual environment and reinstall the project there.",
  )


def check_ollama_reachable() -> DoctorCheckResult:
  try:
    models = list_models_via_api(timeout_seconds=2.0)
    detail = f"Ollama responded at {OLLAMA_TAGS_URL}."
    if models:
      detail += f" API reported {len(models)} local model(s)."
    else:
      detail += " API is up, but no local models were reported yet."
    return DoctorCheckResult(name="Ollama reachable", passed=True, detail=detail)
  except Exception as exc:
    return DoctorCheckResult(
      name="Ollama reachable",
      passed=False,
      detail=compact_message(str(exc)),
      next_step="Start Ollama locally and make sure the server is reachable at http://127.0.0.1:11434.",
    )


def check_local_model_listing() -> tuple[DoctorCheckResult, list[OllamaModel]]:
  try:
    models = list_local_models()
    return (
      DoctorCheckResult(
        name="Local models",
        passed=True,
        detail=f"Found {len(models)} local Ollama model(s).",
      ),
      models,
    )
  except Exception as exc:
    return (
      DoctorCheckResult(
        name="Local models",
        passed=False,
        detail=compact_message(str(exc)),
        next_step="Run `ollama list` manually and make sure at least one local model is installed.",
      ),
      [],
    )


def check_chat_model_availability(models: list[OllamaModel]) -> DoctorCheckResult:
  chat_models = filter_chat_models(models)
  if chat_models and any(not is_embedding_model_name(model.name) for model in chat_models):
    default_model = choose_default_chat_model(chat_models)
    return DoctorCheckResult(
      name="Chat model",
      passed=True,
      detail=f"Usable chat models were found. Default choice: `{default_model}`.",
    )
  return DoctorCheckResult(
    name="Chat model",
    passed=False,
    detail="No usable local chat model was found.",
    next_step="Install one with `ollama pull llama3.2:latest` or another local chat model.",
  )


def check_embedding_model_availability(models: list[OllamaModel]) -> DoctorCheckResult:
  embedding_model = choose_embedding_model(models)
  if embedding_model:
    return DoctorCheckResult(
      name="Embedding model",
      passed=True,
      detail=f"Embedding retrieval can use `{embedding_model}`.",
      critical=False,
    )
  return DoctorCheckResult(
    name="Embedding model",
    passed=False,
    detail="No local embedding model was found. chatpersona can still run with lexical retrieval.",
    next_step="Install one with `ollama pull qwen3-embedding:0.6b` to enable embedding retrieval.",
    critical=False,
  )


def check_profile_storage_writable() -> DoctorCheckResult:
  profiles_dir = get_profiles_dir()
  try:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=profiles_dir, prefix=".chatpersona-doctor-", delete=True):
      pass
    return DoctorCheckResult(
      name="Profile storage",
      passed=True,
      detail=f"Local profile storage is writable at `{profiles_dir}`.",
    )
  except Exception as exc:
    return DoctorCheckResult(
      name="Profile storage",
      passed=False,
      detail=compact_message(str(exc)),
      next_step=f"Make sure `{profiles_dir}` is writable, or set `CHATPERSONA_HOME` to a writable directory.",
    )


def run_doctor_checks() -> list[DoctorCheckResult]:
  results = [check_python_version(), check_ollama_reachable()]
  model_result, models = check_local_model_listing()
  results.append(model_result)
  results.append(check_chat_model_availability(models))
  results.append(check_embedding_model_availability(models))
  results.append(check_profile_storage_writable())
  return results


def render_doctor_results(results: list[DoctorCheckResult]) -> tuple[int, int]:
  console.print(screen_title("chatpersona doctor", "Local setup checks"))
  console.print(section_rule("Checks"))
  table = compact_table(None, ["Check", "Status", "Detail", "Next step"])

  critical_failures = 0
  noncritical_failures = 0

  for result in results:
    if result.passed:
      status = "[green]PASS[/green]"
    else:
      status = "[red]FAIL[/red]"
      if result.critical:
        critical_failures += 1
      else:
        noncritical_failures += 1
    table.add_row(
      result.name,
      status,
      result.detail,
      result.next_step or "-",
    )

  console.print(table)
  return critical_failures, noncritical_failures


def format_build_stage(stage: str) -> str:
  labels = {
    "parse": "Import",
    "turns": "Import",
    "examples": "Examples",
    "export": "Export dataset",
    "retrieval": "Embeddings",
    "vector": "Embeddings",
    "index": "Index",
    "done": "Ready",
  }
  return labels.get(stage, stage.replace("_", " ").title())


def print_onboarding_stage(step: int, total: int, label: str, description: str) -> None:
  console.print("")
  console.print(status_line(f"Step {step}/{total}", label))
  console.print(notice_line(description))


def render_environment_snapshot(results: list[DoctorCheckResult]):
  critical_failures = 0
  noncritical_failures = 0
  for result in results:
    if not result.passed:
      if result.critical:
        critical_failures += 1
      else:
        noncritical_failures += 1

  if critical_failures:
    return status_line("Local", f"{critical_failures} issue(s) need attention", WARNING)
  if noncritical_failures:
    return status_line("Local", f"Ready with {noncritical_failures} optional fallback(s)", ACCENT)
  return status_line("Local", "Ready", SUCCESS)


def render_profiles_table(profiles: list[ChatProfile]):
  table = compact_table(
    None, ["Profile", "Persona", "Partner", "Chat model", "Retrieval", "Status"]
  )
  for profile in profiles:
    status = "stale" if profile_needs_rebuild(profile) else "ready"
    table.add_row(
      profile.profile_name,
      profile.persona_name,
      profile.partner_name or "-",
      profile.chat_model,
      format_retrieval_summary(profile),
      status,
    )
  return table


def render_models_table(models: list[OllamaModel]):
  table = compact_table(None, ["Name", "Kind", "Size", "Source"])
  for model in models:
    table.add_row(
      model.name,
      "embedding" if is_embedding_model_name(model.name) else "chat",
      model.size or "-",
      model.source or "-",
    )
  return table


def render_action_table():
  table = compact_table(None, ["#", "Action", "What it does"])
  for index, row in enumerate(
    [
      ("Chat", "Start chatting with a saved profile."),
      ("Create profile", "Run setup for a new export."),
      ("Rebuild profile", "Refresh artifacts after the export changes."),
      ("Doctor", "Run local environment checks."),
      ("Models", "List local Ollama models."),
      ("Exit", "Close the launcher."),
    ],
    start=1,
  ):
    table.add_row(str(index), row[0], row[1])
  return table


def render_launcher(profiles: list[ChatProfile], doctor_results: list[DoctorCheckResult]) -> None:
  ready_profiles = sum(0 if profile_needs_rebuild(profile) else 1 for profile in profiles)
  print_screen(
    console,
    screen_title("chatpersona", "Local WhatsApp persona chat"),
    render_environment_snapshot(doctor_results),
    status_line("Profiles", f"{len(profiles)} saved, {ready_profiles} ready"),
    section_rule("Saved profiles"),
    render_profiles_table(profiles),
    section_rule("Actions"),
    render_action_table(),
    shortcut_footer("Type a number to continue."),
    clear=True,
  )


def render_first_run_launcher() -> None:
  print_screen(
    console,
    screen_title("chatpersona", "Create your first local persona."),
    notice_line("No saved profiles yet."),
    notice_line("Need: a WhatsApp 1:1 `.txt` export and a local Ollama chat model."),
    shortcut_footer("Opening setup now."),
    clear=True,
  )


def prompt_for_dashboard_action() -> str:
  options = ["Chat", "Create profile", "Rebuild profile", "Doctor", "Models", "Exit"]
  selected_index = prompt_for_number_choice(console, "Choose an action", len(options), default="1")
  return options[selected_index]


def choose_saved_profile(profiles: list[ChatProfile], prompt_title: str) -> ChatProfile | None:
  if not profiles:
    return None
  selected_name = choose_from_options(
    console, prompt_title, [profile.profile_name for profile in profiles]
  )
  for profile in profiles:
    if profile.profile_name == selected_name:
      return profile
  return None


def launch_dashboard() -> None:
  profiles = list_profiles()
  if not profiles:
    render_first_run_launcher()
    run_onboarding_flow()
    return

  doctor_results = run_doctor_checks()
  render_launcher(profiles, doctor_results)
  try:
    action = prompt_for_dashboard_action()
  except (EOFError, KeyboardInterrupt):
    raise typer.Exit()

  if action == "Chat":
    profile = choose_saved_profile(profiles, "Choose A Profile To Chat With")
    if profile is not None:
      chat_command(profile.profile_name)
    return
  if action == "Create profile":
    run_onboarding_flow()
    return
  if action == "Rebuild profile":
    profile = choose_saved_profile(profiles, "Choose A Profile To Rebuild")
    if profile is not None:
      rebuild_command(profile.profile_name)
    return
  if action == "Doctor":
    doctor_command()
    return
  if action == "Models":
    models_list_command()
    return


def run_with_build_progress(label: str, builder):
  with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    console=console,
    transient=False,
  ) as progress:
    task_id = progress.add_task(label, total=None)

    def handle_update(update: BuildProgressUpdate) -> None:
      stage_label = format_build_stage(update.stage)
      description = stage_label
      if update.stage == "index" and update.total:
        completed = max(0, min(update.completed or 0, update.total))
        description = f"{stage_label} {completed}/{update.total}"
      if update.total is None:
        progress.update(
          task_id,
          description=description,
          total=None,
        )
        return

      total = max(update.total, 1)
      completed = max(0, min(update.completed or 0, total))
      progress.update(
        task_id,
        description=description,
        total=total,
        completed=completed,
      )

    result = builder(handle_update)
    current_total = progress.tasks[0].total
    if current_total is None:
      progress.update(task_id, description=f"{label} complete", total=1, completed=1)
    else:
      progress.update(task_id, description=f"{label} complete", completed=current_total)
    return result


def prompt_for_start_chat(console_obj: Console) -> bool:
  return Confirm.ask("Start chatting with this profile now?", default=True)


def run_onboarding_flow(start_chat_after_setup: bool = True) -> ChatProfile | None:
  show_onboarding_intro(console)
  print_onboarding_stage(1, 4, "Export", "Choose the WhatsApp `.txt` export you want to use.")
  export_path = prompt_for_export_path(console)

  try:
    messages = parse_whatsapp_messages(export_path)
    participants = ensure_two_party_chat(messages)
  except Exception as exc:
    fail(format_command_error("init", exc))

  if not participants:
    fail("No parseable WhatsApp messages were found in that export.")

  print_onboarding_stage(2, 4, "Persona", "Choose who to emulate, then name the saved profile.")
  persona_name = choose_persona(console, participants)
  other_participants = [name for name in detect_participants(messages) if name != persona_name]
  partner_name = other_participants[0] if other_participants else ""
  default_profile_name = f"{persona_name} - {export_path.stem}"
  profile_name = prompt_for_profile_name(console, default_profile_name)

  available_models, model_error = list_models_or_fallback()
  print_onboarding_stage(3, 4, "Model", "Choose the chat model and review retrieval mode.")
  if available_models:
    chat_model = choose_chat_model(console, available_models)
  else:
    console.print(notice_line("Could not auto-detect local Ollama models.", WARNING))
    if model_error:
      console.print(notice_line(f"Reason: {model_error}", WARNING))
    chat_model = (
      Prompt.ask("Chat model name", default=DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
    )

  embedding_model = choose_embedding_model(available_models)
  retrieval_mode = "embedding" if embedding_model else "lexical"
  console.print(section_rule("Retrieval"))
  console.print(status_line("Preferred", retrieval_mode))
  console.print(status_line("Embedding model", embedding_model or "-"))
  if not embedding_model:
    console.print(
      notice_line("No embedding model found, so this profile will use lexical retrieval.", WARNING)
    )

  profile = create_profile(
    profile_name=profile_name,
    chat_export_path=export_path,
    persona_name=persona_name,
    partner_name=partner_name,
    chat_model=chat_model,
    embedding_model=embedding_model,
    retrieval_mode=retrieval_mode,
  )

  print_onboarding_stage(
    4, 4, "Build", "Building local artifacts. The first run can take a little longer."
  )
  try:
    updated_profile, warnings = run_with_build_progress(
      "Building profile artifacts",
      lambda progress_callback: build_and_save_profile(
        profile, progress_callback=progress_callback
      ),
    )
  except Exception as exc:
    fail(format_command_error("init", exc))
    return None

  for warning in warnings:
    console.print(notice_line(warning, WARNING))

  console.print(
    message_panel(
      "Profile ready",
      format_profile_ready_message(updated_profile),
      SUCCESS,
    )
  )
  if start_chat_after_setup and prompt_for_start_chat(console):
    ready_profile, assets, runtime_warnings = run_with_build_progress(
      "Preparing chat profile",
      lambda progress_callback: build_profile_artifacts(
        updated_profile,
        progress_callback=progress_callback,
      ),
    )
    save_profile(ready_profile)
    for warning in runtime_warnings:
      console.print(notice_line(warning, WARNING))
    interactive_chat_session(ready_profile, assets, console)
    return ready_profile

  return updated_profile


@app.callback(invoke_without_command=True)
def app_callback(
  ctx: typer.Context,
  version: bool = typer.Option(
    False,
    "--version",
    help="Show the installed chatpersona version and exit.",
    callback=version_callback,
    is_eager=True,
  ),
) -> None:
  del version
  if ctx.invoked_subcommand is not None:
    return
  launch_dashboard()
  raise typer.Exit()


@app.command("init")
def init_command() -> None:
  run_onboarding_flow(start_chat_after_setup=True)


@app.command("onboard")
def onboard_command() -> None:
  run_onboarding_flow(start_chat_after_setup=True)


@app.command("doctor")
def doctor_command() -> None:
  results = render_doctor_results(run_doctor_checks())
  critical_failures, noncritical_failures = results

  if critical_failures:
    fail(
      "chatpersona doctor found blocking local setup issues. "
      "Work through the failed checks above, then run `chatpersona doctor` again.",
      exit_code=1,
    )
  if noncritical_failures:
    console.print(
      notice_line(
        "Non-blocking issues were found. Some features may fall back to simpler behavior.", WARNING
      )
    )
    return

  console.print(notice_line("Local setup looks ready.", SUCCESS))


@app.command("chat")
def chat_command(profile: str) -> None:
  try:
    loaded_profile = load_profile(profile)
  except (FileNotFoundError, ValueError) as exc:
    fail(format_command_error("chat", exc))

  if profile_needs_rebuild(loaded_profile):
    console.print(
      notice_line("Profile artifacts are stale, so chatpersona will rebuild them first.", WARNING)
    )

  try:
    ready_profile, assets, warnings = run_with_build_progress(
      "Preparing chat profile",
      lambda progress_callback: build_profile_artifacts(
        loaded_profile,
        progress_callback=progress_callback,
      ),
    )
    save_profile(ready_profile)
  except Exception as exc:
    fail(format_command_error("chat", exc))
    return

  for warning in warnings:
    console.print(notice_line(warning, WARNING))

  interactive_chat_session(ready_profile, assets, console)


@app.command("rebuild")
def rebuild_command(profile: str) -> None:
  try:
    loaded_profile = load_profile(profile)
  except (FileNotFoundError, ValueError) as exc:
    fail(format_command_error("rebuild", exc))

  try:
    rebuilt_profile, warnings = run_with_build_progress(
      "Rebuilding profile artifacts",
      lambda progress_callback: build_and_save_profile(
        loaded_profile,
        progress_callback=progress_callback,
      ),
    )
  except Exception as exc:
    fail(format_command_error("rebuild", exc))
    return

  for warning in warnings:
    console.print(notice_line(warning, WARNING))

  console.print(
    message_panel(
      "Rebuild complete",
      f"Profile: {rebuilt_profile.profile_name}\n"
      f"Persona: {rebuilt_profile.persona_name}\n"
      f"Partner: {rebuilt_profile.partner_name}\n"
      f"Chat model: {rebuilt_profile.chat_model}\n"
      f"Retrieval mode: {format_retrieval_summary(rebuilt_profile)}",
      SUCCESS,
    )
  )


@profiles_app.command("list")
def profiles_list_command() -> None:
  profiles = list_profiles()
  if not profiles:
    console.print(screen_title("Profiles", "Saved local persona profiles"))
    console.print(notice_line("No saved profiles yet."))
    console.print(shortcut_footer("Run `chatpersona init` to create your first one."))
    return
  console.print(screen_title("Profiles", "Saved local persona profiles"))
  console.print(render_profiles_table(profiles))


@models_app.command("list")
def models_list_command() -> None:
  try:
    models = list_local_models()
  except RuntimeError as exc:
    fail(str(exc))

  if not models:
    console.print(screen_title("Models", "Installed Ollama models"))
    console.print(notice_line("No local Ollama models were found."))
    console.print(shortcut_footer("Run `ollama pull llama3.2:latest` and try again."))
    return
  console.print(screen_title("Models", "Installed Ollama models"))
  console.print(render_models_table(models))


def run() -> None:
  app()
