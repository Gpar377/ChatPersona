from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from rich.console import Console
from typer.testing import CliRunner

from chatpersona import __version__
from chatpersona.cli import app, prompt_for_export_path
from chatpersona.ollama import OllamaModel
from chatpersona.profiles import build_profile_artifacts, create_profile, load_profile, save_profile

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "sample_whatsapp_chat.txt"
SAMPLE_CHAT = FIXTURE_PATH.read_text(encoding="utf-8")


class CliIntegrationTests(unittest.TestCase):
  def setUp(self) -> None:
    self.runner = CliRunner()

  def test_module_help_lists_doctor_command(self) -> None:
    result = subprocess.run(
      [sys.executable, "-m", "chatpersona", "--help"],
      capture_output=True,
      text=True,
      check=False,
      cwd=Path(__file__).resolve().parents[1],
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("doctor", result.stdout)

  def test_doctor_help_works(self) -> None:
    result = self.runner.invoke(app, ["doctor", "--help"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("doctor", result.output)

  def test_version_option_outputs_package_version(self) -> None:
    result = self.runner.invoke(app, ["--version"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn(__version__, result.output)

  def test_init_command_creates_profile(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      export_path = Path(temp_dir) / "chat.txt"
      export_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      with (
        mock.patch.dict(os.environ, {"CHATPERSONA_HOME": temp_dir}, clear=False),
        mock.patch("chatpersona.cli.prompt_for_export_path", return_value=export_path),
        mock.patch("chatpersona.cli.choose_persona", return_value="Sam"),
        mock.patch("chatpersona.cli.prompt_for_profile_name", return_value="sam-sample"),
        mock.patch(
          "chatpersona.cli.list_models_or_fallback",
          return_value=([OllamaModel(name="llama3.2:latest", source="test")], None),
        ),
        mock.patch("chatpersona.cli.choose_chat_model", return_value="llama3.2:latest"),
        mock.patch("chatpersona.cli.prompt_for_start_chat", return_value=False),
      ):
        result = self.runner.invoke(app, ["init"])

      self.assertEqual(result.exit_code, 0, result.output)
      self.assertIn("chatpersona setup", result.output)
      self.assertIn("Profile ready", result.output)
      saved_profile = load_profile("sam-sample", base_dir=temp_dir)
      self.assertEqual(saved_profile.persona_name, "Sam")
      self.assertEqual(saved_profile.retrieval_mode, "lexical")
      self.assertEqual(saved_profile.last_built_retrieval_mode, "lexical")
      self.assertTrue((Path(saved_profile.artifact_dir) / "reply_examples.jsonl").exists())

  def test_running_without_arguments_starts_onboarding_when_no_profiles_exist(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      export_path = Path(temp_dir) / "chat.txt"
      export_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      with (
        mock.patch.dict(os.environ, {"CHATPERSONA_HOME": temp_dir}, clear=False),
        mock.patch("chatpersona.cli.prompt_for_export_path", return_value=export_path),
        mock.patch("chatpersona.cli.choose_persona", return_value="Sam"),
        mock.patch("chatpersona.cli.prompt_for_profile_name", return_value="sam-sample"),
        mock.patch(
          "chatpersona.cli.list_models_or_fallback",
          return_value=([OllamaModel(name="llama3.2:latest", source="test")], None),
        ),
        mock.patch("chatpersona.cli.choose_chat_model", return_value="llama3.2:latest"),
        mock.patch("chatpersona.cli.prompt_for_start_chat", return_value=False),
      ):
        result = self.runner.invoke(app, [])

      self.assertEqual(result.exit_code, 0, result.output)
      self.assertIn("Create your first local persona.", result.output)
      saved_profile = load_profile("sam-sample", base_dir=temp_dir)
      self.assertEqual(saved_profile.persona_name, "Sam")

  def test_running_without_arguments_opens_dashboard_when_profiles_exist(self) -> None:
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

        with (
          mock.patch("chatpersona.cli.run_doctor_checks", return_value=[]),
          mock.patch("chatpersona.cli.prompt_for_dashboard_action", return_value="Exit"),
        ):
          result = self.runner.invoke(app, [])

      self.assertEqual(result.exit_code, 0, result.output)
      self.assertIn("Local WhatsApp persona chat", result.output)
      self.assertIn("Saved profiles", result.output)
      self.assertIn("Actions", result.output)
      self.assertIn("sam-sample", result.output)

  def test_init_parse_error_shows_whatsapp_guidance(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      export_path = Path(temp_dir) / "chat.txt"
      export_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      with (
        mock.patch("chatpersona.cli.prompt_for_export_path", return_value=export_path),
        mock.patch(
          "chatpersona.cli.parse_whatsapp_messages",
          side_effect=ValueError("Only 1:1 WhatsApp exports are supported right now."),
        ),
      ):
        result = self.runner.invoke(app, ["init"])

      self.assertEqual(result.exit_code, 1, result.output)
      self.assertIn("parseable 1:1 WhatsApp", result.output)

  def test_chat_command_loads_profile_and_starts_session(self) -> None:
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

        with mock.patch("chatpersona.cli.interactive_chat_session") as session_mock:
          result = self.runner.invoke(app, ["chat", "sam-sample"])

      self.assertEqual(result.exit_code, 0, result.output)
      session_mock.assert_called_once()

  def test_chat_missing_profile_shows_next_steps(self) -> None:
    result = self.runner.invoke(app, ["chat", "missing-profile"])

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("profiles list", result.output)

  def test_rebuild_command_refreshes_profile_hash(self) -> None:
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
        old_hash = built_profile.source_hash

        export_path.write_text(
          SAMPLE_CHAT
          + "\n22/08/2025, 09:54 - Alex: library tomorrow?\n22/08/2025, 09:55 - Sam: haa",
          encoding="utf-8",
        )

        result = self.runner.invoke(app, ["rebuild", "sam-sample"])
        refreshed = load_profile("sam-sample")

      self.assertEqual(result.exit_code, 0, result.output)
      self.assertNotEqual(old_hash, refreshed.source_hash)

  def test_doctor_command_reports_ready_environment(self) -> None:
    models = [
      OllamaModel(name="llama3.2:latest", source="test"),
      OllamaModel(name="qwen3-embedding:0.6b", source="test"),
    ]

    with (
      tempfile.TemporaryDirectory() as temp_dir,
      mock.patch("chatpersona.cli.list_models_via_api", return_value=models),
      mock.patch("chatpersona.cli.list_local_models", return_value=models),
      mock.patch("chatpersona.cli.get_profiles_dir", return_value=Path(temp_dir)),
    ):
      result = self.runner.invoke(app, ["doctor"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Local setup looks ready.", result.output)
    self.assertIn("qwen3-embedding:0.6b", result.output)

  def test_doctor_command_handles_ollama_unavailable(self) -> None:
    with (
      tempfile.TemporaryDirectory() as temp_dir,
      mock.patch(
        "chatpersona.cli.list_models_via_api", side_effect=RuntimeError("connection refused")
      ),
      mock.patch(
        "chatpersona.cli.list_local_models", side_effect=RuntimeError("connection refused")
      ),
      mock.patch("chatpersona.cli.get_profiles_dir", return_value=Path(temp_dir)),
    ):
      result = self.runner.invoke(app, ["doctor"])

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("blocking local setup issues", result.output)
    self.assertIn("Ollama reachable", result.output)

  def test_doctor_command_reports_no_embedding_model(self) -> None:
    models = [OllamaModel(name="llama3.2:latest", source="test")]

    with (
      tempfile.TemporaryDirectory() as temp_dir,
      mock.patch("chatpersona.cli.list_models_via_api", return_value=models),
      mock.patch("chatpersona.cli.list_local_models", return_value=models),
      mock.patch("chatpersona.cli.get_profiles_dir", return_value=Path(temp_dir)),
    ):
      result = self.runner.invoke(app, ["doctor"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Embedding model", result.output)
    self.assertIn("qwen3-embedding:0.6b", result.output)
    self.assertIn("Non-blocking issues were found", result.output)

  def test_doctor_command_handles_unwritable_profile_storage(self) -> None:
    models = [
      OllamaModel(name="llama3.2:latest", source="test"),
      OllamaModel(name="qwen3-embedding:0.6b", source="test"),
    ]

    with (
      tempfile.TemporaryDirectory() as temp_dir,
      mock.patch("chatpersona.cli.list_models_via_api", return_value=models),
      mock.patch("chatpersona.cli.list_local_models", return_value=models),
      mock.patch("chatpersona.cli.get_profiles_dir", return_value=Path(temp_dir)),
      mock.patch(
        "chatpersona.cli.tempfile.NamedTemporaryFile",
        side_effect=PermissionError("Permission denied"),
      ),
    ):
      result = self.runner.invoke(app, ["doctor"])

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("Profile storage", result.output)
    self.assertIn("CHATPERSONA_HOME", result.output)


class ExportBrowserTests(unittest.TestCase):
  def make_console(self) -> tuple[Console, StringIO]:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120)
    console.clear = mock.Mock()  # type: ignore[method-assign]
    return console, stream

  def run_export_picker(
    self,
    start_dir: Path,
    home_dir: Path,
    number_choices: list[int],
    manual_inputs: list[str] | None = None,
  ) -> tuple[Path, str]:
    console, stream = self.make_console()
    unexpected_manual_prompt = AssertionError("Manual path prompt was not expected in this test.")

    with (
      mock.patch("chatpersona.cli.Path.cwd", return_value=start_dir),
      mock.patch("chatpersona.cli.Path.home", return_value=home_dir),
      mock.patch("chatpersona.cli.prompt_for_number_choice", side_effect=number_choices),
      mock.patch(
        "chatpersona.cli.Prompt.ask",
        side_effect=manual_inputs if manual_inputs is not None else unexpected_manual_prompt,
      ),
    ):
      selected_path = prompt_for_export_path(console)

    return selected_path, stream.getvalue()

  def test_export_browser_selects_txt_file_from_start_directory(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      start_dir.mkdir(parents=True)
      home_dir.mkdir(parents=True)
      chat_path = start_dir / "sample_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[0])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("sample_chat.txt", output)
      self.assertIn("Folder View", output)

  def test_export_browser_navigates_into_directory_and_selects_nested_txt_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      nested_dir = start_dir / "exports"
      nested_dir.mkdir(parents=True)
      home_dir.mkdir(parents=True)
      chat_path = nested_dir / "nested_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[0, 0])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("exports/", output)
      self.assertIn("nested_chat.txt", output)

  def test_export_browser_jump_to_downloads_selects_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      downloads_dir = home_dir / "Downloads"
      desktop_dir = home_dir / "Desktop"
      start_dir.mkdir(parents=True)
      downloads_dir.mkdir(parents=True)
      desktop_dir.mkdir(parents=True)
      chat_path = downloads_dir / "download_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[2, 0])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("Jump to Downloads", output)

  def test_export_browser_jump_to_desktop_selects_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      downloads_dir = home_dir / "Downloads"
      desktop_dir = home_dir / "Desktop"
      start_dir.mkdir(parents=True)
      downloads_dir.mkdir(parents=True)
      desktop_dir.mkdir(parents=True)
      chat_path = desktop_dir / "desktop_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[3, 0])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("Jump to Desktop", output)

  def test_export_browser_jump_to_home_selects_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      downloads_dir = home_dir / "Downloads"
      desktop_dir = home_dir / "Desktop"
      start_dir.mkdir(parents=True)
      downloads_dir.mkdir(parents=True)
      desktop_dir.mkdir(parents=True)
      chat_path = home_dir / "home_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[4, 2])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("Jump to Home", output)

  def test_export_browser_jump_to_start_directory_selects_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      downloads_dir = home_dir / "Downloads"
      desktop_dir = home_dir / "Desktop"
      start_dir.mkdir(parents=True)
      downloads_dir.mkdir(parents=True)
      desktop_dir.mkdir(parents=True)
      chat_path = start_dir / "start_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[3, 1, 0])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("Jump to Start Directory", output)

  def test_export_browser_manual_path_fallback_accepts_valid_txt_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      downloads_dir = home_dir / "Downloads"
      desktop_dir = home_dir / "Desktop"
      start_dir.mkdir(parents=True)
      downloads_dir.mkdir(parents=True)
      desktop_dir.mkdir(parents=True)
      chat_path = Path(temp_dir) / "manual_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(
        start_dir,
        home_dir,
        number_choices=[5],
        manual_inputs=[str(chat_path)],
      )

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("Enter path manually", output)

  def test_export_browser_manual_path_retry_shows_guidance(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      downloads_dir = home_dir / "Downloads"
      desktop_dir = home_dir / "Desktop"
      start_dir.mkdir(parents=True)
      downloads_dir.mkdir(parents=True)
      desktop_dir.mkdir(parents=True)
      chat_path = Path(temp_dir) / "manual_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(
        start_dir,
        home_dir,
        number_choices=[5],
        manual_inputs=["missing.txt", str(chat_path)],
      )

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("That path does not point to a readable `.txt` export", output)

  def test_export_browser_hides_non_txt_files(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      start_dir.mkdir(parents=True)
      home_dir.mkdir(parents=True)
      (start_dir / "notes.md").write_text("# not a chat export", encoding="utf-8")
      chat_path = start_dir / "visible_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[0])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("visible_chat.txt", output)
      self.assertNotIn("notes.md", output)

  def test_export_browser_hides_hidden_files_and_directories(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      start_dir.mkdir(parents=True)
      home_dir.mkdir(parents=True)
      hidden_dir = start_dir / ".hidden"
      hidden_dir.mkdir()
      (hidden_dir / "hidden_chat.txt").write_text(SAMPLE_CHAT, encoding="utf-8")
      (start_dir / ".secret_chat.txt").write_text(SAMPLE_CHAT, encoding="utf-8")
      chat_path = start_dir / "visible_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(start_dir, home_dir, number_choices=[0])

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("visible_chat.txt", output)
      self.assertNotIn(".hidden", output)
      self.assertNotIn(".secret_chat.txt", output)

  def test_export_browser_empty_directory_renders_actions_without_crashing(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      start_dir = Path(temp_dir) / "start"
      home_dir = Path(temp_dir) / "home"
      downloads_dir = home_dir / "Downloads"
      desktop_dir = home_dir / "Desktop"
      start_dir.mkdir(parents=True)
      downloads_dir.mkdir(parents=True)
      desktop_dir.mkdir(parents=True)
      chat_path = Path(temp_dir) / "manual_chat.txt"
      chat_path.write_text(SAMPLE_CHAT, encoding="utf-8")

      selected_path, output = self.run_export_picker(
        start_dir,
        home_dir,
        number_choices=[5],
        manual_inputs=[str(chat_path)],
      )

      self.assertEqual(selected_path, chat_path.resolve())
      self.assertIn("(no folders or .txt files here)", output)
      self.assertIn("Back", output)
      self.assertIn("Enter path manually", output)


if __name__ == "__main__":
  unittest.main()
