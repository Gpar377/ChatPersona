from __future__ import annotations

import unittest

from chatpersona.ollama import (
  OllamaModel,
  choose_default_chat_model,
  choose_embedding_model,
  parse_ollama_list_output,
  parse_ollama_tags_payload,
)


class OllamaHelpersTests(unittest.TestCase):
  def test_parse_ollama_list_output_extracts_models(self) -> None:
    output = """
NAME                    ID              SIZE      MODIFIED
llama3.2:latest         abc123          2.0 GB    2 days ago
qwen3-embedding:0.6b    def456          900 MB    5 days ago
""".strip()

    models = parse_ollama_list_output(output)

    self.assertEqual([model.name for model in models], ["llama3.2:latest", "qwen3-embedding:0.6b"])
    self.assertEqual(models[0].size, "2.0 GB")

  def test_parse_ollama_tags_payload_extracts_models(self) -> None:
    payload = """
{
  "models": [
    {"name": "llama3.2:latest", "size": 2147483648, "modified_at": "2025-08-20T10:00:00Z"},
    {"name": "nomic-embed-text", "size": 274877906, "modified_at": "2025-08-18T10:00:00Z"}
  ]
}
""".strip()

    models = parse_ollama_tags_payload(payload)

    self.assertEqual(models[0].name, "llama3.2:latest")
    self.assertEqual(models[1].kind, "embedding")

  def test_embedding_selection_prefers_known_embedding_models(self) -> None:
    models = [
      OllamaModel(name="llama3.2:latest"),
      OllamaModel(name="nomic-embed-text"),
      OllamaModel(name="qwen3-embedding:0.6b"),
    ]

    self.assertEqual(choose_embedding_model(models), "qwen3-embedding:0.6b")
    self.assertEqual(choose_default_chat_model(models), "llama3.2:latest")


if __name__ == "__main__":
  unittest.main()
