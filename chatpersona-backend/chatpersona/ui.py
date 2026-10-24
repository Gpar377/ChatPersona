from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

APP_NAME = "chatpersona"
ACCENT = "cyan"
MUTED = "grey62"
SUCCESS = "green"
WARNING = "yellow"
ERROR = "red"
USER_COLOR = "bright_white"
PERSONA_COLOR = ACCENT


def screen_title(title: str, subtitle: str | None = None) -> Group:
  lines: list[RenderableType] = [Text(title, style=f"bold {ACCENT}")]
  if subtitle:
    lines.append(Text(subtitle, style=MUTED))
  return Group(*lines)


def section_rule(title: str) -> Rule:
  return Rule(title, align="left", style="grey35")


def status_line(label: str, value: str, tone: str = ACCENT) -> Text:
  text = Text()
  text.append(f"{label}  ", style=f"bold {tone}")
  text.append(value, style="white")
  return text


def compact_table(title: str | None, columns: list[str]) -> Table:
  table = Table(
    title=Text(title, style=f"bold {ACCENT}") if title else None,
    show_header=True,
    header_style=f"bold {MUTED}",
    expand=True,
    show_edge=False,
    pad_edge=False,
    box=None,
  )
  for index, column in enumerate(columns):
    table.add_column(column, style="bold" if index == 0 else "")
  return table


def notice_line(message: str, tone: str = MUTED) -> Text:
  text = Text()
  text.append("• ", style=tone)
  text.append(message, style=tone)
  return text


def shortcut_footer(message: str) -> Text:
  return Text(message, style=MUTED)


def message_panel(title: str, body: str, tone: str) -> Panel:
  return Panel.fit(
    Text.from_markup(body),
    title=Text(title, style=f"bold {tone}"),
    border_style=tone,
    padding=(0, 1),
  )


def print_screen(console: Console, *renderables: RenderableType, clear: bool = False) -> None:
  if clear:
    console.clear()
  console.print(Group(*renderables))
