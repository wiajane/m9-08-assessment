#!/usr/bin/env python3
"""
Capture a live run and paste it into the README.

    python capture_run.py

Runs the agent twice against Gemini - once on the ordinary goal, once on the
poisoned order - writes both transcripts to runs/, and drops them into the README
between the RUN:START / RUN:END markers so the captured run in the repo is always
the real thing rather than something I typed by hand.

Add --offline to do the same with the scripted model (no API key needed); the
README will say so.
"""

from __future__ import annotations

import contextlib
import io
import sys
from datetime import date
from pathlib import Path

from order_agent.main import main as run_cli

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
START = "<!-- RUN:START -->"
END = "<!-- RUN:END -->"


def capture(argv: list[str]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_cli(argv)
    return buffer.getvalue().rstrip()


def main() -> int:
    offline = "--offline" in sys.argv
    today = date.today().isoformat()
    base = ["--today", today] + (["--offline"] if offline else [])

    print("running the normal goal...", file=sys.stderr)
    normal = capture(base + ["--save", "runs/run_normal.json"])

    print("running the poisoned order...", file=sys.stderr)
    attack = capture(base + ["--attack", "--save", "runs/run_attack.json"])

    label = "scripted model, offline" if offline else "live Gemini call"
    block = "\n".join([
        START,
        f"*Captured on {today} with `python capture_run.py"
        f"{' --offline' if offline else ''}` ({label}). "
        f"Full traces: [`runs/run_normal.json`](runs/run_normal.json), "
        f"[`runs/run_attack.json`](runs/run_attack.json).*",
        "",
        "### Run 1 - the ordinary goal (CUS-014)",
        "",
        "```text",
        normal,
        "```",
        "",
        "### Run 2 - the same goal on the poisoned order (CUS-777)",
        "",
        "```text",
        attack,
        "```",
        END,
    ])

    text = README.read_text(encoding="utf-8")
    before, _, rest = text.partition(START)
    _, _, after = rest.partition(END)
    README.write_text(before + block + after, encoding="utf-8")
    print(f"README updated ({len(block)} chars of transcript).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
