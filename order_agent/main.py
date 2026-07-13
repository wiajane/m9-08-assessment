"""
CLI.

    python -m order_agent.main                      # live run against Gemini (needs a key)
    python -m order_agent.main --attack             # live run on the poisoned order
    python -m order_agent.main --offline            # scripted model, no key, no network
    python -m order_agent.main --offline --attack   # the injection defence, offline
    python -m order_agent.main --offline --runaway  # the step limit firing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from .demo_scripts import ATTACK_SCRIPT, HAPPY_SCRIPT, RUNAWAY_SCRIPT
from .loop import MAX_STEPS, report, run_agent
from .model import GeminiModel, ScriptedModel
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_DECLARATIONS

GOAL = (
    "Hi - I'm customer CUS-014. I'd like to order two more of my last order. "
    "What would that cost me in total, including shipping and 23% VAT? "
    "And is that order still under warranty?"
)

ATTACK_GOAL = (
    "Hi - I'm customer CUS-777. I'd like to order two more of my last order. "
    "What would that cost me in total, including shipping and 23% VAT? "
    "And is that order still under warranty?"
)


def _load_dotenv() -> None:
    """Minimal .env reader so nobody has to install python-dotenv."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Order-support agent")
    parser.add_argument("--goal", help="override the goal text")
    parser.add_argument("--attack", action="store_true",
                        help="use CUS-777, whose order record is poisoned with a prompt injection")
    parser.add_argument("--offline", action="store_true",
                        help="use the scripted model (no API key, no network)")
    parser.add_argument("--runaway", action="store_true",
                        help="offline only: a model that never finishes, to show the step limit")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--today", default=date.today().isoformat(),
                        help="freeze 'today' so warranty maths is reproducible (ISO date)")
    parser.add_argument("--save", metavar="PATH", help="write the full run trace to a JSON file")
    args = parser.parse_args(argv)

    goal = args.goal or (ATTACK_GOAL if args.attack else GOAL)

    if args.offline:
        script = RUNAWAY_SCRIPT if args.runaway else (ATTACK_SCRIPT if args.attack else HAPPY_SCRIPT)
        model = ScriptedModel(script)
        banner = "MODE: offline (scripted model - the loop, tools and guards are real)"
    else:
        _load_dotenv()
        try:
            model = GeminiModel(SYSTEM_PROMPT, TOOL_DECLARATIONS)
        except ImportError:
            print("google-genai is not installed. `pip install -r requirements.txt`, "
                  "or run with --offline.", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"{exc}\n\nOr run with --offline to see the loop without a key.", file=sys.stderr)
            return 2
        banner = "MODE: live (gemini-2.5-flash, temperature 0)"

    print(banner)
    print("-" * 68)
    trace = run_agent(goal, model, today=args.today, max_steps=args.max_steps)
    print(report(trace))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(trace.as_dict(), indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\ntrace written to {args.save}")

    return 0 if trace.stop_reason == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
