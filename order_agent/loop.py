"""
The agent loop.

Plain think -> act -> observe. The model gets the goal and the tool declarations;
on every turn it either asks for tool calls (which I run and feed back) or produces
the final JSON. I never tell it which tool to use or in what order - the only thing
I enforce is the boundary conditions:

    MAX_STEPS      how many times the model may think
    MAX_TOOL_CALLS how many tool executions the whole run may spend
    MAX_REPAIRS    how many times a malformed final answer may be sent back

Hitting a bound is not a crash. The run ends with a valid, schema-shaped result
whose status is "incomplete" and whose `unmet_requirements` says what went wrong.
Anything consuming this agent gets the same shape whether the run went well or not,
which is the whole point of a structured contract.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import safety
from .model import FunctionCall, Model, ModelTurn
from .prompts import SYSTEM_PROMPT
from .schema import SchemaError, parse_final_json, validate_result
from .tools import TOOL_DECLARATIONS, call_tool

# A correct run needs 5-7 steps (look the order up, one or two calculations, one
# warranty check, one final answer). 10 leaves room for a couple of self-corrections
# without letting a confused model spend my quota all afternoon.
MAX_STEPS = 10
MAX_TOOL_CALLS = 12
MAX_REPAIRS = 1


@dataclass
class RunTrace:
    goal: str
    steps_used: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    security_events: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    result: dict | None = None
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps_used": self.steps_used,
            "stop_reason": self.stop_reason,
            "elapsed_s": round(self.elapsed_s, 2),
            "tool_calls": self.tool_calls,
            "security_events": self.security_events,
            "result": self.result,
        }


def _incomplete(reason: str, detail: str) -> dict:
    """A schema-valid result for a run that could not finish. Never a crash."""
    return {
        "status": "incomplete",
        "order": {"order_id": "", "customer_id": "", "order_date": "", "items": []},
        "reorder": {
            "multiplier": 0, "items_subtotal_eur": 0.0, "shipping_eur": 0.0,
            "vat_eur": 0.0, "total_eur": 0.0, "currency": "EUR",
        },
        "warranty": {
            "sku_checked": "", "in_warranty": False, "expires_on": "", "days_remaining": 0,
        },
        "summary": f"The agent stopped before finishing: {detail}",
        "unmet_requirements": [reason],
    }


def run_agent(goal: str, model: Model, *, today: str | None = None,
              max_steps: int = MAX_STEPS, verbose: bool = True) -> RunTrace:
    started = time.time()
    trace = RunTrace(goal=goal)
    history: list[dict] = [{"role": "user", "text": goal}]
    repairs_left = MAX_REPAIRS

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(f"GOAL: {goal}")
    log(f"bounds: max_steps={max_steps} max_tool_calls={MAX_TOOL_CALLS}\n")

    for step in range(1, max_steps + 1):
        trace.steps_used = step

        # --- think -------------------------------------------------------
        try:
            turn: ModelTurn = model.generate(history)
        except Exception as exc:  # network blip, quota, bad key...
            trace.stop_reason = "model_error"
            trace.result = _incomplete("model_unavailable", f"{type(exc).__name__}: {exc}")
            break

        # --- act ---------------------------------------------------------
        if turn.function_calls:
            if len(trace.tool_calls) + len(turn.function_calls) > MAX_TOOL_CALLS:
                trace.stop_reason = "tool_call_budget_exhausted"
                trace.result = _incomplete(
                    "tool_call_budget_exhausted",
                    f"the run hit the {MAX_TOOL_CALLS}-tool-call budget",
                )
                break

            history.append({"role": "model", "function_calls": turn.function_calls,
                            "text": turn.text})
            responses = []

            for call in turn.function_calls:
                log(f"[step {step}] -> {call.name}({json.dumps(call.args)})")

                raw = call_tool(call.name, call.args, today=today)

                # SAFETY: scan the result for instruction-shaped text and redact it
                # BEFORE it is ever put in the model's context.
                cleaned, events = safety.scan_for_injection(raw)
                for event in events:
                    event["step"] = step
                    event["tool"] = call.name
                    log(f"   !! SAFETY: {event['patterns']} in field '{event['field']}' -> redacted")
                trace.security_events.extend(events)

                ok = bool(cleaned.get("ok"))
                log(f"   <- {'ok' if ok else 'ERROR ' + str(cleaned.get('error'))}: "
                    f"{json.dumps(cleaned)[:160]}")

                trace.tool_calls.append({
                    "step": step,
                    "tool": call.name,
                    "args": call.args,
                    "ok": ok,
                    "result": cleaned,
                })
                responses.append({
                    "name": call.name,
                    # SAFETY: and it goes back inside an "this is data" envelope.
                    "response": safety.wrap_untrusted(call.name, cleaned),
                })

            history.append({"role": "tool", "function_responses": responses})
            continue

        # --- finish ------------------------------------------------------
        try:
            payload = validate_result(parse_final_json(turn.text))
        except SchemaError as exc:
            if repairs_left > 0:
                repairs_left -= 1
                log(f"[step {step}] final answer failed validation: {exc}")
                log("   -> asking the model to repair it (1 attempt allowed)")
                history.append({"role": "model", "text": turn.text})
                history.append({"role": "user", "text": (
                    "Your final answer did not match the required schema:\n- "
                    + "\n- ".join(exc.problems)
                    + "\nSend the corrected JSON object only. No prose, no code fences."
                )})
                continue
            trace.stop_reason = "invalid_final_output"
            trace.result = _incomplete("invalid_final_output",
                                       f"the model's answer never matched the schema ({exc})")
            break

        trace.stop_reason = "completed"
        trace.result = payload
        log(f"\n[step {step}] final answer received and validated against the schema.")
        break
    else:
        # Loop ran out of steps without a `break`: this is the step limit firing.
        trace.stop_reason = "step_limit_reached"
        trace.result = _incomplete(
            "step_limit_reached",
            f"the {max_steps}-step limit was reached before a final answer",
        )
        log(f"\n!! step limit ({max_steps}) reached - returning a structured 'incomplete' result")

    # The loop owns these three fields, not the model. The model cannot inflate its
    # own step count or quietly drop a security event it did not like.
    trace.elapsed_s = time.time() - started
    return trace


def report(trace: RunTrace) -> str:
    """Human-readable tail for the console; the JSON above it is the real product."""
    result = trace.result or {}
    lines = [
        "",
        "=" * 68,
        "STRUCTURED RESULT (validated against order_agent/schema.py)",
        "=" * 68,
        json.dumps(result, indent=2, ensure_ascii=False),
        "",
        f"steps used     : {trace.steps_used}/{MAX_STEPS}",
        f"tool calls     : {len(trace.tool_calls)} "
        f"({sum(1 for c in trace.tool_calls if not c['ok'])} returned an error)",
        f"stop reason    : {trace.stop_reason}",
        f"security events: {len(trace.security_events)}",
    ]
    for event in trace.security_events:
        lines.append(f"  - {event['type']} in {event['tool']}.{event['field']} "
                     f"({', '.join(event['patterns'])})")
    return "\n".join(lines)
