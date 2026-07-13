"""
The structured result contract.

The agent's last turn must be a single JSON object matching `RESULT_SCHEMA`. It is
validated here, and if validation fails the loop gets one chance to send the error
back to the model and ask for a repair (see loop.py). Only a result that passes
validation is ever printed as the answer - so downstream code can trust the shape.

I wrote the validator by hand rather than reaching for pydantic. It is ~60 lines,
it has no install step, and the error messages are phrased as instructions for a
model rather than for a Python traceback ("field 'total_eur': expected number, got
string"), which is what makes the repair turn work.
"""

from __future__ import annotations

import json
from typing import Any

NUMBER = ("number",)
STRING = ("string",)
BOOLEAN = ("boolean",)

# type, required?  -- nested dicts describe nested objects.
RESULT_SCHEMA: dict[str, Any] = {
    "status": {"type": "string", "enum": ["ok", "incomplete", "failed"], "required": True},
    "order": {
        "type": "object",
        "required": True,
        "fields": {
            "order_id": {"type": "string", "required": True},
            "customer_id": {"type": "string", "required": True},
            "order_date": {"type": "string", "required": True},
            "items": {
                "type": "array",
                "required": True,
                "items": {
                    "sku": {"type": "string", "required": True},
                    "name": {"type": "string", "required": True},
                    "qty": {"type": "number", "required": True},
                    "unit_price_eur": {"type": "number", "required": True},
                },
            },
        },
    },
    "reorder": {
        "type": "object",
        "required": True,
        "fields": {
            "multiplier": {"type": "number", "required": True},
            "items_subtotal_eur": {"type": "number", "required": True},
            "shipping_eur": {"type": "number", "required": True},
            "vat_eur": {"type": "number", "required": True},
            "total_eur": {"type": "number", "required": True},
            "currency": {"type": "string", "required": True},
        },
    },
    "warranty": {
        "type": "object",
        "required": True,
        "fields": {
            "sku_checked": {"type": "string", "required": True},
            "in_warranty": {"type": "boolean", "required": True},
            "expires_on": {"type": "string", "required": True},
            "days_remaining": {"type": "number", "required": True},
        },
    },
    "summary": {"type": "string", "required": True},
    "unmet_requirements": {"type": "array", "required": False, "items": None},
}

_PY_TYPES = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


class SchemaError(ValueError):
    """Carries a list of human/model-readable problems."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _check(value: Any, spec: dict, path: str, problems: list[str]) -> None:
    expected = spec["type"]
    python_type = _PY_TYPES[expected]

    # bool is a subclass of int in Python; don't let True pass as a number.
    if expected == "number" and isinstance(value, bool):
        problems.append(f"field '{path}': expected number, got boolean")
        return
    if not isinstance(value, python_type):
        problems.append(f"field '{path}': expected {expected}, got {type(value).__name__}")
        return

    if "enum" in spec and value not in spec["enum"]:
        problems.append(f"field '{path}': must be one of {spec['enum']}, got '{value}'")

    if expected == "object":
        _validate_object(value, spec["fields"], path, problems)

    if expected == "array" and spec.get("items"):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                problems.append(f"field '{path}[{index}]': expected object")
                continue
            _validate_object(item, spec["items"], f"{path}[{index}]", problems)


def _validate_object(obj: dict, fields: dict, path: str, problems: list[str]) -> None:
    for key, spec in fields.items():
        child = f"{path}.{key}" if path else key
        if key not in obj:
            if spec.get("required"):
                problems.append(f"field '{child}': required but missing")
            continue
        _check(obj[key], spec, child, problems)


def validate_result(payload: Any) -> dict:
    """Raise SchemaError if `payload` is not a valid final result; else return it."""
    if not isinstance(payload, dict):
        raise SchemaError([f"top level: expected a JSON object, got {type(payload).__name__}"])
    problems: list[str] = []
    _validate_object(payload, RESULT_SCHEMA, "", problems)
    if problems:
        raise SchemaError(problems)
    return payload


def parse_final_json(text: str) -> dict:
    """
    Pull the JSON object out of the model's final message.

    Models like wrapping JSON in ```json fences even when told not to, so strip
    those, then take the outermost {...}.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise SchemaError(["final message contained no JSON object"])
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaError([f"final message was not valid JSON: {exc.msg} (line {exc.lineno})"]) from exc
