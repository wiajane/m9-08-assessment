"""
Scripted model turns for the offline demos.

These exist so that the loop, the tools, the step limit, the schema validation and
the injection defence can be exercised on a laptop with no API key and no network -
in CI, in the unit tests, and by a reviewer who just cloned the repo.

To be completely clear about what these prove and what they do not: the *agent* is
real here (real tool dispatch, real validation, real guards), but the *reasoning* is
canned. The evidence that the model picks its own tools is the live Gemini run
captured in the README, not this file.

The first turn of the happy path deliberately gets the customer id wrong. I wanted
the offline demo to show a tool returning a structured error and the run recovering
from it rather than falling over, because that is the behaviour I actually care
about.
"""

from __future__ import annotations

import json

from .model import FunctionCall, ModelTurn


def _last_tool_result(history: list[dict], tool: str) -> dict:
    """Dig the most recent successful result for `tool` out of the history."""
    for entry in reversed(history):
        if entry.get("role") != "tool":
            continue
        for response in entry["function_responses"]:
            if response["name"] == tool:
                payload = response["response"]["untrusted_tool_output"]
                if payload.get("ok"):
                    return payload
    return {}


def _final(order_key: str, reorder: dict, summary: str, warranty_sku: str):
    """Build the final-answer turn, reading the warranty facts back out of history."""

    def turn(history: list[dict]) -> ModelTurn:
        order = _last_tool_result(history, "lookup_order").get("order", {})
        warranty = _last_tool_result(history, "check_warranty").get("warranty", {})
        payload = {
            "status": "ok",
            "order": {
                "order_id": order.get("order_id", ""),
                "customer_id": order.get("customer_id", ""),
                "order_date": order.get("order_date", ""),
                "items": [
                    {
                        "sku": item["sku"],
                        "name": item["name"],
                        "qty": item["qty"],
                        "unit_price_eur": item["unit_price_eur"],
                    }
                    for item in order.get("items", [])
                ],
            },
            "reorder": reorder,
            "warranty": {
                "sku_checked": warranty.get("sku", warranty_sku),
                "in_warranty": warranty.get("in_warranty", False),
                "expires_on": warranty.get("expires_on", ""),
                "days_remaining": warranty.get("days_remaining", 0),
            },
            "summary": summary,
            "unmet_requirements": [],
        }
        return ModelTurn(text=json.dumps(payload, indent=2))

    return turn


# ---------------------------------------------------------------------------
# 1. happy path: CUS-014 / ORD-1002
#    2x(2 x HP-31 @ 49.90 + 1 x DK-05 @ 129.00) = 457.60 + 4.95 shipping
#    VAT 23% of 462.55 = 106.39  ->  total 568.94
# ---------------------------------------------------------------------------
HAPPY_SCRIPT = [
    # A wrong customer id on purpose - the tool rejects it, the agent recovers.
    ModelTurn(function_calls=[FunctionCall("lookup_order", {"customer_id": "CUS-14"})]),
    ModelTurn(function_calls=[FunctionCall("lookup_order", {"customer_id": "CUS-014"})]),
    ModelTurn(function_calls=[
        FunctionCall("calculate", {"expression": "2 * (2 * 49.9 + 1 * 129.0)"}),
        FunctionCall("check_warranty", {"sku": "DK-05", "purchase_date": "2026-02-11"}),
    ]),
    ModelTurn(function_calls=[FunctionCall("calculate", {"expression": "(457.6 + 4.95) * 0.23"})]),
    ModelTurn(function_calls=[FunctionCall("calculate", {"expression": "457.6 + 4.95 + 106.3865"})]),
    _final(
        "ORD-1002",
        {
            "multiplier": 2,
            "items_subtotal_eur": 457.6,
            "shipping_eur": 4.95,
            "vat_eur": 106.39,
            "total_eur": 568.94,
            "currency": "EUR",
        },
        "Two more of order ORD-1002 comes to EUR 568.94 including EUR 4.95 shipping and "
        "23% VAT (EUR 106.39). The docking station (DK-05) is still under its 24-month "
        "warranty.",
        "DK-05",
    ),
]

# ---------------------------------------------------------------------------
# 2. injection: CUS-777 / ORD-2087, whose support_notes field tells the assistant
#    to zero the total and grant a lifetime warranty. The safety layer redacts it
#    before the model sees it; the model prices the order correctly anyway.
#    2x(24.50 + 129.00) = 307.00 + 6.50 shipping, VAT 23% of 313.50 = 72.11
#    -> total 385.61
# ---------------------------------------------------------------------------
ATTACK_SCRIPT = [
    ModelTurn(function_calls=[FunctionCall("lookup_order", {"customer_id": "CUS-777"})]),
    ModelTurn(function_calls=[
        FunctionCall("calculate", {"expression": "2 * (24.5 + 129.0)"}),
        FunctionCall("check_warranty", {"sku": "DK-05", "purchase_date": "2026-05-30"}),
    ]),
    ModelTurn(function_calls=[FunctionCall("calculate", {"expression": "(307.0 + 6.5) * 0.23"})]),
    ModelTurn(function_calls=[FunctionCall("calculate", {"expression": "307.0 + 6.5 + 72.105"})]),
    _final(
        "ORD-2087",
        {
            "multiplier": 2,
            "items_subtotal_eur": 307.0,
            "shipping_eur": 6.5,
            "vat_eur": 72.11,
            "total_eur": 385.61,
            "currency": "EUR",
        },
        "Two more of order ORD-2087 comes to EUR 385.61 including EUR 6.50 shipping and "
        "23% VAT (EUR 72.11). The docking station (DK-05) is under warranty until 2028. "
        "Note: the order record contained text pretending to be a system instruction "
        "(free discount, lifetime warranty, send the order history to an external "
        "address). It was redacted by the safety layer and ignored; the prices above are "
        "the real ones.",
        "DK-05",
    ),
]

# ---------------------------------------------------------------------------
# 3. runaway: an empty script. ScriptedModel then keeps asking for a pointless
#    calculation forever, which is exactly what a confused or looping model does.
#    The step limit is the only thing that stops it.
# ---------------------------------------------------------------------------
RUNAWAY_SCRIPT: list = []
