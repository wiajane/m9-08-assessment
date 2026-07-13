"""
The three tools the agent can call, plus their argument validation.

Design rules I stuck to:

* A tool never raises. Ever. It returns {"ok": False, "error": ..., "hint": ...}.
  An exception would kill the run; a structured error is something the model can
  read and recover from, which is exactly what I want an agent to do (and the
  captured run shows it doing it).

* Arguments are validated *before* the tool body runs, against a schema declared
  next to the tool. The model is a text generator; it will eventually hand me
  `order_id="the last one"` or `qty=-3`, and I would rather catch that at the
  boundary than let it reach the data.

* `calculate` never sees `eval()`. It walks an AST with a whitelist of node
  types. `eval("__import__('os').system('rm -rf /')")` is a perfectly valid
  arithmetic-looking string to a naive calculator tool.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ORDER_ID_RE = re.compile(r"^ORD-\d{4}$")
CUSTOMER_ID_RE = re.compile(r"^CUS-\d{3}$")
SKU_RE = re.compile(r"^[A-Z]{2}-\d{2}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MAX_EXPRESSION_LENGTH = 120


class ToolError(Exception):
    """Raised only inside this module; the dispatcher turns it into a dict."""

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict:
        return {"ok": False, "error": self.code, "message": self.message, "hint": self.hint}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def _load(filename: str) -> dict:
    with open(DATA_DIR / filename, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# tool 1: lookup_order
# --------------------------------------------------------------------------

def lookup_order(order_id: str | None = None, customer_id: str | None = None) -> dict:
    """Fetch one order, either by its ID or the most recent one for a customer."""
    if bool(order_id) == bool(customer_id):
        raise ToolError(
            "INVALID_ARGUMENTS",
            "Provide exactly one of order_id or customer_id, not both and not neither.",
            hint="Example: lookup_order(customer_id='CUS-014')",
        )

    if order_id and not ORDER_ID_RE.match(order_id):
        raise ToolError(
            "INVALID_ORDER_ID",
            f"'{order_id}' is not a valid order id.",
            hint="Order ids look like ORD-1002.",
        )
    if customer_id and not CUSTOMER_ID_RE.match(customer_id):
        raise ToolError(
            "INVALID_CUSTOMER_ID",
            f"'{customer_id}' is not a valid customer id.",
            hint="Customer ids look like CUS-014.",
        )

    db = _load("orders.json")
    orders = db["orders"]

    if order_id:
        matches = [o for o in orders if o["order_id"] == order_id]
        if not matches:
            raise ToolError(
                "ORDER_NOT_FOUND",
                f"No order with id {order_id}.",
                hint="Try looking the customer up instead.",
            )
        order = matches[0]
    else:
        matches = [o for o in orders if o["customer_id"] == customer_id]
        if not matches:
            raise ToolError(
                "NO_ORDERS_FOR_CUSTOMER",
                f"Customer {customer_id} has no orders on file.",
                hint="Check the customer id with the user.",
            )
        # "my last order" -> the most recent by order_date.
        order = max(matches, key=lambda o: o["order_date"])

    order = dict(order)
    order["customer_name"] = db["customers"].get(order["customer_id"], "unknown")
    order["items_total_eur"] = round(
        sum(item["qty"] * item["unit_price_eur"] for item in order["items"]), 2
    )
    return {"ok": True, "order": order}


# --------------------------------------------------------------------------
# tool 2: check_warranty
# --------------------------------------------------------------------------

def _add_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    # Clamp for short months (31 Jan + 1 month -> 28/29 Feb).
    day = min(start.day, [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


def check_warranty(sku: str, purchase_date: str, today: str | None = None) -> dict:
    """Is this SKU, bought on this date, still covered?"""
    if not isinstance(sku, str) or not SKU_RE.match(sku):
        raise ToolError(
            "INVALID_SKU",
            f"'{sku}' is not a valid SKU.",
            hint="SKUs look like DK-05 and come from the order's items.",
        )
    if not isinstance(purchase_date, str) or not ISO_DATE_RE.match(purchase_date):
        raise ToolError(
            "INVALID_DATE",
            f"'{purchase_date}' is not an ISO date.",
            hint="Use the order_date from lookup_order, e.g. 2026-02-11.",
        )

    policy = _load("warranty_policy.json")
    entry = policy["skus"].get(sku)
    if entry is None:
        months = policy["default_months"]
        category = "unknown"
    else:
        months = entry["months"]
        category = entry["category"]

    try:
        bought = date.fromisoformat(purchase_date)
    except ValueError as exc:  # pragma: no cover - regex already caught the shape
        raise ToolError("INVALID_DATE", str(exc)) from exc

    now = date.fromisoformat(today) if today else date.today()
    expires = _add_months(bought, months)
    days_remaining = (expires - now).days

    return {
        "ok": True,
        "warranty": {
            "sku": sku,
            "category": category,
            "policy_months": months,
            "purchase_date": purchase_date,
            "expires_on": expires.isoformat(),
            "evaluated_on": now.isoformat(),
            "in_warranty": days_remaining >= 0,
            "days_remaining": days_remaining,
        },
    }


# --------------------------------------------------------------------------
# tool 3: calculate  (safe arithmetic, no eval)
# --------------------------------------------------------------------------

_ALLOWED_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("UNSAFE_EXPRESSION", "Only numbers are allowed in expressions.")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ToolError("DIVISION_BY_ZERO", "Division by zero.", hint="Check the denominator.")
        if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1e6):
            raise ToolError("UNSAFE_EXPRESSION", "Exponent too large; refusing to evaluate.")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ToolError(
        "UNSAFE_EXPRESSION",
        f"Expression contains a construct that is not plain arithmetic: {type(node).__name__}.",
        hint="Only + - * / % ** on numbers, with parentheses.",
    )


def calculate(expression: str) -> dict:
    """Evaluate a plain arithmetic expression. No names, no calls, no attributes."""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("INVALID_ARGUMENTS", "expression must be a non-empty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(
            "EXPRESSION_TOO_LONG",
            f"Expression is {len(expression)} chars; the limit is {MAX_EXPRESSION_LENGTH}.",
            hint="Break the sum into smaller calculate() calls.",
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError("SYNTAX_ERROR", f"Could not parse '{expression}': {exc.msg}") from exc

    value = _eval_node(tree)
    return {"ok": True, "expression": expression, "result": round(value, 4)}


# --------------------------------------------------------------------------
# registry + dispatcher
# --------------------------------------------------------------------------

TOOLS: dict[str, Callable[..., dict]] = {
    "lookup_order": lookup_order,
    "check_warranty": check_warranty,
    "calculate": calculate,
}

# Gemini function-declaration schemas. Kept next to the implementations on
# purpose: if I change an argument, the declaration is right there.
TOOL_DECLARATIONS = [
    {
        "name": "lookup_order",
        "description": (
            "Fetch a single order. Pass EITHER order_id (when the user names an order) "
            "OR customer_id (to get that customer's most recent order). Never both."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "e.g. ORD-1002"},
                "customer_id": {"type": "string", "description": "e.g. CUS-014"},
            },
        },
    },
    {
        "name": "check_warranty",
        "description": (
            "Check whether a SKU bought on a given date is still under warranty. "
            "Get both arguments from a previous lookup_order result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "e.g. DK-05"},
                "purchase_date": {"type": "string", "description": "ISO date, e.g. 2026-02-11"},
            },
            "required": ["sku", "purchase_date"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate one arithmetic expression and return the number. Use this for every "
            "sum, multiplication or percentage - do not do arithmetic in your head."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Plain arithmetic, e.g. '2 * (2*49.9 + 129.0)'",
                }
            },
            "required": ["expression"],
        },
    },
]


def call_tool(name: str, args: dict, today: str | None = None) -> dict:
    """
    Single entry point the agent loop uses. Guarantees: returns a dict, never raises.
    """
    func = TOOLS.get(name)
    if func is None:
        return {
            "ok": False,
            "error": "UNKNOWN_TOOL",
            "message": f"There is no tool called '{name}'.",
            "hint": f"Available tools: {', '.join(TOOLS)}",
        }

    if not isinstance(args, dict):
        return {"ok": False, "error": "INVALID_ARGUMENTS", "message": "Arguments must be an object."}

    # Reject unexpected keyword names before they hit the function signature.
    allowed = {
        "lookup_order": {"order_id", "customer_id"},
        "check_warranty": {"sku", "purchase_date"},
        "calculate": {"expression"},
    }[name]
    unexpected = set(args) - allowed
    if unexpected:
        return {
            "ok": False,
            "error": "UNEXPECTED_ARGUMENTS",
            "message": f"{name} does not accept: {', '.join(sorted(unexpected))}.",
            "hint": f"{name} accepts: {', '.join(sorted(allowed))}.",
        }

    if name == "check_warranty" and today:
        args = {**args, "today": today}

    try:
        return func(**args)
    except ToolError as exc:
        return exc.as_dict()
    except TypeError as exc:
        return {"ok": False, "error": "INVALID_ARGUMENTS", "message": str(exc)}
    except Exception as exc:  # last-resort net: a crashing tool must not kill the run
        return {"ok": False, "error": "TOOL_CRASHED", "message": f"{type(exc).__name__}: {exc}"}
