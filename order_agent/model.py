"""
Everything that knows about an LLM lives here.

The agent loop talks to a tiny interface - `generate(history) -> ModelTurn` - so I
can run the whole loop, the step limit, the schema validation and the injection
defence in unit tests with a scripted stand-in, and swap in Gemini for the real run
by changing one line. It also means no API key is needed to run the test suite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any]


@dataclass
class ModelTurn:
    """One assistant turn: either some tool calls, or a final text answer."""

    text: str = ""
    function_calls: list[FunctionCall] = field(default_factory=list)


class Model(Protocol):
    def generate(self, history: list[dict]) -> ModelTurn: ...


# --------------------------------------------------------------------------
# real model
# --------------------------------------------------------------------------

class GeminiModel:
    """Thin wrapper over google-genai function calling."""

    def __init__(self, system_prompt: str, declarations: list[dict],
                 model_name: str = DEFAULT_MODEL, api_key: str | None = None):
        # Imported lazily so the tests (and anyone without the SDK) can still import
        # this module.
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key. Put GOOGLE_API_KEY=... in your environment or .env file "
                "(see .env.example). The key is never committed."
            )

        self._types = types
        self._client = genai.Client(api_key=key)
        self._model_name = model_name
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,  # this is a bookkeeping task; creativity is a bug here
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _to_contents(self, history: list[dict]) -> list[Any]:
        types = self._types
        contents = []
        for entry in history:
            if entry["role"] == "user":
                contents.append(types.Content(role="user",
                                              parts=[types.Part(text=entry["text"])]))
            elif entry["role"] == "model":
                parts = []
                if entry.get("text"):
                    parts.append(types.Part(text=entry["text"]))
                for call in entry.get("function_calls", []):
                    parts.append(types.Part(function_call=types.FunctionCall(
                        name=call.name, args=call.args)))
                contents.append(types.Content(role="model", parts=parts))
            elif entry["role"] == "tool":
                parts = [
                    types.Part.from_function_response(name=r["name"], response=r["response"])
                    for r in entry["function_responses"]
                ]
                contents.append(types.Content(role="user", parts=parts))
        return contents

    def generate(self, history: list[dict]) -> ModelTurn:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=self._to_contents(history),
            config=self._config,
        )

        text_parts: list[str] = []
        calls: list[FunctionCall] = []
        candidates = response.candidates or []
        if candidates and candidates[0].content and candidates[0].content.parts:
            for part in candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    calls.append(FunctionCall(
                        name=part.function_call.name,
                        args=dict(part.function_call.args or {}),
                    ))
                elif getattr(part, "text", None):
                    text_parts.append(part.text)

        return ModelTurn(text="\n".join(text_parts).strip(), function_calls=calls)


# --------------------------------------------------------------------------
# test double
# --------------------------------------------------------------------------

class ScriptedModel:
    """
    Replays a fixed list of turns. Used ONLY by the tests and the offline demo.

    This is a stand-in for the model, not a stand-in for the agent: the loop, the
    tools, the validation, the step limit and the injection scanner all run for
    real against it. It exists so those things can be tested deterministically and
    without a network call.
    """

    def __init__(self, turns: list[Any]):
        # A turn is either a ModelTurn, or a callable(history) -> ModelTurn for the
        # rare case where the scripted answer has to depend on what a tool returned.
        self.turns = list(turns)
        self.calls_received = 0

    def generate(self, history: list[dict]) -> ModelTurn:
        self.calls_received += 1
        if not self.turns:
            # Behave like a model that keeps trying rather than one that stops: this
            # is what lets the step-limit test actually exercise the cap.
            return ModelTurn(function_calls=[FunctionCall("calculate", {"expression": "1+1"})])
        turn = self.turns.pop(0)
        return turn(history) if callable(turn) else turn
