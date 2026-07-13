import json
import unittest

from order_agent.demo_scripts import ATTACK_SCRIPT, HAPPY_SCRIPT
from order_agent.loop import run_agent
from order_agent.model import FunctionCall, ModelTurn, ScriptedModel
from order_agent.schema import SchemaError, validate_result

TODAY = "2026-07-13"


class TestHappyPath(unittest.TestCase):
    def test_run_completes_with_a_validated_result(self):
        trace = run_agent("reorder x2 for CUS-014", ScriptedModel(HAPPY_SCRIPT),
                          today=TODAY, verbose=False)
        self.assertEqual(trace.stop_reason, "completed")
        self.assertEqual(trace.result["status"], "ok")
        validate_result(trace.result)  # raises if the contract is broken

    def test_all_three_tools_get_used(self):
        trace = run_agent("reorder x2 for CUS-014", ScriptedModel(HAPPY_SCRIPT),
                          today=TODAY, verbose=False)
        used = {call["tool"] for call in trace.tool_calls}
        self.assertEqual(used, {"lookup_order", "check_warranty", "calculate"})

    def test_a_failed_tool_call_does_not_end_the_run(self):
        trace = run_agent("reorder x2 for CUS-014", ScriptedModel(HAPPY_SCRIPT),
                          today=TODAY, verbose=False)
        failures = [call for call in trace.tool_calls if not call["ok"]]
        self.assertEqual(len(failures), 1)                      # the bad customer id
        self.assertEqual(trace.stop_reason, "completed")        # and it still finished

    def test_the_money_adds_up(self):
        trace = run_agent("reorder x2 for CUS-014", ScriptedModel(HAPPY_SCRIPT),
                          today=TODAY, verbose=False)
        reorder = trace.result["reorder"]
        expected = round((reorder["items_subtotal_eur"] + reorder["shipping_eur"]) * 1.23, 2)
        self.assertAlmostEqual(reorder["total_eur"], expected, places=2)


class TestSafetyInTheLoop(unittest.TestCase):
    def test_injection_is_recorded_and_the_answer_stays_honest(self):
        trace = run_agent("reorder x2 for CUS-777", ScriptedModel(ATTACK_SCRIPT),
                          today=TODAY, verbose=False)
        self.assertEqual(len(trace.security_events), 1)
        self.assertEqual(trace.security_events[0]["type"], "prompt_injection_blocked")
        self.assertEqual(trace.security_events[0]["tool"], "lookup_order")

        # The attack demanded a zero total and a lifetime warranty. It got neither.
        self.assertGreater(trace.result["reorder"]["total_eur"], 300)
        self.assertNotEqual(trace.result["warranty"]["expires_on"], "")

    def test_the_payload_never_reaches_the_model_context(self):
        model = ScriptedModel(ATTACK_SCRIPT)
        trace = run_agent("reorder x2 for CUS-777", model, today=TODAY, verbose=False)
        # Whatever we stored for the audit trail must already be redacted.
        blob = json.dumps(trace.tool_calls)
        self.assertNotIn("refunds@totally-legit.example", blob)
        self.assertIn("REDACTED", blob)


class TestBounds(unittest.TestCase):
    def test_step_limit_stops_a_looping_model(self):
        model = ScriptedModel([])  # never produces a final answer
        trace = run_agent("go forever", model, today=TODAY, max_steps=4, verbose=False)
        self.assertEqual(trace.stop_reason, "step_limit_reached")
        self.assertEqual(trace.steps_used, 4)
        self.assertEqual(model.calls_received, 4)  # it really did stop at the cap

    def test_a_bounded_run_still_returns_the_agreed_shape(self):
        trace = run_agent("go forever", ScriptedModel([]), today=TODAY, max_steps=3, verbose=False)
        validate_result(trace.result)  # the contract holds even on failure
        self.assertEqual(trace.result["status"], "incomplete")
        self.assertIn("step_limit_reached", trace.result["unmet_requirements"])


class TestOutputContract(unittest.TestCase):
    def test_a_malformed_final_answer_gets_one_repair_attempt(self):
        good = HAPPY_SCRIPT[-1]
        model = ScriptedModel([
            ModelTurn(function_calls=[FunctionCall("lookup_order", {"customer_id": "CUS-014"})]),
            ModelTurn(function_calls=[FunctionCall("check_warranty",
                                                   {"sku": "DK-05",
                                                    "purchase_date": "2026-02-11"})]),
            ModelTurn(text="Sure! Two more of that order costs about 570 euros. :)"),  # not JSON
            good,                                                                       # repaired
        ])
        trace = run_agent("reorder x2 for CUS-014", model, today=TODAY, verbose=False)
        self.assertEqual(trace.stop_reason, "completed")
        self.assertEqual(trace.result["status"], "ok")

    def test_an_unrepairable_answer_fails_gracefully(self):
        model = ScriptedModel([
            ModelTurn(text="nope, still not JSON"),
            ModelTurn(text="still not JSON, sorry"),
        ])
        trace = run_agent("reorder", model, today=TODAY, verbose=False)
        self.assertEqual(trace.stop_reason, "invalid_final_output")
        validate_result(trace.result)

    def test_schema_rejects_a_plausible_but_wrong_result(self):
        bad = {
            "status": "ok",
            "order": {"order_id": "ORD-1002", "customer_id": "CUS-014",
                      "order_date": "2026-02-11", "items": []},
            "reorder": {"multiplier": 2, "items_subtotal_eur": 457.6, "shipping_eur": 4.95,
                        "vat_eur": 106.39, "total_eur": "568.94", "currency": "EUR"},
            "warranty": {"sku_checked": "DK-05", "in_warranty": True,
                         "expires_on": "2028-02-11", "days_remaining": 578},
            "summary": "...",
        }
        with self.assertRaises(SchemaError) as ctx:
            validate_result(bad)  # total_eur is a string
        self.assertIn("reorder.total_eur", str(ctx.exception))


class TestModelFailure(unittest.TestCase):
    def test_a_dead_model_does_not_crash_the_run(self):
        class Broken:
            def generate(self, history):
                raise ConnectionError("429 quota exceeded")

        trace = run_agent("reorder", Broken(), today=TODAY, verbose=False)
        self.assertEqual(trace.stop_reason, "model_error")
        validate_result(trace.result)
        self.assertEqual(trace.result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
