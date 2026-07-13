import unittest

from order_agent.tools import call_tool


class TestLookupOrder(unittest.TestCase):
    def test_finds_order_by_id(self):
        result = call_tool("lookup_order", {"order_id": "ORD-1002"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["order"]["customer_id"], "CUS-014")
        self.assertEqual(result["order"]["items_total_eur"], 228.8)

    def test_last_order_for_customer_is_the_most_recent_one(self):
        result = call_tool("lookup_order", {"customer_id": "CUS-014"})
        # CUS-014 has ORD-1001 (2024) and ORD-1002 (2026); "last" must mean 2026.
        self.assertEqual(result["order"]["order_id"], "ORD-1002")

    def test_rejects_both_arguments_at_once(self):
        result = call_tool("lookup_order", {"order_id": "ORD-1002", "customer_id": "CUS-014"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "INVALID_ARGUMENTS")

    def test_rejects_no_arguments(self):
        self.assertEqual(call_tool("lookup_order", {})["error"], "INVALID_ARGUMENTS")

    def test_rejects_malformed_id_before_touching_the_data(self):
        result = call_tool("lookup_order", {"order_id": "the last one"})
        self.assertEqual(result["error"], "INVALID_ORDER_ID")
        self.assertIn("ORD-1002", result["hint"])

    def test_missing_order_is_an_error_not_a_crash(self):
        self.assertEqual(call_tool("lookup_order", {"order_id": "ORD-9999"})["error"],
                         "ORDER_NOT_FOUND")

    def test_unexpected_argument_is_rejected(self):
        result = call_tool("lookup_order", {"order_id": "ORD-1002", "discount": "100%"})
        self.assertEqual(result["error"], "UNEXPECTED_ARGUMENTS")


class TestCheckWarranty(unittest.TestCase):
    def test_in_warranty(self):
        result = call_tool("check_warranty",
                           {"sku": "DK-05", "purchase_date": "2026-02-11"},
                           today="2026-07-13")
        warranty = result["warranty"]
        self.assertTrue(warranty["in_warranty"])
        self.assertEqual(warranty["policy_months"], 24)
        self.assertEqual(warranty["expires_on"], "2028-02-11")

    def test_out_of_warranty(self):
        result = call_tool("check_warranty",
                           {"sku": "MS-12", "purchase_date": "2024-01-10"},
                           today="2026-07-13")
        self.assertFalse(result["warranty"]["in_warranty"])
        self.assertLess(result["warranty"]["days_remaining"], 0)

    def test_unknown_sku_falls_back_to_the_default_policy(self):
        result = call_tool("check_warranty",
                           {"sku": "ZZ-99", "purchase_date": "2026-01-01"},
                           today="2026-07-13")
        self.assertEqual(result["warranty"]["policy_months"], 12)

    def test_rejects_a_hallucinated_date(self):
        result = call_tool("check_warranty", {"sku": "DK-05", "purchase_date": "last February"})
        self.assertEqual(result["error"], "INVALID_DATE")


class TestCalculate(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(call_tool("calculate", {"expression": "2 * (2*49.9 + 129.0)"})["result"],
                         457.6)

    def test_refuses_code_execution(self):
        # The reason calculate() does not use eval(): this string is "arithmetic".
        payload = "__import__('os').system('echo pwned')"
        result = call_tool("calculate", {"expression": payload})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "UNSAFE_EXPRESSION")

    def test_refuses_names_and_attributes(self):
        for payload in ["open('/etc/passwd').read()", "().__class__", "price * 2"]:
            with self.subTest(payload=payload):
                self.assertFalse(call_tool("calculate", {"expression": payload})["ok"])

    def test_refuses_a_denial_of_service_exponent(self):
        self.assertEqual(call_tool("calculate", {"expression": "9**9**9"})["error"],
                         "UNSAFE_EXPRESSION")

    def test_division_by_zero_is_reported_not_raised(self):
        self.assertEqual(call_tool("calculate", {"expression": "1/0"})["error"],
                         "DIVISION_BY_ZERO")

    def test_length_cap(self):
        self.assertEqual(call_tool("calculate", {"expression": "1+" * 200 + "1"})["error"],
                         "EXPRESSION_TOO_LONG")


class TestDispatcher(unittest.TestCase):
    def test_unknown_tool_does_not_raise(self):
        result = call_tool("issue_refund", {"amount": 9999})
        self.assertEqual(result["error"], "UNKNOWN_TOOL")
        self.assertIn("lookup_order", result["hint"])


if __name__ == "__main__":
    unittest.main()
