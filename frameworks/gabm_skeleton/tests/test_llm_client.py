"""Tests for instrumented LLM client (stub + optional live API)."""

import os
import unittest

from frameworks.gabm_skeleton.llm_client import InstrumentedOpenAIClient, StubLLMClient
from frameworks.gabm_skeleton.metrics import Metrics


class TestStubLLMClient(unittest.TestCase):
    def test_records_fixed_token_counts(self):
        metrics = Metrics()
        client = StubLLMClient(
            responses=['{"answer": "4"}'],
            input_tokens=12,
            output_tokens=3,
            metrics=metrics,
        )
        client.complete("test prompt")
        self.assertEqual(metrics.llm_calls, 1)
        self.assertEqual(metrics.input_tokens, 12)
        self.assertEqual(metrics.output_tokens, 3)

    def test_reset_for_run_clears_accumulated_metrics(self):
        metrics = Metrics()
        client = StubLLMClient(responses=['{"answer": "4"}'], metrics=metrics)
        client.complete("first")
        self.assertEqual(metrics.llm_calls, 1)

        client.reset_for_run()
        self.assertEqual(metrics.llm_calls, 0)
        self.assertEqual(client._index, 0)


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY required for live API test")
class TestInstrumentedOpenAIClientLive(unittest.TestCase):
    def test_real_call_captures_tokens_and_latency(self):
        metrics = Metrics()
        client = InstrumentedOpenAIClient(model="gpt-4o-mini", metrics=metrics)
        text = client.complete(
            'Answer the math query. Return JSON: {"answer": "4"}',
            max_tokens=16,
            temperature=0,
        )
        self.assertTrue(text)
        self.assertEqual(metrics.llm_calls, 1)
        self.assertGreater(metrics.input_tokens, 0, "input_tokens must be captured from response.usage")
        self.assertGreater(metrics.output_tokens, 0, "output_tokens must be captured from response.usage")
        self.assertGreater(metrics.llm_time_s, 0.01, "llm_time_s should reflect real network latency")


if __name__ == "__main__":
    unittest.main()
