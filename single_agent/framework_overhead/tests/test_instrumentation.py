"""Tests for overhead instrumentation module."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from single_agent.framework_overhead.instrumentation import (
    MeasureResult,
    install_patches,
    measure,
    overlap_flag,
    residual_ms,
)


class TestInstrumentationHelpers(unittest.TestCase):
    def test_overlap_flag_serial(self):
        m = MeasureResult(llm_calls=1, llm_time_s=0.5, max_concurrency=1)
        self.assertFalse(overlap_flag(m, 600.0))

    def test_overlap_flag_concurrent(self):
        m = MeasureResult(llm_calls=2, llm_time_s=0.5, max_concurrency=2)
        self.assertTrue(overlap_flag(m, 600.0))

    def test_overlap_flag_exceeds_total(self):
        m = MeasureResult(llm_calls=1, llm_time_s=1.0, max_concurrency=1)
        self.assertTrue(overlap_flag(m, 500.0))

    def test_residual_not_floored_negative(self):
        m = MeasureResult(llm_calls=1, llm_time_s=1.0, max_concurrency=1)
        res, valid = residual_ms(m, 500.0)
        self.assertLess(res, 0)  # not silently max(0, ...)
        self.assertFalse(valid)  # overlap_flag because llm_ms > total_ms

    def test_residual_span_fallback_on_overlap(self):
        m = MeasureResult(
            llm_calls=2,
            llm_time_s=2.0,
            wall_llm_span_s=0.8,
            max_concurrency=2,
        )
        res, valid = residual_ms(m, 1000.0)
        self.assertFalse(valid)
        self.assertAlmostEqual(res, 200.0)  # 1000 - 800


class TestPatchIdempotency(unittest.TestCase):
    def test_install_patches_idempotent(self):
        install_patches()
        from openai.resources.chat.completions import Completions

        first = Completions.create
        install_patches()
        second = Completions.create
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
