import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.config import settings
from app.services import gemini


class RateLimited(Exception):
    code = 429


class GeminiFallbackTests(unittest.TestCase):
    def test_rate_limit_uses_fallback_without_repeated_primary_calls(self) -> None:
        fake = Mock()
        fake.interactions.create.side_effect = [
            RateLimited("quota exhausted"),
            SimpleNamespace(output_text='{"answer":"ok"}'),
        ]
        with (
            patch.object(gemini, "client", return_value=fake),
            patch.object(settings, "gemini_model", "gemini-3.8-flash"),
            patch.object(settings, "gemini_fallback_model", "gemini-3.5-flash-lite"),
        ):
            schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
            self.assertEqual('{"answer":"ok"}', gemini.generate(["question"], schema=schema))

        self.assertEqual(2, fake.interactions.create.call_count)
        self.assertEqual(
            ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
            [call.kwargs["model"] for call in fake.interactions.create.call_args_list],
        )
        self.assertEqual(schema, fake.interactions.create.call_args.kwargs["response_format"]["schema"])

    def _run(self, side_effect, max_attempts=2):
        fake = Mock()
        fake.interactions.create.side_effect = side_effect
        with (
            patch.object(gemini, "client", return_value=fake),
            patch.object(gemini.time, "sleep"),
            patch.object(settings, "gemini_model", "primary"),
            patch.object(settings, "gemini_fallback_model", "fallback"),
            patch.object(settings, "gemini_max_attempts", max_attempts),
        ):
            try:
                return gemini.generate(["q"]), fake
            except gemini.GeminiUnavailable as exc:
                return exc, fake

    def test_overloaded_primary_is_retried_once_then_falls_back(self) -> None:
        result, fake = self._run([Overloaded("503"), Overloaded("503"), SimpleNamespace(output_text="ok")])
        self.assertEqual("ok", result)
        self.assertEqual(["primary", "primary", "fallback"],
                         [c.kwargs["model"] for c in fake.interactions.create.call_args_list])

    def test_timeout_counts_as_transient(self) -> None:
        result, _ = self._run([TimeoutError("read timed out"), SimpleNamespace(output_text="ok")])
        self.assertEqual("ok", result)

    def test_all_models_failing_raises_unavailable_after_bounded_attempts(self) -> None:
        result, fake = self._run([Overloaded("503")] * 10)
        self.assertIsInstance(result, gemini.GeminiUnavailable)
        self.assertEqual(4, fake.interactions.create.call_count)  # 2 per model, never more

    def test_permanent_error_is_not_retried(self) -> None:
        fake = Mock()
        fake.interactions.create.side_effect = BadRequest("400 invalid argument")
        with patch.object(gemini, "client", return_value=fake), self.assertRaises(BadRequest):
            gemini.generate(["q"])
        self.assertEqual(1, fake.interactions.create.call_count)


class HedgingTests(unittest.TestCase):
    def test_slow_request_is_raced_by_a_duplicate(self) -> None:
        import threading
        import time as _time
        release = threading.Event()
        calls = {"n": 0}
        lock = threading.Lock()

        def create(**_kwargs):
            with lock:
                calls["n"] += 1
                n = calls["n"]
            if n == 1:            # first request is stuck
                release.wait(5)
                return SimpleNamespace(output_text="slow")
            return SimpleNamespace(output_text="fast")

        fake = Mock()
        fake.interactions.create.side_effect = create
        with (
            patch.object(gemini, "client", return_value=fake),
            patch.object(settings, "gemini_hedge_after_s", 0.2),
        ):
            start = _time.perf_counter()
            result = gemini.generate(["q"])
            elapsed = _time.perf_counter() - start
        release.set()
        self.assertEqual("fast", result)
        self.assertLess(elapsed, 2)
        self.assertEqual(2, calls["n"])


class MissingModelTests(unittest.TestCase):
    def test_retired_primary_model_falls_back(self) -> None:
        fake = Mock()
        fake.interactions.create.side_effect = [NotFound("404 model not found"), SimpleNamespace(output_text="ok")]
        with (
            patch.object(gemini, "client", return_value=fake),
            patch.object(settings, "gemini_model", "retired-model"),
            patch.object(settings, "gemini_fallback_model", "fallback"),
        ):
            self.assertEqual("ok", gemini.generate(["q"]))
        self.assertEqual(["retired-model", "fallback"],
                         [c.kwargs["model"] for c in fake.interactions.create.call_args_list])


class FallbackChainTests(unittest.TestCase):
    def test_comma_separated_fallbacks_are_tried_in_order(self) -> None:
        fake = Mock()
        fake.interactions.create.side_effect = [RateLimited("429"), NotFound("404"), SimpleNamespace(output_text="ok")]
        with (
            patch.object(gemini, "client", return_value=fake),
            patch.object(settings, "gemini_model", "lite"),
            patch.object(settings, "gemini_fallback_model", "fast, stable"),
        ):
            self.assertEqual("ok", gemini.generate(["q"]))
        self.assertEqual(["lite", "fast", "stable"],
                         [c.kwargs["model"] for c in fake.interactions.create.call_args_list])


class NotFound(Exception):
    code = 404


class Overloaded(Exception):
    code = 503


class BadRequest(Exception):
    code = 400


if __name__ == "__main__":
    unittest.main()
