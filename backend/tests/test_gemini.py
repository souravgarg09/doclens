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


if __name__ == "__main__":
    unittest.main()
