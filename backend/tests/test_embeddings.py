import unittest
from unittest.mock import patch

import torch
from PIL import Image

from app.services import embeddings


class BatchedEmbeddingTests(unittest.TestCase):
    def test_regions_are_batched_and_returned_in_original_order(self):
        image = Image.new("RGB", (8, 8), "white")
        text_calls: list[list[str]] = []
        image_calls: list[int] = []

        def fake_text(texts: list[str]) -> torch.Tensor:
            text_calls.append(texts)
            return torch.tensor([[1.0, 0.0] for _ in texts])

        def fake_images(images: list[Image.Image]) -> torch.Tensor:
            image_calls.append(len(images))
            return torch.tensor([[0.0, 1.0] for _ in images])

        with (
            patch.object(embeddings, "embed_text", side_effect=fake_text),
            patch.object(embeddings, "embed_images", side_effect=fake_images),
        ):
            vectors = embeddings.embed_regions(
                [(None, "first"), (image, "second"), (None, "third")],
                batch_size=2,
            )

        self.assertEqual(text_calls, [["first", "second"], ["third"]])
        self.assertEqual(image_calls, [1])
        self.assertEqual(vectors[0], [1.0, 0.0])
        self.assertAlmostEqual(vectors[1][0], 2 ** -0.5, places=6)
        self.assertAlmostEqual(vectors[1][1], 2 ** -0.5, places=6)
        self.assertEqual(vectors[2], [1.0, 0.0])
        image.close()


if __name__ == "__main__":
    unittest.main()
