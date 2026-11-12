"""Security guards that must hold without databases running."""
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import gemini


class CrossSiteGuardTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)  # no `with`: lifespan (database init) is not started

    def test_foreign_origin_write_is_refused(self):
        response = self.client.post("/api/ask", json={"document_id": 1, "question": "x"},
                                    headers={"Origin": "https://evil.example"})
        self.assertEqual(response.status_code, 403)

    def test_foreign_origin_delete_is_refused(self):
        response = self.client.delete("/api/documents/1", headers={"Origin": "http://localhost:9999"})
        self.assertEqual(response.status_code, 403)


class CropContainmentTests(unittest.TestCase):
    def test_file_outside_crops_dir_is_not_sent_to_gemini(self):
        outside = Path(__file__).resolve()
        with self.assertRaises(OSError):
            gemini.image_part(outside)

    def test_traversal_out_of_crops_dir_is_refused(self):
        sneaky = settings.crops_dir / ".." / ".." / "backend" / "requirements.txt"
        with self.assertRaises(OSError):
            gemini.image_part(sneaky)


if __name__ == "__main__":
    unittest.main()
