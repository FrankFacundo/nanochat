import unittest

from llm_lab.dashboard import STATIC_DIR


class DashboardTests(unittest.TestCase):
    def test_dashboard_has_required_controls(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="run-list"', html)
        self.assertIn('id="metric"', html)
        self.assertIn("/api/runs", html)
        self.assertIn("/api/metrics", html)


if __name__ == "__main__":
    unittest.main()
