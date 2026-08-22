import unittest

from llm_lab.dashboard import STATIC_DIR


class DashboardTests(unittest.TestCase):
    def test_dashboard_has_required_controls(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="challenge-card"', html)
        self.assertIn('id="stats-grid"', html)
        self.assertIn('id="metric-select"', html)
        self.assertIn("/api/learning", html)
        self.assertIn("/api/evaluate", html)
        self.assertIn("/api/metrics", html)
        self.assertNotIn("ANTHROPIC_API_KEY =", html)


if __name__ == "__main__":
    unittest.main()
