import unittest

from llm_lab.catalog import Catalog
from llm_lab.runner import build_commands


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog()

    def test_requested_ablation_topics_are_covered(self):
        expected = {
            "learning-rate", "warmup", "schedule", "optimizer", "weight-decay", "gradient-clipping",
            "batch-size", "training-tokens", "depth-width", "attention-heads", "mlp-ratio", "context-length",
            "tokenizer-vocab", "tied-embeddings", "norm-placement", "activation", "rope", "data-mixture",
        }
        actual = {experiment.topic for experiment in self.catalog.all()}
        self.assertEqual(expected - actual, set())

    def test_ids_are_unique(self):
        ids = [experiment.id for experiment in self.catalog.all()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_smoke_command_is_safe_and_isolated(self):
        experiment = self.catalog.get("w0-smoke")
        command = build_commands(experiment, self.catalog.defaults, wandb=False)[0]
        rendered = " ".join(command)
        self.assertIn("scripts.base_train", rendered)
        self.assertIn("--model-tag lab", rendered)
        self.assertIn("--run dummy", rendered)
        self.assertIn("--num-iterations 30", rendered)

    def test_wandb_run_is_enabled_only_on_request(self):
        experiment = self.catalog.get("w1-pilot")
        command = build_commands(experiment, self.catalog.defaults, wandb=True)[0]
        self.assertEqual(command[-1], experiment.id)


if __name__ == "__main__":
    unittest.main()
