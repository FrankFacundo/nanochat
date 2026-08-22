import unittest

from llm_lab.metrics import parse_metrics, summarize_metrics


class MetricParserTests(unittest.TestCase):
    def test_base_training_line(self):
        line = (
            "step 00100/05000 (2.00%) | loss: 6.372221 | lrm: 1.00 | "
            "dt: 372.01ms | tok/sec: 44,042 | bf16_mfu: 0.00"
        )
        metrics = {metric.name: metric for metric in parse_metrics(line)}
        self.assertEqual(metrics["train_loss"].step, 100)
        self.assertAlmostEqual(metrics["train_loss"].value, 6.372221)
        self.assertEqual(metrics["tokens_per_second"].value, 44042)

    def test_sft_training_line(self):
        line = "step 00200 (13.3%) | loss: 2.125000 | lrm: 0.80 | dt: 400.0ms | tok/sec: 40,960 | mfu: 0.00"
        metrics = {metric.name: metric for metric in parse_metrics(line)}
        self.assertEqual(metrics["train_loss"].step, 200)
        self.assertAlmostEqual(metrics["lr_multiplier"].value, 0.8)

    def test_validation_and_chatcore(self):
        validation = parse_metrics("Step 00100 | Validation bpb: 1.937422")
        self.assertEqual(validation[0].name, "val_bpb")
        chatcore = {metric.name: metric.value for metric in parse_metrics("Step 00600 | ChatCORE: 0.1234 | ChatCORE_cat: 0.2345")}
        self.assertEqual(chatcore, {"chatcore": 0.1234, "chatcore_categorical": 0.2345})

    def test_rl_metrics(self):
        metrics = {metric.name: metric.value for metric in parse_metrics("Step 5/100 | Average reward: 0.25 | Average sequence length: 84.50")}
        self.assertEqual(metrics, {"reward": 0.25, "sequence_length": 84.5})
        pass_metrics = {metric.name: metric.value for metric in parse_metrics("Step 5 | Pass@1: 0.1000, Pass@2: 0.2000")}
        self.assertEqual(pass_metrics, {"pass_at_1": 0.1, "pass_at_2": 0.2})

    def test_summary(self):
        summary = summarize_metrics(
            [
                {"name": "val_bpb", "step": 0, "value": 3.2},
                {"name": "val_bpb", "step": 100, "value": 1.9},
            ]
        )
        self.assertEqual(summary["val_bpb"]["last"], 1.9)
        self.assertEqual(summary["val_bpb"]["min"], 1.9)
        self.assertEqual(summary["val_bpb"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
