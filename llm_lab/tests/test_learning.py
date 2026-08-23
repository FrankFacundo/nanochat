import tempfile
import unittest
from pathlib import Path

from llm_lab.learning import (
    LearningCatalog,
    LearningError,
    ProgressStore,
    collect_evidence,
    evaluate_submission,
    learning_snapshot,
)


class FakeGrader:
    def __init__(self, credit: float = 0.95):
        self.credit = credit

    def grade(self, activity, answer, evidence, reference_text, previous_attempt=None):
        return {
            "assessment": {
                "summary": "A rigorous answer with a small remaining gap.",
                "rubric_results": {
                    item["id"]: {
                        "points": item["points"],
                        "evidence": "Demonstrated in the submitted answer.",
                        "missing": "",
                    }
                    for item in activity["rubric"]
                },
                "skill_outcomes": {
                    skill_id: {
                        "credit": self.credit,
                        "evidence": "The answer applies the skill correctly.",
                        "next_move": "Apply it to a new case.",
                    }
                    for skill_id in activity["skills"]
                },
                "strengths": ["Controlled reasoning"],
                "gaps": ["Test one transfer case"],
                "next_actions": ["Answer the follow-up without notes"],
                "follow_up_question": "Which observation would falsify your mechanism?",
                "confidence": 0.91,
            },
            "model": "fake-opus",
            "effort": "test",
            "latency_seconds": 0.01,
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "raw_response": {"test": True},
        }


class LearningCatalogTests(unittest.TestCase):
    def test_catalog_covers_the_full_course(self):
        catalog = LearningCatalog()
        self.assertEqual(len(catalog.raw["weeks"]), 7)
        self.assertGreaterEqual(len(catalog.activities), 15)
        self.assertGreaterEqual(len(catalog.skills), 18)
        self.assertEqual(sum(catalog.raw["week_weights"].values()), 100)
        self.assertTrue(all(sum(item["points"] for item in activity["rubric"]) == 100 for activity in catalog.activities.values()))

    def test_empty_snapshot_starts_with_the_first_method_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = learning_snapshot(LearningCatalog(), ProgressStore(Path(tmp)))
        self.assertEqual(snapshot["next_activity_id"], "w0-method-defense")
        self.assertEqual(snapshot["stats"]["attempts"], 0)
        self.assertEqual(snapshot["activities"][0]["status"], "ready")
        self.assertEqual(snapshot["activities"][1]["status"], "locked")

    def test_first_activity_explains_the_exact_deliverable(self):
        activity = LearningCatalog().activity("w0-method-defense")
        self.assertEqual(activity["submission_type"], "Written response only")
        self.assertFalse(activity["evidence_required"])
        self.assertIn("do not need to run code", activity["deliverable"])
        self.assertEqual(len(activity["lesson"]), 4)
        self.assertIn("1. Variables", activity["prompt"])


class EvaluationTests(unittest.TestCase):
    def test_llm_attempt_is_persisted_and_unlocks_the_next_activity(self):
        catalog = LearningCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp))
            result = evaluate_submission(
                {
                    "activity_id": "w0-method-defense",
                    "answer": "A controlled answer with a quantitative falsifier and the correct budget axes.",
                    "evidence_paths": [],
                    "run_ids": [],
                },
                catalog,
                store,
                FakeGrader(),
            )
            attempts = store.attempts()
        self.assertEqual(result["attempt"]["score"], 100)
        self.assertEqual(result["attempt"]["verdict"], "mastery")
        self.assertEqual(len(attempts), 1)
        self.assertIn("raw_response", attempts[0])
        self.assertNotIn("raw_response", result["attempt"])
        second = next(item for item in result["learning"]["activities"] if item["id"] == "w0-artifact-audit")
        self.assertEqual(second["status"], "ready")

    def test_locked_activity_cannot_be_graded_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LearningError):
                evaluate_submission(
                    {
                        "activity_id": "w1-tokenizer-defense",
                        "answer": "Trying to skip the advancement gate.",
                        "evidence_paths": [],
                        "run_ids": [],
                    },
                    LearningCatalog(),
                    ProgressStore(Path(tmp)),
                    FakeGrader(),
                )

    def test_mastery_requires_repeated_strong_evidence(self):
        catalog = LearningCatalog()
        payload = {
            "activity_id": "w0-method-defense",
            "answer": "A complete transfer answer with controlled reasoning and a quantitative falsifier.",
            "evidence_paths": [],
            "run_ids": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp))
            first = evaluate_submission(payload, catalog, store, FakeGrader(1.0))["learning"]
            second = evaluate_submission(payload, catalog, store, FakeGrader(1.0))["learning"]
        first_skill = next(item for item in first["skills"] if item["id"] == "method.hypothesis")
        second_skill = next(item for item in second["skills"] if item["id"] == "method.hypothesis")
        self.assertLess(first_skill["mastery"], catalog.mastery_threshold)
        self.assertGreaterEqual(second_skill["mastery"], catalog.mastery_threshold)

    def test_workspace_evidence_is_hashed_and_outside_paths_are_rejected(self):
        evidence, metadata = collect_evidence(["llm_lab/README.md", "/etc/hosts"], [])
        attached = next(item for item in metadata if item["status"] == "attached")
        rejected = next(item for item in metadata if item["status"] == "outside_workspace")
        self.assertEqual(attached["label"], "llm_lab/README.md")
        self.assertEqual(len(attached["sha256"]), 64)
        self.assertEqual(rejected["label"], "/etc/hosts")
        self.assertEqual(len(evidence), 1)


if __name__ == "__main__":
    unittest.main()
