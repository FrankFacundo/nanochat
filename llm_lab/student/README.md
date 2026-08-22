# Student workspace

This directory is yours. Commit predictions before launching experiments and conclusions after inspecting results.

- `predictions.json`: pre-registered hypotheses and falsifiers.
- `conclusions.json`: evidence-calibrated conclusions tied to execution IDs.
- `final-report.md`: capstone report.

Generated comparison memos may also be written here:

```bash
python -m llm_lab compare RUN_A RUN_B --output llm_lab/student/lr-comparison.md
```

Do not edit captured manifests or metrics. If parsing is wrong, fix the parser and regenerate a derived summary while retaining the original stdout.
