# Ablation matrix

The control for short pretraining studies is `w1-pilot`. The confirmation control is `w1-baseline`. “Native” means the current nanochat CLI can run the study. “Contract” means the implementation is part of the homework.

| # | Topic | Intervention | Primary measurement | Required controls | Main confound | Status |
|---:|---|---|---|---|---|---|
| 1 | Learning rate | 0.5× / 1× / 2× every parameter-group LR | BPB curve and instability | Same data, batch, schedule, init | A short run may unfairly favor high LR | Native |
| 2 | Warmup | 0 / 40 / 200 steps | Early loss and final BPB | Same peak LR and horizon | Long warmup changes effective LR budget | Native |
| 3 | Schedule | Constant / linear warmdown / cosine | Anytime BPB and final BPB | Same warmup, peak/final LR, steps | Different integral of LR over time | Contract |
| 4 | Optimizer | All-AdamW / hybrid Muon–AdamW | BPB per step and minute; memory | Fair group-specific LR tuning | One optimizer may be better tuned | Contract |
| 5 | Weight decay | 0 / 0.28 / 0.56 nominal | Train–validation gap, BPB | Same automatic scaling policy | nanochat rescales decay for depth/horizon | Native |
| 6 | Gradient clipping | Disabled / measured thresholds | Instability, clipped fraction, BPB | Threshold from control norm distribution | Clipping can hide an excessive LR | Contract |
| 7 | Batch tokens | 8K / 16K / 32K | BPB per token, step, minute | Explicit fixed-token or fixed-step claim | nanochat automatically rescales LR | Native |
| 8 | Tokens trained | 20.5M / 81.9M / 163.8M | BPB versus tokens and minutes | Same architecture and data order | LR schedule changes with horizon | Native |
| 9 | Depth vs width | d8×384 / d6×384 / d6×512 | BPB versus FLOPs and parameters | Fixed-token and fixed-FLOP views | Value embeddings make total count unusual | Native |
| 10 | Attention heads | 3 / 4 / 6 at width 384 | BPB and throughput | Fixed width/depth/context | Head dimension changes RoPE frequencies | Native |
| 11 | MLP ratio | 2 / 4 / 8 | BPB versus params/FLOPs | Raw and compute-matched comparisons | Capacity and compute both change | Contract |
| 12 | Context | 256 / 512 / 1024 | BPB and tokens/s | Fixed total tokens/update | Different sequence packing/distribution | Native |
| 13 | Vocabulary | 8K / 16K / 32K | BPB, bytes/token, parameters | Same tokenizer corpus; report token budget in bytes too | Token counts cease to be comparable | Native |
| 14 | Tied embeddings | Tied / untied | BPB and parameter efficiency | Correct initialization and optimizer grouping | Tying removes parameters and changes prior | Contract |
| 15 | RMSNorm placement | Pre / post / pre+embedding | Stability and BPB | Identical init/data/LR | Residual equations differ materially | Contract |
| 16 | Activation | ReLU² / GELU / SwiGLU | BPB per parameter and FLOP | Raw and parameter-matched widths | SwiGLU needs two input projections | Contract |
| 17 | RoPE | Base and rotated fraction | In-context and extrapolation BPB | Fixed context during training | Head dimension alters frequencies | Contract |
| 18 | Data quality/mixture | Fixed manifests and weights | General + domain held-out metrics | Frozen validation and token budget | Contamination and source duplication | Contract |

## Minimum acceptable portfolio

Complete all native topics and at least four contracts. Select three effects—at least one optimization and one architecture/data effect—for confirmation. At least two portfolio entries should be honest null or negative findings.

For each row, include:

- Pre-registered directional hypothesis.
- Exact execution IDs.
- Parameter count, estimated FLOPs/token, total tokens, and wall time.
- Learning curves, not only final values.
- Effect size relative to control.
- One alternative explanation.
- A falsifying follow-up.

## Comparison rules

Use fixed steps only to study optimizer dynamics at identical update count. Use fixed tokens to study data efficiency. Use fixed wall time to study practical hardware efficiency. Use fixed FLOPs when making architecture-efficiency claims. A single run pair rarely supports all four claims.
