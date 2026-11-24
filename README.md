# b8g-micro-moe

Sub-millisecond, **53.7 KB INT8** offline multi-head AI classifiers and routers — zero Python, zero PyTorch, zero GPU.

## Measured Performance (CPU Only)
- **Model Size:** `53.74 KB` INT8 quantized binary.
- **Training Time:** `~235 ms` on a standard laptop CPU (`80` epochs).
- **Held-Out Accuracy:** `100.00%` across 5 simultaneous classification heads on 64 strictly disjoint unseen queries.
- **Inference Latency:** `0.682 ms` single-pass across all 5 heads.
- **Fail-Closed Gating:** Built-in Bloom unknown-token detector + confidence threshold (`escalateToLlm`) prevents silent hallucinations.
- **Live Learning:** Teach a new phrasing in `~3 ms` via `crystallizeSample()` without full retraining.

## Research Paper
See [PAPER.md](./PAPER.md) for the full formal formulation, architecture, and reproducibility protocol.

## Quick Start
```bash
node src/train-router.mts
node src/cli.mts "I want a split browser with frozen tabs and audio lease"
node --test src/algebra-router.test.mts
```

## License
MIT

## Quick Install & Run (Windows / Linux / macOS)

```sh
# Download latest release assets (all platforms)
gh release download -R awesome-os/b8g-micro-moe --dir ./b8g-micro-moe-release

# Or clone source directly (all platforms)
git clone https://github.com/awesome-os/b8g-micro-moe.git && cd b8g-micro-moe
```
