# microTabICLv2

A compact, trainable implementation of the
[TabICLv2](https://arxiv.org/abs/2602.11139) architecture that runs locally on
Apple Silicon or CPU and scales to larger CUDA training jobs on
[Modal](https://modal.com/).

Like [microTabPFN](https://github.com/jxucoder/microtabpfn), this project is for
learning and experimentation. It trains from scratch on an on-the-fly synthetic
prior. It is **not** a repackaging of the official pretrained TabICLv2 weights,
and a short laptop run should not be expected to match the paper's benchmark
results.

## What is preserved from TabICLv2

- repeated feature grouping with circular shifts `(0, 1, 3)`;
- target-aware embeddings before column compression;
- induced Set-Transformer attention over rows within each feature;
- feature-wise row interaction, RoPE, and learnable CLS tokens;
- dataset-wise in-context learning where test rows attend only to train rows;
- query-aware scalable softmax (QASSMax);
- classification and quantile-regression checkpoints;
- a diverse directed-graph prior and optional Muon training.

The local model and prior are intentionally smaller. See
[docs/RESEARCH.md](docs/RESEARCH.md) for the exact paper/code/PR findings that
guided the implementation.

## Run on a Mac

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --extra dev
uv run microtabiclv2 info
uv run microtabiclv2 demo --steps 50 --device auto
```

`--device auto` prefers CUDA, then Apple MPS, then CPU. Intel Macs use their
last wheel-supported PyTorch release (2.2.2); Apple Silicon and CUDA machines
use a current PyTorch release. Local training defaults
to float32. MPS AMP can roughly halve activation memory, but is opt-in because
the current upstream testing found little speed benefit on a real M4:

```bash
uv run microtabiclv2 train \
  --profile micro \
  --task classification \
  --steps 500 \
  --device mps \
  --amp \
  --output checkpoints/microtabiclv2-clf.pt
```

For the most conservative Mac path use `--device cpu --no-amp`. The model also
avoids the biased rank-3 `F.linear` path implicated in a current PyTorch MPS bug.

## Make it bigger with Modal

Install and authenticate Modal once:

```bash
uv sync --extra modal
uv run modal setup
```

Run the small profile on an L40S/A10/T4 fallback pool:

```bash
uv run modal run modal_app.py \
  --profile small \
  --task classification \
  --steps 5000 \
  --run-name small-clf
```

Run the paper-size architecture on H100/A100-80GB, detached for a long job:

```bash
uv run modal run --detach modal_app.py \
  --profile paper \
  --task regression \
  --run-name paper-reg
```

The `large` profile increases the paper architecture to width 192 and
`4/4/18` column/row/ICL blocks. This is a research scaling option, not a claim
that it improves accuracy—the TabICLv2 paper reports only marginal gains from a
deeper ablation. Checkpoints persist in the Modal Volume
`microtabiclv2-checkpoints`; the command prints the exact download instruction.

## Profiles

| Profile | Width | Blocks (column/row/ICL) | Inducing tokens | Intended hardware |
|---|---:|---:|---:|---|
| `micro` | 48 | 1 / 1 / 2 | 16 | Mac / CPU |
| `small` | 64 | 2 / 2 / 4 | 32 | larger Mac / Modal value GPU |
| `paper` | 128 | 3 / 3 / 12 | 128 | Modal H100 / A100-80GB |
| `large` | 192 | 4 / 4 / 18 | 192 | Modal H100 / A100-80GB |

The `paper` architecture matches Appendix A.4, including 4 row CLS tokens and
999 regression quantiles. Its default training schedule is still a practical
single-GPU approximation with physical batch size 4; the published run used
three stages totaling 550K steps, batch size 64 with micro-batching, contexts up
to 60K rows, and about 24.5 H100 GPU-days.

## Python API

```python
from microtabiclv2 import (
    MicroTabICLv2Classifier,
    get_model_config,
    get_train_config,
    train_model,
)

result = train_model(
    get_model_config("micro", task="classification"),
    get_train_config("micro", steps=100, output="checkpoints/example.pt"),
)

classifier = MicroTabICLv2Classifier(result.model)
classifier.fit(X_train, y_train)
y_probability = classifier.predict_proba(X_test)
```

`fit` stores the in-context examples; it does not update model weights. The
gradient-based work happens once during synthetic pretraining.

## Code map

The implementation is deliberately small enough to read in order:

1. [`prior.py`](src/microtabiclv2/prior.py) generates synthetic tabular tasks.
2. [`model.py`](src/microtabiclv2/model.py) contains the complete compact
   TabICLv2 architecture.
3. [`train.py`](src/microtabiclv2/train.py) handles loss, optimization, and
   checkpoints.
4. [`estimator.py`](src/microtabiclv2/estimator.py) provides the scikit-learn
   style in-context API.
5. [`proof.py`](src/microtabiclv2/proof.py) implements the two falsifiable
   learning demonstrations below.
6. [`modal_app.py`](modal_app.py) moves the same training code to larger GPUs.

## Tests

```bash
uv run pytest
uv run ruff check .
```

Evaluate a trained classifier on five datasets and compare it with logistic
regression and random forest using identical training splits:

```bash
uv run microtabiclv2 eval \
  --checkpoint checkpoints/microtabiclv2-clf.pt \
  --device auto
```

## The smallest convincing proof

One score can be a coincidence. This intervention keeps the checkpoint,
feature coordinates, and test grid fixed, then changes only the labels in the
context. The resulting SVG shows the same frozen model producing vertical,
horizontal, and diagonal decision rules. It also repeats the test on random
rotations with wrong-task, shuffled-label, and random-weight controls:

```bash
uv run microtabiclv2 proof \
  --checkpoint checkpoints/microtabiclv2-clf.pt \
  --device auto
```

For a stronger run, use `--repeats 100`. The default 20 keeps the educational
demo quick.

<img src="docs/assets/rule-switching.svg" width="760" alt="Four decision surfaces from one frozen model, followed by randomized controls">

To see whether training—not architecture luck—causes the improvement, train
once and measure binary-Iris AUC at fixed intervals:

```bash
uv run microtabiclv2 auc-curve \
  --steps 500 \
  --every 50 \
  --device auto
```

This writes `evaluations/auc-vs-steps.svg`, the matching CSV, and a checkpoint.
The figure includes a shuffled-context curve; genuine in-context learning
should improve with correct labels while shuffled labels remain near chance.
The control averages ten label permutations per split to avoid a lucky shuffle.
Both figures are plain SVG generated without Matplotlib or another plotting
dependency.

<img src="docs/assets/auc-vs-steps.svg" width="760" alt="Correct-label and shuffled-label ROC AUC during 500 training steps">

## Attribution

The architecture is based on TabICLv2 by Qu, Holzmüller, Varoquaux, and Le
Morvan and was cross-checked against the BSD-licensed
[official TabICL code](https://github.com/soda-inria/tabicl) and
[NanoTabICL](https://github.com/soda-inria/nanotabicl). This repository uses the
BSD 3-Clause license.
