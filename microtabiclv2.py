"""microTabICLv2: a one-file, educational TabICLv2 (~300 lines of core code).

It keeps the paper's main path—feature grouping, target-aware embeddings,
induced column attention, row aggregation, and dataset-wise ICL with QASSMax—
while using a tiny synthetic prior that can train on a Mac.

Run: uv run python microtabiclv2.py --steps 500
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Four sizes, one implementation. Only micro is intended for a laptop.
PRESETS = {
    "micro": dict(d=48, col=1, row=1, icl=2, inducing=16, cls=2, heads=4),
    "small": dict(d=64, col=2, row=2, icl=4, inducing=32, cls=4, heads=4),
    "paper": dict(d=128, col=3, row=3, icl=12, inducing=128, cls=4, heads=8),
    "large": dict(d=192, col=4, row=4, icl=18, inducing=192, cls=4, heads=8),
}


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Prior: correlated features + a random nonlinear decision function


def sample_prior(batch=4, n_train=64, n_test=24, n_features=8, n_classes=2, device="cpu"):
    """Create a batch of synthetic tabular classification tasks."""
    rows, device = n_train + n_test, torch.device(device)
    x = torch.randn(batch, rows, n_features, device=device)
    for feature in range(1, n_features):
        weights = torch.randn(batch, feature, 1, device=device) / feature**0.5
        parent = torch.bmm(x[:, :, :feature], weights).squeeze(-1)
        transform = random.choice((torch.tanh, torch.sin, lambda value: value.square().clamp(max=6)))
        x[:, :, feature] = 0.55 * x[:, :, feature] + transform(parent)

    hidden = 16
    w1 = torch.randn(batch, n_features, hidden, device=device) / n_features**0.5
    w2 = torch.randn(batch, hidden, 1, device=device) / hidden**0.5
    score = torch.bmm(torch.tanh(torch.bmm(x, w1)), w2).squeeze(-1)
    score += 0.15 * torch.randn_like(score)

    # Quantile bins keep every class represented; randomly flipping class order
    # prevents the model from learning one fixed output convention.
    order = score.argsort(dim=1)
    by_rank = torch.arange(rows, device=device) * n_classes // rows
    y = torch.empty_like(order).scatter(1, order, by_rank.expand(batch, -1))
    for task in range(batch):
        y[task] = torch.randperm(n_classes, device=device)[y[task]]
    return x, y[:, :n_train], y[:, n_train:]


# -----------------------------------------------------------------------------
# Model


class Linear(nn.Linear):
    """MPS-safe Linear: apply bias separately from rank-3+ matmul."""

    def forward(self, x):
        out = F.linear(x, self.weight, None)
        return out if self.bias is None else out + self.bias


def mlp(width, output=None):
    return nn.Sequential(Linear(width, width * 2), nn.GELU(), Linear(width * 2, output or width))


def rope(x):
    """Rotary feature positions used inside each table row."""
    half = x.shape[-1] // 2
    freq = 100_000 ** torch.linspace(0.0, -1.0, half + 1, device=x.device)[:-1]
    angle = torch.arange(x.shape[-2], device=x.device)[:, None] * freq[None]
    sin, cos = angle.sin().to(x.dtype), angle.cos().to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, first * sin + second * cos), -1)


class Attention(nn.Module):
    def __init__(self, d, heads, use_rope=False, qass=False):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.q, self.k, self.v, self.out = (Linear(d, d) for _ in range(4))
        self.use_rope, self.qass = use_rope, qass
        if qass:
            self.length_scale = mlp(1, d)
            self.query_scale = mlp(self.head_dim)
            nn.init.zeros_(self.query_scale[-1].weight)
            nn.init.zeros_(self.query_scale[-1].bias)

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    def forward(self, query, context):
        q, k, v = self.split(self.q(query)), self.split(self.k(context)), self.split(self.v(context))
        if self.qass:
            length = q.new_tensor([[math.log(max(k.shape[-2], 1))]])
            base = self.length_scale(length).reshape(1, self.heads, 1, self.head_dim)
            q = q * base * (1 + torch.tanh(self.query_scale(q)))
        if self.use_rope:
            q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(-3, -2).flatten(-2))


class Block(nn.Module):
    def __init__(self, d, heads, use_rope=False, qass=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention, self.ff = Attention(d, heads, use_rope, qass), mlp(d)

    def forward(self, query, context=None):
        q = self.norm1(query)
        context = q if context is None else self.norm1(context)
        hidden = query + self.attention(q, context)
        return hidden + self.ff(self.norm2(hidden))


class InducedBlock(nn.Module):
    """Set Transformer: rows -> inducing tokens -> rows."""

    def __init__(self, d, heads, tokens):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, tokens, d) * 0.02)
        self.read, self.write = Block(d, heads, qass=True), Block(d, heads)

    def forward(self, x, n_train):
        inducing = self.inducing.expand(x.shape[0], -1, -1)
        return self.write(x, self.read(inducing, x[:, :n_train]))


class MicroTabICLv2(nn.Module):
    """Compact TabICLv2. Inputs are (tasks, train+test rows, features)."""

    def __init__(self, d=48, col=1, row=1, icl=2, inducing=16, cls=2, heads=4, classes=6):
        super().__init__()
        self.config = dict(d=d, col=col, row=row, icl=icl, inducing=inducing, cls=cls, heads=heads)
        self.x_embed = Linear(3, d)
        self.y_col = nn.Embedding(classes, d)
        self.y_icl = nn.Embedding(classes, d * cls)
        self.columns = nn.ModuleList(InducedBlock(d, heads, inducing) for _ in range(col))
        self.cls = nn.Parameter(torch.randn(1, 1, cls, d) * 0.02)
        self.rows = nn.ModuleList(Block(d, heads, use_rope=True) for _ in range(row))
        self.icl = nn.ModuleList(Block(d * cls, heads, qass=True) for _ in range(icl))
        self.head = mlp(d * cls, classes)

    def forward(self, x, y_train):
        n_train, features = y_train.shape[1], x.shape[-1]
        mean = x[:, :n_train].mean(1, keepdim=True)
        std = x[:, :n_train].std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
        x = (x.float() - mean) / std

        index = torch.arange(features, device=x.device)
        grouped = torch.stack([x[..., (index + shift) % features] for shift in (0, 1, 3)], -1)
        hidden = self.x_embed(grouped)
        hidden[:, :n_train] += self.y_col(y_train).unsqueeze(2)

        batch, rows, _, width = hidden.shape
        hidden = hidden.transpose(1, 2).reshape(batch * features, rows, width)
        for block in self.columns:
            hidden = block(hidden, n_train)
        hidden = hidden.reshape(batch, features, rows, width).transpose(1, 2)

        cls = self.cls.expand(batch, rows, -1, -1)
        hidden = torch.cat((cls, hidden), 2).reshape(batch * rows, -1, width)
        for block in self.rows[:-1]:
            hidden = block(hidden)
        hidden = self.rows[-1](hidden[:, : self.cls.shape[2]], hidden)
        hidden = hidden.reshape(batch, rows, -1)
        hidden[:, :n_train] += self.y_icl(y_train)

        for block in self.icl[:-1]:
            hidden = block(hidden, hidden[:, :n_train])
        hidden = self.icl[-1](hidden[:, n_train:], hidden[:, :n_train])
        return self.head(hidden)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(profile="micro"):
    return MicroTabICLv2(**PRESETS[profile])


# -----------------------------------------------------------------------------
# Train, save, and use like a tiny sklearn classifier


def train(model, steps=500, batch=4, device="auto", evaluate=None, eval_every=50):
    device, model = get_device(device), model.to(get_device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    history, started = [], time.perf_counter()

    def measure(step):
        if evaluate and (step == 0 or step == steps or step % eval_every == 0):
            model.eval()
            history.append((step, *evaluate(model, str(device))))
            model.train()
            print(f"eval {step:>4} | AUC {history[-1][1]:.3f} | shuffled {history[-1][2]:.3f}")

    print(f"Training {model.parameter_count():,} parameters on {device} for {steps} steps")
    measure(0)
    model.train()
    for step in range(1, steps + 1):
        n_train, n_features, n_classes = random.randint(32, 96), random.randint(4, 12), random.randint(2, 6)
        x, y_train, y_test = sample_prior(batch, n_train, 24, n_features, n_classes, device)
        loss = F.cross_entropy(model(x, y_train)[..., :n_classes].flatten(0, 1), y_test.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % 50 == 0:
            print(f"step {step:>4}/{steps} | loss {loss.item():.4f}")
        measure(step)
    print(f"finished in {time.perf_counter() - started:.1f}s")
    return model.eval(), history


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state": model.state_dict()}, path)


def load(path, device="auto"):
    payload = torch.load(path, map_location=get_device(device), weights_only=True)
    model = MicroTabICLv2(**payload["config"])
    model.load_state_dict(payload["state"])
    return model.to(get_device(device)).eval()


class MicroTabICLv2Classifier:
    def __init__(self, model, device="auto"):
        self.model, self.device = model.to(get_device(device)).eval(), get_device(device)

    def fit(self, x, y):
        self.classes, encoded = np.unique(y, return_inverse=True)
        self.x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.y = torch.as_tensor(encoded, dtype=torch.long, device=self.device)
        return self

    def predict_proba(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        outputs = []
        with torch.no_grad():
            for chunk in x.split(256):
                table = torch.cat((self.x, chunk)).unsqueeze(0)
                logits = self.model(table, self.y.unsqueeze(0))[0, :, : len(self.classes)]
                outputs.append(logits.softmax(-1).cpu())
        return torch.cat(outputs).numpy()

    def predict(self, x):
        return self.classes[self.predict_proba(x).argmax(1)]


# -----------------------------------------------------------------------------
# Two falsifiable checks: AUC while training and rule switching at inference


def iris_auc(model, device="auto", seeds=5, shuffles=10):
    from sklearn.datasets import load_iris
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    keep = iris.target != 0
    x, y = iris.data[keep], iris.target[keep] - 1
    correct, shuffled = [], []
    for seed in range(seeds):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=50, random_state=seed, stratify=y
        )
        probability = MicroTabICLv2Classifier(model, device).fit(x_train, y_train).predict_proba(x_test)[:, 1]
        correct.append(roc_auc_score(y_test, probability))
        for permutation in range(shuffles):
            rng = np.random.default_rng(seed * 100 + permutation)
            probability = MicroTabICLv2Classifier(model, device).fit(
                x_train, rng.permutation(y_train)
            ).predict_proba(x_test)[:, 1]
            shuffled.append(roc_auc_score(y_test, probability))
    return float(np.mean(correct)), float(np.mean(shuffled))


def rule_switching(model, device="auto", repeats=20, grid_size=31):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier

    rng, rows = np.random.default_rng(11), 48
    angle = np.linspace(0, 2 * np.pi, rows, endpoint=False) + 0.07
    radius = rng.uniform(0.55, 1.75, rows)
    context = np.c_[np.cos(angle) * radius, np.sin(angle) * radius]
    axis = np.linspace(-2, 2, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.c_[xx.ravel(), yy.ravel()]
    def features(value):
        return np.c_[value, value].astype("float32")
    tasks = {
        "vertical": np.array([1.0, 0.0]),
        "diagonal": np.array([1.0, 1.0]),
        "horizontal": np.array([0.0, 1.0]),
        "anti-diagonal": np.array([1.0, -1.0]),
    }
    surfaces, scores = {}, {}
    for name, weight in tasks.items():
        labels, truth = (context @ weight > 0).astype(int), (grid @ weight > 0).astype(int)
        probability = MicroTabICLv2Classifier(model, device).fit(
            features(context), labels
        ).predict_proba(features(grid))[:, 1]
        surfaces[name], scores[name] = probability, roc_auc_score(truth, probability)

    controls = {name: [] for name in ("correct", "wrong", "shuffled", "random", "logreg", "5-NN")}
    torch.manual_seed(11)
    random_model = MicroTabICLv2(**model.config)
    for _ in range(repeats):
        theta = rng.uniform(0, np.pi)
        weight, wrong = np.array([np.cos(theta), np.sin(theta)]), np.array([-np.sin(theta), np.cos(theta)])
        train_x, test_x = rng.uniform(-2, 2, (48, 2)), rng.uniform(-2, 2, (192, 2))
        train_y, test_y = (train_x @ weight > 0).astype(int), (test_x @ weight > 0).astype(int)
        variants = {
            "correct": (model, train_y),
            "wrong": (model, (train_x @ wrong > 0).astype(int)),
            "shuffled": (model, rng.permutation(train_y)),
            "random": (random_model, train_y),
        }
        for name, (candidate, labels) in variants.items():
            probability = MicroTabICLv2Classifier(candidate, device).fit(
                features(train_x), labels
            ).predict_proba(features(test_x))[:, 1]
            controls[name].append(roc_auc_score(test_y, probability))
        for name, estimator in (("logreg", LogisticRegression()), ("5-NN", KNeighborsClassifier(5))):
            probability = estimator.fit(train_x, train_y).predict_proba(test_x)[:, 1]
            controls[name].append(roc_auc_score(test_y, probability))
    scores.update({name: float(np.mean(values)) for name, values in controls.items()})
    return scores, (axis, context, tasks, surfaces)


def write_surface_svg(data, path):
    """A deliberately tiny, dependency-free 2×2 decision-surface plot."""
    axis, context, tasks, surfaces = data
    path, size = Path(path), 240
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 620">']
    parts += [
        '<rect width="620" height="620" fill="#f7f4ed"/>',
        '<text x="25" y="32" font-size="22">One model, four context-defined rules</text>',
    ]
    for index, (name, weight) in enumerate(tasks.items()):
        left, top = 40 + index % 2 * 300, 65 + index // 2 * 285
        cell, values = size / len(axis), surfaces[name]
        parts.append(f'<text x="{left}" y="{top - 8}" font-size="14">{name}</text>')
        for i, value in enumerate(values):
            red, blue = int(245 - 25 * value), int(130 + 95 * value)
            x, y = left + i % len(axis) * cell, top + (len(axis) - 1 - i // len(axis)) * cell
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell + 0.3:.1f}" '
                f'height="{cell + 0.3:.1f}" fill="rgb({red},190,{blue})"/>'
            )
        labels = (context @ weight > 0).astype(int)
        for point, label in zip(context, labels, strict=True):
            x, y = left + (point[0] + 2) / 4 * size, top + (2 - point[1]) / 4 * size
            color = "#e06934" if label else "#224684"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="white"/>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def write_curve_svg(history, path):
    path, width, height = Path(path), 620, 360

    def x(step):
        return 55 + step / max(history[-1][0], 1) * 530

    def y(value):
        return 300 - value * 240

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="620" height="360" fill="#f7f4ed"/>',
        '<text x="25" y="32" font-size="22">AUC while synthetic pretraining</text>',
        '<line x1="55" y1="180" x2="585" y2="180" stroke="#777" stroke-dasharray="5 5"/>',
        '<text x="48" y="64" text-anchor="end" font-size="12">1.0</text>',
        '<text x="48" y="184" text-anchor="end" font-size="12">0.5</text>',
        '<text x="48" y="304" text-anchor="end" font-size="12">0.0</text>',
        '<line x1="390" y1="27" x2="415" y2="27" stroke="#287d67" stroke-width="3"/>',
        '<text x="421" y="31" font-size="12">correct context</text>',
        '<line x1="390" y1="45" x2="415" y2="45" stroke="#7654a6" stroke-width="3"/>',
        '<text x="421" y="49" font-size="12">shuffled context</text>',
    ]
    for column, color in ((1, "#287d67"), (2, "#7654a6")):
        points = " ".join(f"{x(row[0]):.1f},{y(row[column]):.1f}" for row in history)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    parts += [
        '<text x="55" y="325">0</text>',
        f'<text x="585" y="325" text-anchor="end">{history[-1][0]} steps</text>',
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


# Optional Modal entrypoints live here too: one file locally and in the cloud.
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "torch>=2.3,<3", "numpy>=1.24,<2", "scikit-learn>=1.3"
    ).add_local_file("microtabiclv2.py", "/root/microtabiclv2.py")
    app = modal.App("microtabiclv2", image=image)
    volume = modal.Volume.from_name("microtabiclv2-checkpoints", create_if_missing=True)

    def _modal_run(profile, steps, name):
        from microtabiclv2 import build_model, save, train

        model, _ = train(build_model(profile), steps=steps, device="cuda")
        save(model, f"/checkpoints/{name}.pt")
        return {"parameters": model.parameter_count(), "checkpoint": f"{name}.pt"}

    @app.function(gpu=["L40S", "A10", "T4"], timeout=86_400, volumes={"/checkpoints": volume})
    def modal_small(profile="small", steps=5_000, name="small"):
        return _modal_run(profile, steps, name)

    @app.function(gpu=["H100", "A100-80GB"], timeout=86_400, volumes={"/checkpoints": volume})
    def modal_large(profile="paper", steps=100_000, name="paper"):
        return _modal_run(profile, steps, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=PRESETS, default="micro")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="checkpoints/microtabiclv2.pt")
    parser.add_argument("--load")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--proof-repeats", type=int, default=20)
    parser.add_argument("--svg-dir")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = load(args.load, args.device) if args.load else build_model(args.profile)
    history = []
    if args.steps:
        model, history = train(
            model,
            args.steps,
            args.batch,
            args.device,
            evaluate=iris_auc,
            eval_every=args.eval_every,
        )
        save(model, args.save)
    auc, shuffled = iris_auc(model, args.device)
    scores, surface_data = rule_switching(model, args.device, args.proof_repeats)
    print(f"\nBinary Iris AUC: {auc:.3f} (shuffled context {shuffled:.3f})")
    rules = ("vertical", "diagonal", "horizontal", "anti-diagonal")
    controls = ("correct", "wrong", "shuffled", "random", "logreg", "5-NN")
    print("Rule switching:", " | ".join(f"{name} {scores[name]:.3f}" for name in rules))
    print("Controls:", " | ".join(f"{name} {scores[name]:.3f}" for name in controls))
    if args.svg_dir:
        write_surface_svg(surface_data, Path(args.svg_dir) / "rule-switching.svg")
        if history:
            write_curve_svg(history, Path(args.svg_dir) / "auc-vs-steps.svg")


if __name__ == "__main__":
    main()
