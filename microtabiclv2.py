"""A small, one-file TabICLv2 for learning. Run: python microtabiclv2.py"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The Mac model and one deliberately larger Modal model use the same code.
MODES = {
    "micro": dict(d=48, heads=4, columns=1, rows=1, icl=2, inducing=16, cls=2),
    "big": dict(d=128, heads=8, columns=3, rows=3, icl=8, inducing=128, cls=4),
}


def device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Synthetic pretraining gives the model many small classification problems.
def prior(batch=4, train_rows=64, test_rows=24, features=8, classes=2, dev="cpu"):
    total, dev = train_rows + test_rows, torch.device(dev)
    x = torch.randn(batch, total, features, device=dev)
    for j in range(1, features):
        parent = torch.bmm(x[:, :, :j], torch.randn(batch, j, 1, device=dev) / j**0.5).squeeze(-1)
        x[:, :, j] += random.choice((torch.tanh, torch.sin))(parent)
    w1 = torch.randn(batch, features, 16, device=dev) / features**0.5
    w2 = torch.randn(batch, 16, 1, device=dev) / 4
    score = torch.bmm(torch.tanh(torch.bmm(x, w1)), w2).squeeze(-1)

    # Rank bins balance the classes. Permuting their names forces true ICL.
    order = score.argsort(1)
    bins = torch.arange(total, device=dev) * classes // total
    y = torch.empty_like(order).scatter(1, order, bins.expand(batch, -1))
    for task in range(batch):
        y[task] = torch.randperm(classes, device=dev)[y[task]]
    return x, y[:, :train_rows], y[:, train_rows:]


class Linear(nn.Linear):
    """Avoid a biased high-rank Linear path that is unreliable on some Macs."""

    def forward(self, x):
        return F.linear(x, self.weight, None) + self.bias


def mlp(width, output=None):
    return nn.Sequential(Linear(width, 2 * width), nn.GELU(), Linear(2 * width, output or width))


class Attention(nn.Module):
    def __init__(self, d, heads, qass=False):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.q, self.k, self.v, self.out = (Linear(d, d) for _ in range(4))
        self.qass = qass
        if qass:  # QASSMax: scale attention using context length and each query.
            self.length, self.query = Linear(1, d), Linear(self.head_dim, self.head_dim)
            nn.init.zeros_(self.query.weight)
            nn.init.zeros_(self.query.bias)

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    def forward(self, query, context):
        q, k, v = self.split(self.q(query)), self.split(self.k(context)), self.split(self.v(context))
        if self.qass:
            log_n = q.new_tensor([[math.log(k.shape[-2])]])
            scale = F.softplus(self.length(log_n)).reshape(1, self.heads, 1, self.head_dim)
            q = q * scale * (1 + torch.tanh(self.query(q)))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(-3, -2).flatten(-2))


class Block(nn.Module):
    def __init__(self, d, heads, qass=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention, self.ff = Attention(d, heads, qass), mlp(d)

    def forward(self, query, context=None):
        q = self.norm1(query)
        context = q if context is None else self.norm1(context)
        hidden = query + self.attention(q, context)
        return hidden + self.ff(self.norm2(hidden))


class MicroTabICLv2(nn.Module):
    """Columns learn patterns over rows; test rows then attend to labeled rows."""

    def __init__(self, d=48, heads=4, columns=1, rows=1, icl=2, inducing=16, cls=2, classes=6):
        super().__init__()
        self.config = dict(
            d=d, heads=heads, columns=columns, rows=rows, icl=icl, inducing=inducing, cls=cls
        )
        self.x_embed, self.y_column = Linear(3, d), nn.Embedding(classes, d)
        self.column = nn.ModuleList(
            nn.ModuleList((Block(d, heads, True), Block(d, heads))) for _ in range(columns)
        )
        self.cls = nn.Parameter(torch.randn(1, 1, cls, d) * 0.02)
        self.row = nn.ModuleList(Block(d, heads) for _ in range(rows))
        self.y_icl = nn.Embedding(classes, d * cls)
        self.icl = nn.ModuleList(Block(d * cls, heads, True) for _ in range(icl))
        self.head = mlp(d * cls, classes)
        self.inducing = nn.Parameter(torch.randn(1, inducing, d) * 0.02)

    def forward(self, x, y_train):
        n_train, n_features = y_train.shape[1], x.shape[-1]
        mean, std = x[:, :n_train].mean(1, keepdim=True), x[:, :n_train].std(1, keepdim=True)
        x = (x.float() - mean) / std.clamp_min(1e-6)

        # Each feature sees itself and two cyclic neighbors, then its known labels.
        x = torch.stack([x.roll(-shift, -1) for shift in (0, 1, 3)], -1)
        x = self.x_embed(x)
        x[:, :n_train] += self.y_column(y_train).unsqueeze(2)

        batch, total, _, width = x.shape
        x = x.transpose(1, 2).reshape(batch * n_features, total, width)
        for read, write in self.column:  # induced attention compresses long columns
            tokens = self.inducing.expand(x.shape[0], -1, -1)
            x = write(x, read(tokens, x[:, :n_train]))
        x = x.reshape(batch, n_features, total, width).transpose(1, 2)

        x = x.reshape(batch * total, n_features, width)
        for block in self.row[:-1]:
            x = block(x)
        cls = self.cls.expand(batch, total, -1, -1).reshape(batch * total, -1, width)
        x = self.row[-1](cls, x).reshape(batch, total, -1)
        x[:, :n_train] += self.y_icl(y_train)

        for block in self.icl[:-1]:
            x = block(x, x[:, :n_train])
        return self.head(self.icl[-1](x[:, n_train:], x[:, :n_train]))

@torch.inference_mode()
def predict(model, x_train, y_train, x_test, dev="auto"):
    dev, classes = device(dev), len(np.unique(y_train))
    x = torch.as_tensor(np.r_[x_train, x_test], dtype=torch.float32, device=dev)[None]
    y = torch.as_tensor(y_train, dtype=torch.long, device=dev)[None]
    return model.to(dev).eval()(x, y)[0, :, :classes].softmax(-1).cpu().numpy()


def iris_auc(model, dev="auto", seeds=5, shuffles=5):
    from sklearn.datasets import load_iris
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    data = load_iris()
    keep = data.target != 0
    x, y, correct, shuffled = data.data[keep], data.target[keep] - 1, [], []
    for seed in range(seeds):
        xt, xv, yt, yv = train_test_split(x, y, train_size=50, stratify=y, random_state=seed)
        correct.append(roc_auc_score(yv, predict(model, xt, yt, xv, dev)[:, 1]))
        rng = np.random.default_rng(seed)
        for _ in range(shuffles):
            shuffled.append(roc_auc_score(yv, predict(model, xt, rng.permutation(yt), xv, dev)[:, 1]))
    return np.mean(correct), np.mean(shuffled)


def train(model, steps=500, batch=4, dev="auto", evaluate_every=50):
    dev, model = device(dev), model.to(device(dev))
    optimizer, history = torch.optim.AdamW(model.parameters(), lr=3e-4), []

    def evaluate(step):
        if step % evaluate_every == 0 or step == steps:
            model.eval()
            history.append((step, *iris_auc(model, dev)))
            print(f"eval {step:>3}: AUC {history[-1][1]:.3f}, shuffled {history[-1][2]:.3f}")

    print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {dev}")
    evaluate(0)
    for step in range(1, steps + 1):
        n, f, c = random.randint(32, 96), random.randint(4, 12), random.randint(2, 6)
        x, y_train, y_test = prior(batch, n, 24, f, c, dev)
        loss = F.cross_entropy(model.train()(x, y_train)[..., :c].flatten(0, 1), y_test.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        evaluate(step)
    return model.eval(), history


def rule_auc(model, dev="auto", repeats=100):
    from sklearn.metrics import roc_auc_score

    rng, scores = np.random.default_rng(11), {name: [] for name in ("correct", "wrong", "shuffled")}
    for _ in range(repeats):
        angle = rng.uniform(0, np.pi)
        rule, wrong = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
        xt, xv = rng.uniform(-2, 2, (48, 2)), rng.uniform(-2, 2, (192, 2))
        yt, yv = (xt @ rule > 0).astype(int), (xv @ rule > 0).astype(int)
        for name, labels in (("correct", yt), ("wrong", xt @ wrong > 0), ("shuffled", rng.permutation(yt))):
            probability = predict(model, np.c_[xt, xt], labels, np.c_[xv, xv], dev)[:, 1]
            scores[name].append(roc_auc_score(yv, probability))
    return {name: np.mean(values) for name, values in scores.items()}


def plot_proof(model, history, folder, dev="auto"):
    """Two ordinary Matplotlib figures keep visualization out of the model code."""
    import matplotlib.pyplot as plt

    folder, rng = Path(folder), np.random.default_rng(11)
    folder.mkdir(parents=True, exist_ok=True)
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False) + 0.07
    context = np.c_[np.cos(angle), np.sin(angle)] * rng.uniform(0.55, 1.75, (48, 1))
    axis = np.linspace(-2, 2, 31)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.c_[xx.ravel(), yy.ravel()]
    rules = {"vertical": (1, 0), "diagonal": (1, 1), "horizontal": (0, 1), "anti-diagonal": (1, -1)}

    figure, axes = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)
    for ax, (name, rule) in zip(axes.flat, rules.items(), strict=True):
        labels = context @ rule > 0
        probability = predict(model, np.c_[context, context], labels, np.c_[grid, grid], dev)[:, 1]
        ax.contourf(xx, yy, probability.reshape(xx.shape), levels=20, cmap="PuOr_r", vmin=0, vmax=1)
        ax.scatter(*context.T, c=labels, cmap="coolwarm", edgecolor="white", s=22)
        ax.set(title=name, xticks=[], yticks=[], xlim=(-2, 2), ylim=(-2, 2))
    figure.suptitle("One frozen model, four context-defined rules")
    figure.savefig(folder / "rule-switching.png", dpi=150)
    plt.close(figure)

    if history:
        figure, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot([x[0] for x in history], [x[1] for x in history], "o-", label="correct context")
        ax.plot([x[0] for x in history], [x[2] for x in history], "o-", label="shuffled context")
        ax.axhline(0.5, color="grey", linestyle="--")
        ax.set(xlabel="synthetic training steps", ylabel="binary Iris AUC", ylim=(0, 1))
        ax.legend()
        figure.savefig(folder / "auc-vs-steps.png", dpi=150)
        plt.close(figure)


# Optional cloud scaling, still in this file: modal run microtabiclv2.py::modal_train
try:
    import modal
except ImportError:
    modal = None

if modal:
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch", "numpy").add_local_file(
        "microtabiclv2.py", "/root/microtabiclv2.py"
    )
    app = modal.App("microtabiclv2", image=image)
    volume = modal.Volume.from_name("microtabiclv2-checkpoints", create_if_missing=True)

    @app.function(gpu=["L40S", "A10"], timeout=86_400, volumes={"/checkpoints": volume})
    def modal_train(steps=10_000):
        from microtabiclv2 import MODES, MicroTabICLv2, train

        model, _ = train(MicroTabICLv2(**MODES["big"]), steps, dev="cuda")
        torch.save(model.state_dict(), "/checkpoints/big.pt")
        return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot", default="docs/assets")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model, history = train(MicroTabICLv2(**MODES["micro"]), args.steps, dev=args.device)
    auc, shuffled = iris_auc(model, args.device)
    rules = rule_auc(model, args.device)
    print(f"Iris AUC: {auc:.3f} (shuffled {shuffled:.3f})")
    print("Rule AUC:", ", ".join(f"{name} {score:.3f}" for name, score in rules.items()))
    if args.plot:
        plot_proof(model, history, args.plot, args.device)


if __name__ == "__main__":
    main()
