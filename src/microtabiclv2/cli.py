"""Command-line interface."""

from __future__ import annotations

import argparse

from .config import get_model_config, get_train_config, profile_names
from .estimator import MicroTabICLv2Classifier, MicroTabICLv2Regressor
from .evaluate import evaluate_checkpoint, print_report, save_report
from .model import MicroTabICLv2
from .proof import auc_learning_curve, rule_switching_proof
from .train import train_model


def _profile(value: str) -> str:
    if value not in profile_names():
        raise argparse.ArgumentTypeError(f"choose from {', '.join(profile_names())}")
    return value


def _train(args: argparse.Namespace) -> None:
    model_config = get_model_config(args.profile, args.task)
    overrides = {
        name: value
        for name, value in {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "device": args.device,
            "amp": args.amp,
            "optimizer": args.optimizer,
            "output": args.output,
            "seed": args.seed,
        }.items()
        if value is not None
    }
    train_config = get_train_config(args.profile, **overrides)
    result = train_model(model_config, train_config, resume=args.resume)
    print(
        f"done | loss {result.final_loss:.4f} | {result.elapsed_seconds:.1f}s | "
        f"checkpoint {result.checkpoint}"
    )


def _info(_: argparse.Namespace) -> None:
    print(f"{'profile':<10} {'parameters':>14} {'embed':>8} {'col/row/icl':>14} {'inducing':>10}")
    print("-" * 64)
    for name in profile_names():
        config = get_model_config(name)
        model = MicroTabICLv2(config)
        blocks = f"{config.col_blocks}/{config.row_blocks}/{config.icl_blocks}"
        print(
            f"{name:<10} {model.parameter_count():>14,} {config.embed_dim:>8} "
            f"{blocks:>14} {config.inducing_tokens:>10}"
        )


def _demo(args: argparse.Namespace) -> None:
    from sklearn.datasets import load_diabetes, load_iris
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split

    model_config = get_model_config(args.profile, args.task)
    train_config = get_train_config(
        args.profile,
        steps=args.steps,
        device=args.device,
        amp=args.amp,
        output="",
        log_every=max(1, args.steps // 5),
    )
    result = train_model(model_config, train_config)

    if args.task == "classification":
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=90, random_state=args.seed, stratify=y
        )
        estimator = MicroTabICLv2Classifier(result.model, device=result.device)
        estimator.fit(x_train, y_train)
        score = accuracy_score(y_test, estimator.predict(x_test))
        print(f"Iris accuracy after synthetic pretraining: {score:.3f}")
    else:
        x, y = load_diabetes(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=96, test_size=96, random_state=args.seed
        )
        estimator = MicroTabICLv2Regressor(result.model, device=result.device)
        estimator.fit(x_train, y_train)
        score = r2_score(y_test, estimator.predict(x_test))
        print(f"Diabetes R2 after synthetic pretraining: {score:.3f}")


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_checkpoint(
        args.checkpoint,
        device=args.device,
        seeds=args.seeds,
        max_train_rows=args.max_train_rows,
    )
    print_report(report)
    destination = save_report(report, args.output)
    print(f"report: {destination}")


def _proof(args: argparse.Namespace) -> None:
    scores = rule_switching_proof(
        args.checkpoint,
        args.output,
        device=args.device,
        seed=args.seed,
        repeats=args.repeats,
    )
    for name in ("vertical", "diagonal", "horizontal", "anti-diagonal"):
        print(f"{name:<14} AUC {scores[name]:.3f}")
    print(f"figure: {args.output}")


def _auc_curve(args: argparse.Namespace) -> None:
    records, result = auc_learning_curve(
        profile=args.profile,
        steps=args.steps,
        every=args.every,
        splits=args.splits,
        shuffles=args.shuffles,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        output=args.output,
        checkpoint=args.checkpoint,
    )
    gain = records[-1]["auc"] - records[0]["auc"]
    print(f"AUC gain: {gain:+.3f}")
    print(f"figure: {args.output}")
    print(f"data: {args.output.rsplit('.', 1)[0]}.csv")
    print(f"checkpoint: {result.checkpoint}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microtabiclv2",
        description="Train a compact TabICLv2 locally or inspect its scale profiles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show model profile sizes")
    info.set_defaults(handler=_info)

    train = subparsers.add_parser("train", help="pretrain on the synthetic graph prior")
    train.add_argument("--profile", type=_profile, default="micro")
    train.add_argument("--task", choices=("classification", "regression"), default="classification")
    train.add_argument("--steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument("--optimizer", choices=("adamw", "muon"))
    train.add_argument("--output")
    train.add_argument("--resume")
    train.add_argument("--seed", type=int)
    train.set_defaults(handler=_train)

    demo = subparsers.add_parser("demo", help="train briefly and evaluate on a real dataset")
    demo.add_argument("--profile", type=_profile, default="micro")
    demo.add_argument("--task", choices=("classification", "regression"), default="classification")
    demo.add_argument("--steps", type=int, default=50)
    demo.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    demo.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(handler=_demo)

    evaluate = subparsers.add_parser("eval", help="benchmark a classification checkpoint")
    evaluate.add_argument("--checkpoint", default="checkpoints/microtabiclv2-clf.pt")
    evaluate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    evaluate.add_argument("--seeds", type=int, default=5)
    evaluate.add_argument("--max-train-rows", type=int, default=100)
    evaluate.add_argument("--output", default="evaluations/microtabiclv2-clf.json")
    evaluate.set_defaults(handler=_evaluate)

    proof = subparsers.add_parser("proof", help="show one frozen model switching between rules")
    proof.add_argument("--checkpoint", default="checkpoints/microtabiclv2-clf.pt")
    proof.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    proof.add_argument("--repeats", type=int, default=20, help="random rotated tasks for controls")
    proof.add_argument("--seed", type=int, default=11)
    proof.add_argument("--output", default="evaluations/rule-switching.svg")
    proof.set_defaults(handler=_proof)

    curve = subparsers.add_parser("auc-curve", help="plot AUC while synthetic pretraining runs")
    curve.add_argument("--profile", type=_profile, default="micro")
    curve.add_argument("--steps", type=int, default=500)
    curve.add_argument("--every", type=int, default=50)
    curve.add_argument("--splits", type=int, default=5)
    curve.add_argument("--shuffles", type=int, default=10, help="label permutations per split")
    curve.add_argument("--batch-size", type=int)
    curve.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    curve.add_argument("--seed", type=int, default=42)
    curve.add_argument("--output", default="evaluations/auc-vs-steps.svg")
    curve.add_argument("--checkpoint", default="checkpoints/auc-curve.pt")
    curve.set_defaults(handler=_auc_curve)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
