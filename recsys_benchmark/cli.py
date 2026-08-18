from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

from recsys_benchmark.aggregator.results import aggregate_runs, write_leaderboard
from recsys_benchmark.config.loader import load_experiment_config, parse_overrides, save_resolved_config
from recsys_benchmark.config.readiness import inspect_methods
from recsys_benchmark.evaluator.runner import run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "aggregate":
        return _aggregate(args)
    if args.command == "run":
        return _run(args)
    if args.command == "inspect-methods":
        return _inspect_methods(args)
    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recsys-benchmark")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a configured method adapter")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--config-root", default="configs")
    run_parser.add_argument("--override", action="append", default=[])
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--stage", choices=["prepare", "train", "predict", "evaluate", "all"], default="all")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a prediction file")
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--ground-truth", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--input-type", choices=["candidate_scores", "topk"], required=True)
    eval_parser.add_argument("--cutoffs", nargs="+", type=int, default=[5, 10])
    eval_parser.add_argument("--method-id", required=True)
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--task", required=True)
    eval_parser.add_argument("--protocol", choices=["full", "sampled"], required=True)
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument("--catalog")
    eval_parser.add_argument("--item-metadata")

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate run metrics")
    aggregate_parser.add_argument("--results", default="outputs/runs")
    aggregate_parser.add_argument("--output-csv", default="results/leaderboard.csv")
    aggregate_parser.add_argument("--output-md", default="results/leaderboard.md")

    inspect_parser = subparsers.add_parser("inspect-methods", help="Inspect method YAML readiness status")
    inspect_parser.add_argument("--methods", default="configs/methods")
    inspect_parser.add_argument("--output")
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config, config_root=args.config_root, overrides=parse_overrides(args.override))
    if args.dry_run:
        config["dry_run"] = True
    run_id = config.get("run_id") or _default_run_id(config)
    output_dir = Path(str(config.get("output_root", "outputs/runs"))) / run_id
    config["output_dir"] = str(output_dir)
    save_resolved_config(config, output_dir / "config.resolved.yaml")

    adapter = _load_adapter(config)
    stages = ["prepare", "train", "predict", "evaluate"] if args.stage == "all" else [args.stage]
    results: dict[str, Any] = {}
    for stage in stages:
        results[stage] = getattr(adapter, stage)()
    results["artifacts"] = adapter.collect_artifacts()
    (output_dir / "artifacts.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    run_evaluation(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        input_type=args.input_type,
        cutoffs=args.cutoffs,
        metadata={
            "method_id": args.method_id,
            "dataset": args.dataset,
            "task": args.task,
            "protocol": args.protocol,
            "seed": args.seed,
            "eval_input_type": args.input_type,
        },
        catalog_path=args.catalog,
        item_metadata_path=args.item_metadata,
    )
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    rows = aggregate_runs(args.results)
    write_leaderboard(rows, args.output_csv, args.output_md)
    return 0


def _inspect_methods(args: argparse.Namespace) -> int:
    report = inspect_methods(args.methods)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _load_adapter(config: dict[str, Any]) -> Any:
    adapter_path = config["method"].get("adapter")
    if not adapter_path:
        raise ValueError("method.adapter is required")
    module_name, class_name = adapter_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(config)


def _default_run_id(config: dict[str, Any]) -> str:
    method_id = config["method"].get("method_id", "method")
    dataset_id = config["dataset"].get("dataset_id", "dataset")
    seed = config.get("seed", "seed")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{method_id}-{dataset_id}-s{seed}-{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
