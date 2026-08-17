import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lib.generate_training_batches import Train_instance
from sdsr_eager_runner import (
    evaluate_models,
    evaluate_models_full,
    load_checkpoint_models,
    load_torch,
    prepare_common,
    print_metrics,
    set_seed,
)


def latest_checkpoint(output_dir):
    final_path = output_dir / "EAGER_MODEL.pt"
    if final_path.exists():
        return final_path
    checkpoints = sorted(output_dir.glob("EAGER_MODEL_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"no EAGER checkpoint found under {output_dir}")
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate an EAGER checkpoint on one SDSR domain.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="/nfsshare/home/liujingyan/data/CDSR/data")
    parser.add_argument("--work-root", default="runs/sdsr_work")
    parser.add_argument("--output-root", default="runs/sdsr_outputs")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--test-batch-size", type=int, default=50)
    parser.add_argument("--item-batch-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-beams", type=int, default=100)
    parser.add_argument("--eval-mode", choices=("full", "beam"), default="full")
    parser.add_argument("--filter-seen", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(0)

    output_dir = Path(args.output_root) / args.dataset
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else latest_checkpoint(output_dir)
    checkpoint = load_torch(checkpoint_path, "cuda")
    model_args_dict = dict(checkpoint["args"])
    model_args_dict.update(
        {
            "dataset": args.dataset,
            "data_root": args.data_root,
            "work_root": args.work_root,
            "output_root": args.output_root,
            "test_batch_size": args.test_batch_size,
            "topk": args.topk,
            "num_beams": args.num_beams,
            "force_prepare": args.force_prepare,
        }
    )
    model_args = SimpleNamespace(**model_args_dict)

    _, work_dir, output_dir, summary = prepare_common(model_args)
    data_list = [torch.empty((checkpoint["item_num"], dim)) for dim in checkpoint["feature_dims"]]
    checkpoint, models, stream_types = load_checkpoint_models(checkpoint_path, model_args, data_list)

    train_instances = Train_instance(parall=getattr(model_args, "parall", 10))
    test_instances = train_instances.read_test_instances_file(str(work_dir / "test_instances"), summary.item_num)
    if args.eval_mode == "full":
        metrics, predictions = evaluate_models_full(
            models,
            stream_types,
            test_instances,
            train_instances.test_labels,
            item_num=summary.item_num,
            batch_size=args.test_batch_size,
            item_batch_size=args.item_batch_size,
            topk=args.topk,
            filter_seen=args.filter_seen,
        )
    else:
        metrics, predictions = evaluate_models(
            models,
            stream_types,
            test_instances,
            train_instances.test_labels,
            batch_size=args.test_batch_size,
            topk=args.topk,
            num_beams=args.num_beams,
        )
    result = {
        "dataset": args.dataset,
        "checkpoint": str(checkpoint_path),
        "eval_mode": args.eval_mode,
        "filter_seen": args.filter_seen,
        "test_users": len(predictions),
        **metrics,
    }
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print_metrics("EAGER_EVAL_ONLY", checkpoint["step"], metrics)


if __name__ == "__main__":
    main()
