import argparse
from pathlib import Path

import numpy as np
import torch

from lib.generate_training_batches import Train_instance
from sdsr_eager_runner import (
    build_models,
    build_optimizer,
    evaluate_models,
    load_stream_features,
    prepare_common,
    print_metrics,
    save_checkpoint,
    set_seed,
    write_metrics,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train EAGER on one SDSR single-domain dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="/nfsshare/home/liujingyan/data/CDSR/data")
    parser.add_argument("--work-root", default="runs/sdsr_work")
    parser.add_argument("--output-root", default="runs/sdsr_outputs")
    parser.add_argument("--din-model-path", default="")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-batch-num", type=int, default=60000)
    parser.add_argument("--eval-every", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=3000)
    parser.add_argument("--test-batch-size", type=int, default=50)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-beams", type=int, default=100)
    parser.add_argument("--tree-num", type=int, default=2)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--enc-num-layers", type=int, default=1)
    parser.add_argument("--dec-num-layers", default="2,2")
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--init-way", default="embkm,embkm")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--feature-ratio", type=float, default=1.0)
    parser.add_argument("--parall", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--warmup-updates", type=int, default=2000)
    parser.add_argument("--warmup-init-lr", type=float, default=1e-7)
    parser.add_argument("--use-con", action="store_true")
    parser.add_argument("--use-guide", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--force-prepare", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    torch.cuda.set_device(0)

    data_dir, work_dir, output_dir, summary = prepare_common(args)
    if summary.item_num > args.k * args.k:
        raise ValueError(f"item_num={summary.item_num} needs k >= ceil(sqrt(item_num)); current k={args.k}")
    print(f"prepared {summary}")

    data_list, feature_dims, stream_types = load_stream_features(args, data_dir, output_dir, summary.item_num, "cuda")
    models = build_models(args, summary.item_num, data_list, feature_dims, stream_types, tree_has_generated=False)
    optimizer, scheduler = build_optimizer(args, models)

    train_instances = Train_instance(parall=args.parall)
    training_data, training_labels = train_instances.get_training_data(
        str(work_dir / "train_instances"),
        args.segments,
        summary.item_num,
        str(work_dir / "his_matrix.pt"),
        str(work_dir / "labels.pt"),
    )
    test_instances = train_instances.read_test_instances_file(str(work_dir / "test_instances"), summary.item_num)
    metrics_path = output_dir / "eager_metrics.jsonl"
    step = 0

    for model in models:
        model.trm_model.train()

    for batch_x, batch_y in train_instances.generate_training_records(
        training_data, training_labels, batch_size=args.batch_size
    ):
        loss = 0
        contra_loss = 0
        guide_feat = None
        for idx in range(len(models) - 1, -1, -1):
            loss_i, contra_i, guide_feat = models[idx].update_model(
                batch_x,
                batch_y,
                data_list[idx],
                type=stream_types[idx],
                use_con=args.use_con,
                use_guide=args.use_guide,
                guide_feat=guide_feat,
            )
            loss = loss + loss_i
            contra_loss = contra_loss + contra_i

        loss.backward()
        optimizer.step()
        step += 1
        current_lr = scheduler.step_update(step)
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"step={step} lr={current_lr:.6f} loss={loss.item():.6f} contra_loss={contra_loss.item():.6f}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            metrics, _ = evaluate_models(
                models,
                stream_types,
                test_instances,
                train_instances.test_labels,
                batch_size=args.test_batch_size,
                topk=args.topk,
                num_beams=args.num_beams,
            )
            write_metrics(metrics_path, {"step": step, **metrics})
            print_metrics("EAGER_EVAL", step, metrics)

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                output_dir / f"EAGER_MODEL_{step}.pt",
                models,
                step,
                args,
                summary.item_num,
                feature_dims,
                stream_types,
            )

        if step >= args.total_batch_num:
            break

    final_path = output_dir / "EAGER_MODEL.pt"
    save_checkpoint(final_path, models, step, args, summary.item_num, feature_dims, stream_types)
    metrics, _ = evaluate_models(
        models,
        stream_types,
        test_instances,
        train_instances.test_labels,
        batch_size=args.test_batch_size,
        topk=args.topk,
        num_beams=args.num_beams,
    )
    write_metrics(metrics_path, {"step": step, "final": True, **metrics})
    print_metrics("EAGER_FINAL", step, metrics)


if __name__ == "__main__":
    main()
