import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from lib import DINTrain
from lib.generate_training_batches import Train_instance
from lib.metrics import compute_metrics, format_metrics
from lib.sdsr_data import prepare_sdsr_domain


def parse_feature_groups(text, history_len):
    if text:
        groups = [int(part) for part in text.split(",") if part.strip()]
    elif history_len == 19:
        groups = [5, 4, 2, 2, 1, 1, 1, 1, 1, 1]
    else:
        groups = [history_len]
    if sum(groups) != history_len:
        raise ValueError(f"feature groups sum to {sum(groups)}, expected {history_len}")
    return groups


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_din(model, test_instances, labels, item_num, device, topk=10, eval_batch_size=32, item_batch_size=4096):
    model.eval()
    all_items = torch.arange(item_num, device=device).view(-1, 1)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_instances), eval_batch_size):
            users = test_instances[start : start + eval_batch_size].to(device)
            scores = torch.empty((len(users), item_num), dtype=torch.float32, device=device)
            for item_start in range(0, item_num, item_batch_size):
                part_labels = all_items[item_start : item_start + item_batch_size]
                expanded_users = users.repeat_interleave(len(part_labels), dim=0)
                expanded_items = part_labels.repeat(len(users), 1)
                part_scores = model(expanded_users, expanded_items).view(len(users), -1)
                scores[:, item_start : item_start + len(part_labels)] = part_scores
            predictions.extend(torch.topk(scores, k=topk, dim=-1).indices.cpu().tolist())
    model.train()
    return compute_metrics(predictions, labels, cutoffs=(5, 10)), predictions


def main():
    parser = argparse.ArgumentParser(description="Train DIN behavior encoder on one SDSR domain.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="/nfsshare/home/liujingyan/data/CDSR/data")
    parser.add_argument("--work-root", default="runs/sdsr_work")
    parser.add_argument("--output-root", default="runs/sdsr_outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-number", type=int, default=80000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--item-batch-size", type=int, default=4096)
    parser.add_argument("--sample-negative-num", type=int, default=60)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--feature-groups", default="")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.set_device(0)
        device = "cuda"
    else:
        device = "cpu"

    data_dir = Path(args.data_root) / args.dataset
    work_dir = Path(args.work_root) / args.dataset
    output_dir = Path(args.output_root) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = prepare_sdsr_domain(
        data_dir=data_dir,
        output_dir=work_dir,
        dataset=args.dataset,
        seq_len=args.seq_len,
        min_seq_len=args.min_seq_len,
        train_sample_seg_cnt=args.segments,
        seed=args.seed,
        force=args.force_prepare,
    )
    print(f"prepared {summary}")

    train_instances = Train_instance()
    training_data, training_labels = train_instances.get_training_data(
        str(work_dir / "train_instances"),
        args.segments,
        summary.item_num,
        str(work_dir / "his_matrix.pt"),
        str(work_dir / "labels.pt"),
    )
    test_instances = train_instances.read_test_instances_file(str(work_dir / "test_instances"), summary.item_num)
    feature_groups = parse_feature_groups(args.feature_groups, args.seq_len - 1)

    trainer = DINTrain(
        item_num=summary.item_num,
        sample_negative_num=args.sample_negative_num,
        emb_dim=args.emb_dim,
        device=device,
        feature_groups=feature_groups,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True),
    )

    metrics_path = output_dir / "din_metrics.jsonl"
    for batch_x, batch_y in train_instances.generate_training_records(training_data, training_labels, batch_size=args.batch_size):
        loss = trainer.update_DIN(batch_x, batch_y)
        if trainer.batch_num % 100 == 0:
            print(f"step={trainer.batch_num} mean_loss={loss.item():.6f}")
        if args.eval_every > 0 and trainer.batch_num % args.eval_every == 0:
            metrics, _ = evaluate_din(
                trainer.DINModel,
                test_instances,
                train_instances.test_labels,
                summary.item_num,
                device,
                topk=10,
                eval_batch_size=args.eval_batch_size,
                item_batch_size=args.item_batch_size,
            )
            record = {"step": trainer.batch_num, **metrics}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f"DIN_EVAL step={trainer.batch_num} {format_metrics(metrics)}")
        if args.save_every > 0 and trainer.batch_num % args.save_every == 0:
            ckpt = output_dir / f"DIN_MODEL_{trainer.batch_num}.pt"
            torch.save(trainer.DINModel, ckpt)
            print(f"saved {ckpt}")
        if trainer.batch_num >= args.batch_number:
            break

    final_path = output_dir / "DIN_MODEL.pt"
    torch.save(trainer.DINModel, final_path)
    print(f"saved {final_path}")


if __name__ == "__main__":
    main()
