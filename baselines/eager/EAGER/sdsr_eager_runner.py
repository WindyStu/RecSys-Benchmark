import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from lib import Trm4Rec
from lib.generate_training_batches import Train_instance
from lib.metrics import compute_metrics, format_metrics
from lib.sdsr_data import load_semantic_features, prepare_sdsr_domain
from optimizers import AdamOptimizer
from optimizers.lr_schedulers import InverseSquareRootSchedule


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_torch(path, device):
    try:
        return torch.load(path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device(device))


def parse_int_list(text, expected_len):
    values = [int(part) for part in text.split(",") if part.strip()]
    if len(values) == 1 and expected_len > 1:
        values = values * expected_len
    if len(values) != expected_len:
        raise ValueError(f"expected {expected_len} values, got {values}")
    return values


def prepare_common(args):
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
    return data_dir, work_dir, output_dir, summary


def load_stream_features(args, data_dir, output_dir, item_num, device):
    din_path = Path(args.din_model_path) if args.din_model_path else output_dir / "DIN_MODEL.pt"
    if not din_path.exists():
        raise FileNotFoundError(f"missing DIN model: {din_path}")
    din_model = load_torch(din_path, device)
    behavior_features = din_model.item_embedding.embed.weight.data[:item_num, :].detach().cpu()
    if args.tree_num == 1:
        return [behavior_features], [behavior_features.shape[1]], [0]

    semantic_features = torch.from_numpy(load_semantic_features(data_dir, args.dataset, item_num)).cpu()
    return [behavior_features, semantic_features], [behavior_features.shape[1], semantic_features.shape[1]], [0, 1]


def build_models(args, item_num, data_list, feature_dims, stream_types, tree_has_generated):
    tree_dir = Path(args.output_root) / args.dataset / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)
    dec_layers = parse_int_list(args.dec_num_layers, args.tree_num)
    init_ways = [part for part in args.init_way.split(",") if part.strip()]
    if len(init_ways) == 1 and args.tree_num > 1:
        init_ways = init_ways * args.tree_num
    if len(init_ways) != args.tree_num:
        raise ValueError(f"expected {args.tree_num} init ways, got {init_ways}")

    models = []
    for tree_id in range(args.tree_num):
        item_to_code_file = tree_dir / f"{init_ways[tree_id]}{args.feature_ratio}_item_to_code_tree_id_{tree_id}_k{args.k}.npy"
        code_to_item_file = tree_dir / f"{init_ways[tree_id]}{args.feature_ratio}_code_to_item_tree_id_{tree_id}_k{args.k}.npy"
        model = Trm4Rec(
            item_num=int(item_num),
            user_seq_len=args.seq_len - 1,
            d_model=args.d_model,
            d_model2=feature_dims[tree_id],
            nhead=args.n_head,
            device="cuda",
            optimizer=lambda params: torch.optim.Adam(params, lr=args.lr, amsgrad=True),
            enc_num_layers=args.enc_num_layers,
            dec_num_layers=dec_layers[tree_id],
            k=args.k,
            item_to_code_file=str(item_to_code_file),
            code_to_item_file=str(code_to_item_file),
            tree_has_generated=tree_has_generated,
            init_way=init_ways[tree_id],
            max_iters=args.max_iters,
            feature_ratio=args.feature_ratio,
            data=data_list[tree_id],
            parall=args.parall,
            type=stream_types[tree_id],
        )
        if tree_id > 0:
            model.trm_model.trm.encoder = models[0].trm_model.trm.encoder
        models.append(model)
    return models


def build_optimizer(args, models):
    optim_args = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_updates": args.warmup_updates,
        "warmup_init_lr": args.warmup_init_lr,
    }
    parameters = list(models[0].trm_model.trm.encoder.parameters())
    for model in models:
        trm_model = model.trm_model
        parameters += list(trm_model.trm.decoder.parameters())
        parameters += list(trm_model.fc_proj1.parameters())
        parameters += list(trm_model.guide_proj.parameters())
        parameters += list(trm_model.trans_d_rec.parameters())
        parameters += list(trm_model.fc_comp.parameters())
        parameters += [trm_model.start_vec, trm_model.mask_vec]
    optimizer = AdamOptimizer(optim_args, parameters)
    scheduler = InverseSquareRootSchedule(optim_args, optimizer)
    return optimizer, scheduler


def rerank(models, stream_types, batch_x, label, top_k):
    if len(models) == 1:
        return label[:, :top_k]
    scores = torch.full((len(batch_x), top_k * len(models)), -1e15, device="cuda")
    input_labels = torch.zeros((len(batch_x), top_k * len(models)), dtype=torch.int64, device="cuda")
    max_len = top_k
    for row_idx, result in enumerate(label):
        unique = torch.LongTensor(list(dict.fromkeys(result.tolist()))).to("cuda")
        scores[row_idx, : len(unique)] = 0.0
        input_labels[row_idx, : len(unique)] = unique
        max_len = max(max_len, len(unique))
    scores = scores[:, :max_len]
    input_labels = input_labels[:, :max_len]
    input_user = batch_x.repeat_interleave(max_len, dim=0)
    input_item = input_labels.reshape(-1)
    with torch.no_grad():
        for model, stream_type in zip(models, stream_types):
            scores += model.compute_scores(input_user, input_item, type=stream_type).sum(-1).view(batch_x.shape[0], -1)
        indices = scores.argsort(-1, descending=True)[:, :top_k]
        return input_labels.gather(index=indices, dim=-1)


def evaluate_models(models, stream_types, test_instances, labels, batch_size=50, topk=10, num_beams=100):
    for model in models:
        model.trm_model.eval()
    predictions = []
    num_batch = math.ceil(test_instances.shape[0] / batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            batch_user = test_instances[batch_idx * batch_size : (batch_idx + 1) * batch_size].to("cuda")
            batch_result_list = [
                model.predict(batch_user, topk=topk, num_beams=num_beams, type=stream_type)
                for model, stream_type in zip(models, stream_types)
            ]
            batch_result = torch.cat(batch_result_list, dim=-1)
            batch_result = rerank(models, stream_types, batch_user, batch_result, topk)
            predictions.extend(batch_result.cpu().tolist())
    for model in models:
        model.trm_model.train()
    return compute_metrics(predictions, labels, cutoffs=(5, 10)), predictions


def evaluate_models_full(
    models,
    stream_types,
    test_instances,
    labels,
    item_num,
    batch_size=16,
    item_batch_size=2048,
    topk=10,
    filter_seen=False,
):
    for model in models:
        model.trm_model.eval()
    topk = max(topk, 10)
    all_items = torch.arange(item_num, device="cuda", dtype=torch.int64)
    predictions = []
    num_batch = math.ceil(test_instances.shape[0] / batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            batch_user = test_instances[batch_idx * batch_size : (batch_idx + 1) * batch_size].to("cuda")
            scores = torch.full((batch_user.shape[0], item_num), -1e15, device="cuda")
            for item_start in range(0, item_num, item_batch_size):
                item_ids = all_items[item_start : item_start + item_batch_size]
                input_user = batch_user.repeat_interleave(len(item_ids), dim=0)
                input_item = item_ids.repeat(batch_user.shape[0])
                chunk_scores = torch.zeros(len(input_item), device="cuda")
                for model, stream_type in zip(models, stream_types):
                    chunk_scores += model.compute_scores(input_user, input_item, type=stream_type).sum(-1)
                scores[:, item_start : item_start + len(item_ids)] = chunk_scores.view(batch_user.shape[0], -1)
            if filter_seen:
                for row_idx, user_history in enumerate(batch_user):
                    user_seen = user_history[user_history < item_num]
                    scores[row_idx, user_seen] = -1e15
            predictions.extend(torch.topk(scores, k=topk, dim=-1).indices.cpu().tolist())
    for model in models:
        model.trm_model.train()
    return compute_metrics(predictions, labels, cutoffs=(5, 10)), predictions


def save_checkpoint(path, models, step, args, item_num, feature_dims, stream_types):
    state = {
        "step": step,
        "model_state_dicts": [model.trm_model.state_dict() for model in models],
        "args": vars(args),
        "item_num": int(item_num),
        "feature_dims": [int(dim) for dim in feature_dims],
        "stream_types": [int(stream_type) for stream_type in stream_types],
    }
    torch.save(state, path)
    print(f"saved {path}")


def load_checkpoint_models(checkpoint_path, args, data_list):
    checkpoint = load_torch(checkpoint_path, "cuda")
    feature_dims = checkpoint["feature_dims"]
    stream_types = checkpoint["stream_types"]
    models = build_models(args, checkpoint["item_num"], data_list, feature_dims, stream_types, tree_has_generated=True)
    for model, state_dict in zip(models, checkpoint["model_state_dicts"]):
        model.trm_model.load_state_dict(state_dict, strict=False)
    return checkpoint, models, stream_types


def write_metrics(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def print_metrics(prefix, step, metrics):
    print(f"{prefix} step={step} {format_metrics(metrics)}")
