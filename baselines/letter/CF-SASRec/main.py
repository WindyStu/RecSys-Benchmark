import argparse
import json
import logging
import multiprocessing
import os
import pickle
import random
import time
import datetime

import numpy as np
import torch
from tqdm import tqdm

from model import TiSASRec
from utils import *


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=50, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_epochs", default=201, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.00005, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--inference_only", default=False, type=str2bool)
    parser.add_argument("--state_dict_path", default=None, type=str)
    parser.add_argument("--time_span", default=256, type=int)
    parser.add_argument("--eval_step", default=5, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--topk", default=[5, 10], type=int, nargs="+")
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--metrics_path", default=None, type=str)
    parser.add_argument("--cache_dir", default="CF-SASRec/cache", type=str)
    parser.add_argument("--n_workers", default=3, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def setup_key_logger(log_file, dataset):
    if log_file is None:
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        module_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(module_dir, "log", f"{dataset}_sasrec_{run_ts}.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = logging.getLogger("CF-SASRec")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_file


def resolve_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device={device}, but CUDA is not available. Falling back to CPU.")
        return "cpu"
    return device


def load_relation_matrix(args, user_train, usernum):
    os.makedirs(args.cache_dir, exist_ok=True)
    relation_path = os.path.join(
        args.cache_dir,
        "relation_matrix_%s_%d_%d.pickle" % (args.dataset, args.maxlen, args.time_span),
    )
    if os.path.exists(relation_path):
        try:
            with open(relation_path, "rb") as f:
                relation_matrix = pickle.load(f)
            if isinstance(relation_matrix, dict) and len(relation_matrix) >= usernum:
                print(f"Loaded relation matrix cache: {relation_path}")
                return relation_matrix
            print(f"Relation matrix cache is incomplete, rebuilding: {relation_path}")
        except Exception as e:
            print(f"Failed to load relation matrix cache, rebuilding: {relation_path}")
            print(f"Cache load error: {type(e).__name__}: {e}")

    relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
    tmp_path = relation_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(relation_matrix, f)
    os.replace(tmp_path, relation_path)
    return relation_matrix


def build_model(args, usernum, itemnum):
    model = TiSASRec(usernum, itemnum, itemnum, args).to(args.device)
    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except Exception:
            pass
    return model


def export_item_embeddings(model, output_path):
    matrix = model.item_emb.weight[1:].detach().cpu()
    torch.save(matrix, output_path)
    return matrix


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger, key_log_file = setup_key_logger(args.log_file, args.dataset)
    logger.info("CF-SASRec started [dataset=%s, train_dir=%s, key_log=%s]", args.dataset, args.train_dir, key_log_file)
    logger.info("args: %s", vars(args))
    args.device = resolve_device(args.device)
    run_dir = os.path.join("CF-SASRec", "runs", args.dataset + "_" + args.train_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.txt"), "w") as f:
        f.write("\n".join([str(k) + "," + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum, timenum = dataset
    num_batch = len(user_train) // args.batch_size
    average_len = sum(len(user_train[u]) for u in user_train) / len(user_train)
    print("average sequence length: %.2f" % average_len)
    logger.info(
        "dataset loaded [users=%d, items=%d, timenum=%d, avg_seq_len=%.2f, num_batch=%d]",
        usernum, itemnum, timenum, average_len, num_batch,
    )

    log_file = open(os.path.join(run_dir, "log.txt"), "w")
    sampler = None
    try:
        relation_matrix = load_relation_matrix(args, user_train, usernum)
        sampler = WarpSampler(
            user_train,
            usernum,
            itemnum,
            relation_matrix,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            n_workers=args.n_workers,
        )
        model = build_model(args, usernum, itemnum)
        model.train()

        epoch_start_idx = 1
        if args.state_dict_path is not None:
            try:
                model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
                tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
                epoch_start_idx = int(tail[: tail.find(".")]) + 1
            except Exception:
                print("failed loading state_dicts, pls check file path: ", end="")
                print(args.state_dict_path)

        if args.inference_only:
            model.eval()
            t_test = evaluate(model, dataset, args)
            print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

        total_eval_time = 0.0
        t0 = time.time()
        best_valid_ndcg = -1.0
        stale_eval_count = 0
        best_metrics_item = None
        best_state_path = os.path.join(run_dir, "best_model.pth")
        metrics_history = []
        metrics_path = args.metrics_path or os.path.join("CF-SASRec", "results", f"{args.dataset}_metrics.json")
        best_metrics_path = os.path.splitext(metrics_path)[0] + ".best.json"
        output_path = args.output_path or os.path.join("RQ-VAE", "ckpt", f"{args.dataset}-{args.hidden_units}d-sasrec.pt")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            if args.inference_only:
                break
            epoch_loss = 0.0
            for step in tqdm(range(num_batch), desc=f"epoch {epoch}/{args.num_epochs}", ncols=100):
                u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                time_matrix = np.array(time_matrix)
                pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                for param in model.abs_pos_K_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                for param in model.abs_pos_V_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                for param in model.time_matrix_K_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                for param in model.time_matrix_V_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(num_batch, 1)

            if epoch % args.eval_step == 0 or epoch == args.num_epochs:
                model.eval()
                total_eval_time += time.time() - t0
                valid_metrics = evaluate_full(model, dataset, args, split="valid")
                test_metrics = evaluate_full(model, dataset, args, split="test")
                print(
                    "epoch:%d, loss:%.6f, time:%f(s), valid %s, test %s"
                    % (epoch, avg_loss, total_eval_time, valid_metrics, test_metrics)
                )
                logger.info(
                    "epoch:%d, loss:%.6f, time:%f(s), valid %s, test %s",
                    epoch, avg_loss, total_eval_time, valid_metrics, test_metrics,
                )

                metrics_item = {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "time": total_eval_time,
                    "valid": valid_metrics,
                    "test": test_metrics,
                }
                metrics_history.append(metrics_item)
                with open(metrics_path, "w") as mf:
                    json.dump(metrics_history, mf, indent=2)

                log_file.write(json.dumps(metrics_item) + "\n")
                log_file.flush()

                valid_ndcg = valid_metrics.get("NDCG@10", valid_metrics.get("NDCG@%d" % max(args.topk), 0.0))
                if valid_ndcg > best_valid_ndcg:
                    best_valid_ndcg = valid_ndcg
                    best_metrics_item = metrics_item
                    stale_eval_count = 0
                    torch.save(model.state_dict(), best_state_path)
                    with open(best_metrics_path, "w") as bf:
                        json.dump(best_metrics_item, bf, indent=2)
                    best_matrix = export_item_embeddings(model, output_path)
                    print(
                        "best valid NDCG@10 improved to %.6f, saved model=%s, embeddings=%s, shape=%s"
                        % (best_valid_ndcg, best_state_path, output_path, tuple(best_matrix.shape))
                    )
                    logger.info(
                        "best valid NDCG@10 improved to %.6f, saved model=%s, embeddings=%s, shape=%s",
                        best_valid_ndcg, best_state_path, output_path, tuple(best_matrix.shape),
                    )
                else:
                    stale_eval_count += 1
                    print(
                        "early stop counter: %d/%d, best valid NDCG@10=%.6f"
                        % (stale_eval_count, args.patience, best_valid_ndcg)
                    )
                    logger.info(
                        "early stop counter: %d/%d, best valid NDCG@10=%.6f",
                        stale_eval_count, args.patience, best_valid_ndcg,
                    )

                t0 = time.time()
                model.train()
                if args.patience > 0 and stale_eval_count >= args.patience:
                    print("early stopping at epoch %d" % epoch)
                    logger.info("early stopping at epoch %d", epoch)
                    break
            else:
                print("epoch:%d, loss:%.6f" % (epoch, avg_loss))

            if epoch == args.num_epochs:
                fname = "TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(run_dir, fname))

        if os.path.exists(best_state_path):
            model.load_state_dict(torch.load(best_state_path, map_location=args.device))
        matrix = export_item_embeddings(model, output_path)
        if best_metrics_item is not None:
            with open(best_metrics_path, "w") as bf:
                json.dump(best_metrics_item, bf, indent=2)
            print("best result by valid NDCG@10: %s" % best_metrics_item)
            print("saved best metrics to %s" % best_metrics_path)
            logger.info("best result by valid NDCG@10: %s", best_metrics_item)
            logger.info("saved best metrics to %s", best_metrics_path)
        print("exported item embeddings to %s, shape=%s" % (output_path, tuple(matrix.shape)))
        print("Done")
        logger.info("exported item embeddings to %s, shape=%s", output_path, tuple(matrix.shape))
        logger.info("Done")
    finally:
        log_file.close()
        if sampler is not None:
            sampler.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
