import argparse
import json
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data import SeqRecDataset

def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="./ckpt/TIGER",help="basic model path")

    parser.add_argument("--output_dir", type=str, default="./ckpt",
                        help="The output directory")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str,
                        default="/nfsshare/home/liujingyan/data/CDSR/data",
                        help="data directory")
    parser.add_argument("--tasks", type=str, default="seqrec",
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".llamaindex-sk4-sk.json", help="the item indices file")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1",
                        help="the number of sampling prompts for each task")
    parser.add_argument("--train_data_sample_num", type=str, default="-1",
                        help="the number of sampling prompts for each task")

    # arguments related for evaluation
    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")

    return parser

def parse_stage2_args(parser):
    """Stage 2 LETTER-TIGER specific arguments for multi-stage training"""
    parser.add_argument("--stage", type=str, choices=["pretrain", "lora", "test"],
                        default="pretrain",
                        help="Training stage: pretrain (all domains), lora (domain-specific), test")
    parser.add_argument("--datasets", type=str, nargs='+',
                        default=["asc", "ape", "dbm", "ghk"],
                        help="Cross-domain pair names for pretraining")
    parser.add_argument("--target_domain", type=str, default=None,
                        help="Target domain name for domain-specific LoRA fine-tuning")
    parser.add_argument("--pretrained_model", type=str, default="./ckpt/letter_pretrain",
                        help="Path to pretrained LETTER base model (for lora/test modes)")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    parser.add_argument("--domain_label", type=int, default=None,
                        help="Domain label in map_item.txt (0 or 1) for filtering")
    return parser


def parse_generation_eval_args(parser):
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--eval_log_step", type=int, default=100,
                        help="print running test metrics every N batches; <=0 means final only")

    return parser


def parse_train_args(parser):

    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser = parse_generation_eval_args(parser)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--wandb_run_name", type=str, default="default")
    parser.add_argument("--temperature", type=float, default=1.0)

    return parser

def parse_test_args(parser):

    parser = parse_generation_eval_args(parser)

    parser.add_argument("--ckpt_path", type=str,
                        default="./ckpt",
                        help="The checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=True,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--test_task", type=str, default="SeqRec")


    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args, target_domain=None):
    """Load train/valid datasets. If target_domain is set, filter by domain."""

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num, data_sample_num in zip(tasks, train_prompt_sample_num, train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(
                args, mode="train",
                prompt_sample_num=prompt_sample_num, sample_num=data_sample_num,
                target_domain=target_domain
            )
        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args, "valid", args.valid_prompt_sample_num,
                               target_domain=target_domain)

    return train_data, valid_data


def load_pretrain_datasets(args):
    """Load and merge training data from all cross-domain pairs for joint pretraining."""
    all_train_datasets = []
    first_valid_data = None

    for pair_name in args.datasets:
        # Temporarily set dataset to current pair
        pair_args = argparse.Namespace(**vars(args))
        pair_args.dataset = pair_name

        task = "seqrec"
        train_prompt_sample_num = 1
        train_data_sample_num = -1

        dataset = SeqRecDataset(
            pair_args, mode="train",
            prompt_sample_num=train_prompt_sample_num,
            sample_num=train_data_sample_num,
            target_domain=None  # load all users for pretraining
        )
        all_train_datasets.append(dataset)

        if first_valid_data is None:
            first_valid_data = SeqRecDataset(
                pair_args, "valid", 2,
                target_domain=None
            )

    train_data = ConcatDataset(all_train_datasets)
    valid_data = first_valid_data

    return train_data, valid_data

def load_test_dataset(args, target_domain=None):

    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num,
                                  target_domain=target_domain)
    else:
        raise NotImplementedError

    return test_data

def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
