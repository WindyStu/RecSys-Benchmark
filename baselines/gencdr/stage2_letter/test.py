import argparse
import json
import os
import sys
from typing import List

import torch
import transformers
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie


def _load_model_and_tokenizer(args, device_map, add_num):
    """Load model and tokenizer, handling both full model and LoRA adapter checkpoints."""

    # Try loading tokenizer from ckpt_path first, fallback to pretrained_model
    ckpt_path = args.ckpt_path
    pretrained_path = getattr(args, 'pretrained_model', None)

    # Check if this is a LoRA adapter checkpoint
    adapter_config_path = os.path.join(ckpt_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print(f"Detected LoRA adapter checkpoint at {ckpt_path}")
        # Load tokenizer from adapter path (may have been saved there)
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                ckpt_path, model_max_length=512, local_files_only=True,
            )
        except Exception:
            print(f"Tokenizer not found in adapter path, loading from base: {pretrained_path}")
            tokenizer = T5Tokenizer.from_pretrained(
                pretrained_path, model_max_length=512,
            )

        # Load base model
        base_model_path = pretrained_path or ckpt_path
        print(f"Loading base LETTER model from: {base_model_path}")
        base_model = T5ForConditionalGeneration.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            device_map=device_map,
            local_files_only=True,
        )
        if add_num > 0:
            base_model.resize_token_embeddings(len(tokenizer))

        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {ckpt_path}")
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        print("LoRA adapter loaded successfully")
    else:
        # Full model checkpoint (pretrain or single-domain)
        print(f"Loading full model checkpoint from {ckpt_path}")
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                ckpt_path, model_max_length=512, local_files_only=True,
            )
        except Exception:
            tokenizer = T5Tokenizer.from_pretrained(
                pretrained_path, model_max_length=512,
            )

        model = T5ForConditionalGeneration.from_pretrained(
            ckpt_path,
            low_cpu_mem_usage=True,
            device_map=device_map,
            local_files_only=True,
        )
        if add_num > 0:
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def test(args):

    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)

    # Load data to determine new tokens
    target_domain = getattr(args, 'target_domain', None)
    train_data, valid_data = load_datasets(args, target_domain=target_domain)

    # Collect and add new tokens
    all_new_tokens = set()
    for dataset in train_data.datasets:
        all_new_tokens.update(dataset.get_new_tokens())
    all_new_tokens = sorted(list(all_new_tokens))

    # Load model and tokenizer (handles LoRA adapter auto-detection)
    model, tokenizer = _load_model_and_tokenizer(args, device_map, 0)

    add_num = tokenizer.add_tokens(all_new_tokens)
    if add_num > 0:
        print(f"Added {add_num} new tokens, total vocab: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"Data num: {len(train_data)}")

    prompt_ids = [0]

    test_data = load_test_dataset(args, target_domain=target_domain)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()


    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)


    print("data num:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:

            
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            num_test_batches = len(test_loader)
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                total += len(targets)
                if step == 0:
                    print("first batch input shape:", inputs["input_ids"].shape)
                    print("first batch targets:", targets[: min(3, len(targets))])

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    # max_length=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                # print(batch_metrics_res)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                should_log = (
                    args.eval_log_step > 0
                    and ((step + 1) % args.eval_log_step == 0 or step + 1 == num_test_batches)
                )
                if should_log:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print("eval step %d/%d, total=%d, metrics=%s" % (step + 1, num_test_batches, total, temp))

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")


    save_data={}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    results_dir = os.path.dirname(args.results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = parse_stage2_args(parser)

    args = parser.parse_args()

    test(args)
