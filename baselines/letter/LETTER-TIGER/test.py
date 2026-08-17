import argparse
import json
import os
import sys
from typing import List

import torch
import transformers
# from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie


def test(args):

    set_seed(args.seed)
    logger, key_log_file = setup_key_logger(args.log_file, args.dataset, "tiger_test")
    logger.info("key log file: %s", key_log_file)
    logger.info(
        "TIGER test started [dataset=%s, ckpt_path=%s, results_file=%s]",
        args.dataset,
        args.ckpt_path,
        args.results_file,
    )
    logger.info("args: %s", vars(args))
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda",args.gpu_id)

    tokenizer = T5Tokenizer.from_pretrained(
        args.ckpt_path,
        model_max_length=512,
        local_files_only=True,
    )
    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())

    print("add {} new token.".format(add_num))
    print("data num:", len(train_data))
    logger.info(
        "datasets loaded [train=%d, valid=%d, added_tokens=%d]",
        len(train_data),
        len(valid_data),
        add_num,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
        device_map=device_map,
        local_files_only=True,
    )
    if add_num > 0:
        model.resize_token_embeddings(len(tokenizer))

    prompt_ids = [0]

    test_data = load_test_dataset(args)

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
    logger.info(
        "test dataset loaded [test=%d, all_items=%d, batch_size=%d, num_beams=%d]",
        len(test_data),
        len(all_items),
        args.test_batch_size,
        args.num_beams,
    )

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
                    logger.info(
                        "eval step %d/%d [total=%d, metrics=%s]",
                        step + 1,
                        num_test_batches,
                        total,
                        temp,
                    )

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")
            logger.info("Prompt %s results: %s", prompt_id, metrics_results)

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
    logger.info("Mean results: %s", mean_results)
    logger.info("Min results: %s", min_results)
    logger.info("Max results: %s", max_results)


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
    logger.info("saved results file: %s", args.results_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)
