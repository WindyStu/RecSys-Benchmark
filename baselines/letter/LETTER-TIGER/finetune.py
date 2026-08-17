import argparse
import logging
import os

from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Tokenizer, T5Config
from modeling_letter import LETTER
# import wandb
from utils import *
from collator import Collator


class KeyLoggingCallback(transformers.TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def _clean_logs(logs):
        clean = {}
        for key, value in logs.items():
            if hasattr(value, "item"):
                value = value.item()
            clean[key] = value
        return clean

    def on_train_begin(self, args, state, control, **kwargs):
        eval_strategy = getattr(args, "evaluation_strategy", getattr(args, "eval_strategy", None))
        self.logger.info(
            "Trainer loop started [epochs=%s, max_steps=%s, logging_steps=%s, eval_strategy=%s]",
            args.num_train_epochs,
            state.max_steps,
            args.logging_steps,
            eval_strategy,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        clean_logs = self._clean_logs(logs)
        if "eval_loss" in clean_logs:
            self.logger.info(
                "eval [step=%d, epoch=%s, metrics=%s]",
                state.global_step,
                clean_logs.get("epoch", state.epoch),
                clean_logs,
            )
            return

        if "loss" in clean_logs:
            interval = max(int(args.logging_steps) * 100, 1000)
            if state.global_step == 1 or state.global_step % interval == 0:
                self.logger.info(
                    "train [step=%d/%s, epoch=%s, loss=%s, learning_rate=%s]",
                    state.global_step,
                    state.max_steps,
                    clean_logs.get("epoch", state.epoch),
                    clean_logs.get("loss"),
                    clean_logs.get("learning_rate"),
                )

    def on_save(self, args, state, control, **kwargs):
        self.logger.info(
            "checkpoint saved [step=%d, best_model=%s, best_metric=%s]",
            state.global_step,
            state.best_model_checkpoint,
            state.best_metric,
        )

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.info(
            "Trainer loop finished [step=%d, best_model=%s, best_metric=%s]",
            state.global_step,
            state.best_model_checkpoint,
            state.best_metric,
        )


def _maybe_distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def run_test_after_train(args):
    from test import test

    test_args = argparse.Namespace(**vars(args))
    test_args.ckpt_path = args.output_dir
    test_args.gpu_id = 0
    test_args.filter_items = True
    test_args.sample_num = -1
    test_args.test_prompt_ids = "0"
    test_args.test_task = "SeqRec"
    test_args.results_file = os.path.join(
        "./results",
        args.dataset,
        "test_{}.json".format(get_local_time()),
    )

    print("training finished; start final test")
    print("test results file:", test_args.results_file)
    test(test_args)

def train(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    key_logger = logging.getLogger("key.null")
    if local_rank == 0:
        key_logger, key_log_file = setup_key_logger(args.log_file, args.dataset, "tiger_train")
        key_logger.info("key log file: %s", key_log_file)
        key_logger.info("TIGER training started [dataset=%s, output_dir=%s]", args.dataset, args.output_dir)
        key_logger.info("args: %s", vars(args))
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)


    if local_rank == 0:
        key_logger.info("loading base model config/tokenizer from %s", args.base_model)
    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    args.deepspeed = None
    gradient_checkpointing= False


    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        key_logger.info(
            "datasets loaded [train=%d, valid=%d, added_tokens=%d]",
            len(train_data),
            len(valid_data),
            add_num,
        )
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[100])
        print(valid_data[100])


    collator = Collator(args, tokenizer)
    model = LETTER(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if local_rank == 0:
        key_logger.info("model built [%s]", model.__class__.__name__)
        print(model)


    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True


    callbacks = [EarlyStoppingCallback(early_stopping_patience=20)]
    if local_rank == 0:
        callbacks.append(KeyLoggingCallback(key_logger))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            # fp16=args.fp16,
            # bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            # gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            # deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=[],
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks
    )
    model.config.use_cache = False


    if local_rank == 0:
        key_logger.info("Trainer.train started [resume_from_checkpoint=%s]", args.resume_from_checkpoint)
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
    if local_rank == 0:
        key_logger.info("training artifacts saved [output_dir=%s]", args.output_dir)
    _maybe_distributed_barrier()

    if local_rank == 0:
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        key_logger.info("final test started")
        run_test_after_train(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()
    
    train(args)
