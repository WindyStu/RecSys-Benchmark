import argparse
import glob
import os

from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Tokenizer, T5Config
from modeling_letter import LETTER
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
# import wandb
from utils import *
from collator import Collator


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
    test_args.pretrained_model = getattr(args, 'pretrained_model', None)
    test_args.results_file = os.path.join(
        "./results",
        getattr(args, 'target_domain', args.dataset),
        "test_{}.json".format(get_local_time()),
    )

    print("training finished; start final test")
    print("test results file:", test_args.results_file)
    test(test_args)


def _setup_device_env():
    """Setup device and DDP environment."""
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(f"CUDA available: {torch.cuda.is_available()}, DDP: {ddp}, world_size: {world_size}")
    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)
    return device, device_map, ddp, local_rank


def _create_training_args(args):
    """Create HuggingFace TrainingArguments from parsed args."""
    return transformers.TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_step,
        optim=args.optim,
        eval_strategy=args.save_and_eval_strategy,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=2,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None,
        report_to=[],
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
    )


# ============================================================
# Checkpoint helpers
# ============================================================

def _pretrain_exists(output_dir):
    """Check if pretrained model weights already exist in output_dir."""
    return (
        os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) or
        os.path.exists(os.path.join(output_dir, "model.safetensors"))
    )

def _lora_exists(output_dir):
    """Check if LoRA adapter weights already exist in output_dir."""
    return (
        os.path.exists(os.path.join(output_dir, "adapter_config.json")) and
        (os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")) or
         os.path.exists(os.path.join(output_dir, "adapter_model.bin")))
    )


# ============================================================
# Phase 1: Joint Pretraining
# ============================================================

def train_pretrain(args):
    """Joint pretraining on all cross-domain pairs."""
    set_seed(args.seed)

    device, device_map, ddp, local_rank = _setup_device_env()

    # Check if already trained
    if _pretrain_exists(args.output_dir):
        if local_rank == 0:
            print(f"Pretrained weights already exist at {args.output_dir}, skipping training.")
        return

    ensure_dir(args.output_dir)
    if local_rank == 0:
        print(vars(args))

    # Load T5 config and tokenizer from base model
    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )

    # Load all cross-domain pair datasets
    train_data, valid_data = load_pretrain_datasets(args)

    # Collect and add all new tokens (SID tokens like <a_X>, <b_Y>, ...)
    all_new_tokens = set()
    for dataset in train_data.datasets:
        all_new_tokens.update(dataset.get_new_tokens())
    all_new_tokens = sorted(list(all_new_tokens))

    add_num = tokenizer.add_tokens(all_new_tokens)
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"Added {add_num} new tokens, total vocab: {len(tokenizer)}")
        print(f"Train data size: {len(train_data)}, Valid data size: {len(valid_data)}")
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(f"Sample train data: {train_data[100]}")
        print(f"Sample valid data: {valid_data[100]}")

    # Create LETTER model (no LoRA — full training)
    collator = Collator(args, tokenizer)
    model = LETTER(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if local_rank == 0:
        print(model)

    training_args = _create_training_args(args)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
    _maybe_distributed_barrier()

    if local_rank == 0:
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Skip auto-test after pretrain (will test after LoRA)
        print(f"Pretraining complete. Model saved to {args.output_dir}")


# ============================================================
# Phase 2: Domain-Specific LoRA Fine-tuning
# ============================================================

def train_lora(args):
    """Domain-specific LoRA fine-tuning."""
    set_seed(args.seed)

    device, device_map, ddp, local_rank = _setup_device_env()

    # Check if already trained
    if _lora_exists(args.output_dir):
        if local_rank == 0:
            print(f"LoRA weights already exist at {args.output_dir}, skipping training.")
            run_test_after_train(args)
            return

    ensure_dir(args.output_dir)
    if local_rank == 0:
        print(vars(args))

    # Load config and tokenizer from pretrained model
    pretrained_path = args.pretrained_model
    config = T5Config.from_pretrained(pretrained_path)
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_path,
        model_max_length=512,
    )

    # Load domain-filtered data
    train_data, valid_data = load_datasets(args, target_domain=args.target_domain)

    # Add new tokens (SID tokens for this domain)
    all_new_tokens = set()
    for dataset in train_data.datasets:
        all_new_tokens.update(dataset.get_new_tokens())
    all_new_tokens = sorted(list(all_new_tokens))

    add_num = tokenizer.add_tokens(all_new_tokens)
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"Target domain: {args.target_domain}")
        print(f"Added {add_num} new tokens, total vocab: {len(tokenizer)}")
        print(f"Train data size: {len(train_data)}, Valid data size: {len(valid_data)}")
        print(f"Sample train data: {train_data[100] if len(train_data) > 100 else train_data[0]}")

    # Load pretrained LETTER base model
    # CRITICAL ORDERING: load base -> resize embeddings -> wrap with LoRA
    base_model = LETTER.from_pretrained(pretrained_path, config=config)
    base_model.set_hyper(args.temperature)

    # Resize embeddings BEFORE LoRA wrapping (so PEFT freezes the resized embeddings)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(device)

    # LoRA configuration for T5
    lora_config = LoraConfig(
        r=args.lora_r if hasattr(args, 'lora_r') else 8,
        lora_alpha=args.lora_alpha if hasattr(args, 'lora_alpha') else 16,
        lora_dropout=args.lora_dropout if hasattr(args, 'lora_dropout') else 0.1,
        bias='none',
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v", "k", "o", "wi", "wo"],
    )

    # Wrap with PEFT — auto-freezes all base parameters
    model = get_peft_model(base_model, lora_config)

    # Add domain-specific adapter
    adapter_name = args.target_domain or "default"
    model.add_adapter(adapter_name, lora_config)
    model.set_adapter(adapter_name)
    model.to(device)

    if local_rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"LoRA trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        print(model)

    # Setup trainer
    collator = Collator(args, tokenizer)
    training_args = _create_training_args(args)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()

    # Save: PEFT model.save_pretrained() saves only adapter weights
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    _maybe_distributed_barrier()

    if local_rank == 0:
        del trainer
        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_test_after_train(args)


# ============================================================
# Phase 3: Test Only (LoRA mode)
# ============================================================

def run_test_only(args):
    """Run evaluation only using a trained checkpoint."""
    from test import test

    test_args = argparse.Namespace(**vars(args))
    test_args.ckpt_path = args.pretrained_model or args.output_dir
    test_args.gpu_id = 0
    test_args.filter_items = True
    test_args.sample_num = -1
    test_args.test_prompt_ids = "0"
    test_args.test_task = "SeqRec"
    test_args.pretrained_model = args.pretrained_model

    if args.target_domain:
        test_args.results_file = os.path.join(
            "./results", args.target_domain,
            "test_{}.json".format(get_local_time()),
        )
    else:
        test_args.results_file = os.path.join(
            "./results", args.dataset,
            "test_{}.json".format(get_local_time()),
        )

    print(f"Test mode: ckpt_path={test_args.ckpt_path}")
    print(f"Test results file: {test_args.results_file}")
    test(test_args)


# ============================================================
# Single-domain training (backward-compatible fallback)
# ============================================================

def train_single(args):
    """Single-domain training (original behavior, for backward compatibility)."""
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device, device_map, ddp, local_rank = _setup_device_env()
    if local_rank == 0:
        print(vars(args))

    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    args.deepspeed = None
    gradient_checkpointing = False

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
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
        print(model)

    training_args = _create_training_args(args)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
    _maybe_distributed_barrier()

    if local_rank == 0:
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_test_after_train(args)


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LETTER-TIGER Stage2 Training')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_stage2_args(parser)

    args = parser.parse_args()

    if args.stage == "pretrain":
        train_pretrain(args)
    elif args.stage == "lora":
        train_lora(args)
    elif args.stage == "test":
        run_test_only(args)
    else:
        # Fallback: single-domain training (original behavior)
        train_single(args)
