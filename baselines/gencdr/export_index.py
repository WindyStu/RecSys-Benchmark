"""
export_index.py — Generate .index.json for cross-domain pairs from existing embeddings.

Loads .emb-*-td.npy, trains RQ-VAE, converts codes to SID tokens, saves .index.json.

Usage:
    python export_index.py --pair asc
    python export_index.py --pair ape
    python export_index.py --pair dbm
    python export_index.py --all  # process all three pairs
"""

import os
import json
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.rqvae.rqvae import RQVAE
from quantization.utils import set_weight_decay

# ---- Config ----
# Default data path: server cluster
# Override with --data_path
_DEFAULT_DATA = "/nfsshare/home/liujingyan/data/CDSR/data"

# Map pair name to embedding filename pattern
PAIR_EMB_MAP = {
    "asc": "asc.emb-qwen-api-td.npy",
    "ape": "ape.emb-qwen-api-td.npy",
    "dbm": "dbm.emb-qwen-api-td.npy",
    "ghk": "ghk.emb-qwen-api-td.npy",
}

# RQ-VAE config (matching rqvae_config.yaml but with num_layers=4 for Beauty compatibility)
RQVAE_CONFIG = {
    "hidden_dim": [1024, 512],
    "latent_dim": 256,
    "num_layers": 3,       # 3 RQ-VAE levels
    "code_book_size": 256,  # Use 256 per level (Beauty-style)
    "dropout": 0.0,
    "beta": 0.25,
    "epochs": 500,          # Reduced from 3000 for faster iteration
    "batch_size": 4096,
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0,
    "eval_interval": 100,
}

LEVEL_PREFIXES = ["a", "b", "c", "d"]  # 3 RQ-VAE + 1 dedup = 4 levels


def build_dedup_layer(base_codes_np: np.ndarray, vocab_size: int) -> np.ndarray:
    """Add a deduplication layer to distinguish items with identical code sequences."""
    N = base_codes_np.shape[0]
    groups = defaultdict(list)
    for idx, key in enumerate(map(tuple, base_codes_np)):
        groups[key].append(idx)

    dedup_layer = np.zeros((N, 1), dtype=np.int64)
    max_dup, overflow_count = 0, 0

    for idx_list in groups.values():
        k = len(idx_list)
        max_dup = max(max_dup, k)
        if k > vocab_size:
            local_ids = np.arange(k, dtype=np.int64) % vocab_size
            overflow_count += 1
        else:
            local_ids = np.arange(k, dtype=np.int64)
        dedup_layer[np.array(idx_list), 0] = local_ids

    logging.info(f"  Dedup: max duplicates={max_dup}, overflows={overflow_count}")
    return dedup_layer


def load_embeddings(pair_name: str, data_base: str):
    """Load .emb-qwen-api-td.npy from the pair directory.

    These files have shape (N_items, D) — one embedding per item, no pad row.
    Item ID i directly maps to embedding[i].
    """
    emb_path = os.path.join(data_base, pair_name, PAIR_EMB_MAP[pair_name])
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")

    emb = np.load(emb_path)
    logging.info(f"Loaded embeddings: {emb_path}, shape={emb.shape}")
    return emb, emb.shape[1]


def load_inter_json(pair_name: str, data_base: str):
    """Load .inter.json to get number of unique items."""
    inter_path = os.path.join(data_base, pair_name, f"{pair_name}.inter.json")
    if os.path.exists(inter_path):
        with open(inter_path, 'r') as f:
            inter = json.load(f)
        # Collect all unique item IDs
        all_items = set()
        for items in inter.values():
            all_items.update(items)
        return len(all_items)
    return None


def train_epoch(model, dataloader, optimizer, beta, device):
    model.train()
    total_loss = total_rec = total_commit = 0.0
    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon, commit_loss, _ = model(x)
        loss = F.mse_loss(recon, x) + beta * commit_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_rec += F.mse_loss(recon, x, reduction="mean").item()
        total_commit += commit_loss.item()
    n = len(dataloader)
    return total_loss / n, total_rec / n, total_commit / n


def train_rqvae(model, embeddings, device, config):
    """Train RQ-VAE on item embeddings (no pad row)."""
    model.to(device)
    model_config = config

    optimizer = getattr(torch.optim, model_config["optimizer"])(
        model.parameters(), lr=model_config["lr"]
    )
    set_weight_decay(optimizer, model_config["weight_decay"])

    # No padding — use all embeddings directly
    data = torch.tensor(embeddings, dtype=torch.float32)
    train_data, val_data = train_test_split(data, test_size=0.05, random_state=42)

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=model_config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=model_config["batch_size"], shuffle=False
    )

    n_epochs = model_config["epochs"]
    for epoch in tqdm(range(n_epochs), desc="Training RQ-VAE"):
        train_loss, train_rec, train_commit = train_epoch(
            model, train_loader, optimizer, model_config["beta"], device
        )
        if (epoch + 1) % 100 == 0:
            logging.info(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"loss={train_loss:.4f} rec={train_rec:.4f} commit={train_commit:.4f}"
            )
        if (epoch + 1) % model_config.get("eval_interval", 100) == 0:
            model.eval()
            val_loss = val_rec = val_commit = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    recon, commit_loss, _ = model(x)
                    val_rec += F.mse_loss(recon, x, reduction="mean").item()
                    val_commit += commit_loss.item()
            n_val = len(val_loader)
            logging.info(
                f"  VAL epoch {epoch+1}: rec={val_rec/n_val:.4f} commit={val_commit/n_val:.4f}"
            )


@torch.no_grad()
def generate_codes(model, embeddings, device, batch_size=4096):
    """Generate RQ-VAE codes for all items."""
    model.to(device)
    model.eval()

    data = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)

    codes_list = []
    for batch in tqdm(loader, desc="Generating codes"):
        codes = model.get_codes(batch[0].to(device)).cpu().numpy()
        codes_list.append(codes)

    return np.vstack(codes_list)


def codes_to_sid_tokens(codes: np.ndarray) -> list:
    """Convert integer code matrix to SID token strings.

    codes: (N, num_levels), e.g. [[183, 70, 232, 6], ...]
    Returns: [[\"<a_183>\", \"<b_70>\", \"<c_232>\", \"<d_6>\"], ...]
    """
    num_levels = codes.shape[1]
    assert num_levels <= len(LEVEL_PREFIXES), \
        f"Too many levels ({num_levels}), max {len(LEVEL_PREFIXES)}"

    sid_list = []
    for row in codes:
        tokens = [f"<{LEVEL_PREFIXES[i]}_{int(v)}>" for i, v in enumerate(row)]
        sid_list.append(tokens)
    return sid_list


def export_index_json(pair_name: str, sid_tokens: list, data_base: str, output_path: str = None):
    """Save SID tokens as .index.json compatible with LETTER-TIGER.

    Format: {"0": ["<a_X>", "<b_Y>", "<c_Z>", "<d_W>"], "1": [...], ...}
    Maps local item index (0-based) to SID token list.
    """
    if output_path is None:
        output_path = os.path.join(data_base, pair_name, f"{pair_name}.index.json")

    index = {str(i): tokens for i, tokens in enumerate(sid_tokens)}
    with open(output_path, 'w') as f:
        json.dump(index, f, separators=(',', ':'))

    # Also save a human-readable version
    readable_path = output_path.replace('.index.json', '.index_readable.json')
    with open(readable_path, 'w') as f:
        json.dump(index, f, indent=2)

    logging.info(f"Index saved to: {output_path}")
    logging.info(f"Readable version: {readable_path}")
    logging.info(f"Total items indexed: {len(index)}")
    return output_path


def process_pair(pair_name: str, device: str = "cuda:0", data_base: str = None):
    """Full pipeline for one cross-domain pair."""
    if data_base is None:
        data_base = _DEFAULT_DATA

    logging.info(f"{'='*60}")
    logging.info(f"Processing pair: {pair_name} ({data_base}/{pair_name})")
    logging.info(f"{'='*60}")

    # Step 1: Load embeddings
    embeddings, input_dim = load_embeddings(pair_name, data_base)
    n_items = embeddings.shape[0]  # no pad — count is actual items
    logging.info(f"  Items: {n_items}, Embedding dim: {input_dim}")

    # Step 2: Create RQVAE model
    cfg = RQVAE_CONFIG.copy()
    model = RQVAE(
        input_size=input_dim,
        hidden_sizes=cfg["hidden_dim"],
        latent_size=cfg["latent_dim"],
        num_levels=cfg["num_layers"],
        codebook_size=cfg["code_book_size"],
        dropout=cfg["dropout"],
        latent_loss_weight=cfg["beta"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  RQVAE params: {n_params:,}")

    # Step 3: Train
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    logging.info(f"  Training on: {device_t}")

    ckpt_dir = os.path.join("ckpt", "rqvae_export", pair_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"rqvae-{pair_name}.pth")

    if os.path.exists(ckpt_path):
        logging.info(f"  Loading existing checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device_t))
    else:
        train_rqvae(model, embeddings, device_t, cfg)
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"  Checkpoint saved: {ckpt_path}")

    # Step 4: Generate codes
    base_codes = generate_codes(model, embeddings, device_t, cfg["batch_size"])
    logging.info(f"  Base codes shape: {base_codes.shape}")

    # Step 5: Build dedup layer → 4 levels total
    dedup = build_dedup_layer(base_codes, cfg["code_book_size"])
    final_codes = np.hstack([base_codes, dedup])
    logging.info(f"  Final codes shape: {final_codes.shape}")

    # Step 6: Convert to SID tokens and save
    sid_tokens = codes_to_sid_tokens(final_codes)
    output_path = export_index_json(pair_name, sid_tokens, data_base)

    # Print sample
    logging.info(f"  Sample SIDs:")
    for i in range(min(3, len(sid_tokens))):
        logging.info(f"    Item {i}: {sid_tokens[i]} -> {''.join(sid_tokens[i])}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export .index.json for cross-domain pairs")
    parser.add_argument("--pair", type=str, default=None,
                        help="Process a single pair (asc/ape/dbm)")
    parser.add_argument("--all", action="store_true",
                        help="Process all three pairs")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training (default: cuda:0)")
    parser.add_argument("--data_path", type=str, default=None,
                        help=f"Data base directory (default: {_DEFAULT_DATA})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--code_book_size", type=int, default=None,
                        help="Override codebook size")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    if args.epochs:
        RQVAE_CONFIG["epochs"] = args.epochs
    if args.code_book_size:
        RQVAE_CONFIG["code_book_size"] = args.code_book_size

    if args.all:
        pairs = ["asc", "ape", "dbm", "ghk"]
    elif args.pair:
        pairs = [args.pair]
    else:
        parser.print_help()
        print("\nExample: python export_index.py --all")
        print("         python export_index.py --pair asc --epochs 1000")
        return

    for pair in pairs:
        if pair not in PAIR_EMB_MAP:
            logging.error(f"Unknown pair: {pair}. Known: {list(PAIR_EMB_MAP.keys())}")
            continue
        try:
            data_base = args.data_path or _DEFAULT_DATA
            process_pair(pair, args.device, data_base)
        except Exception as e:
            logging.error(f"Failed to process {pair}: {e}", exc_info=True)

    logging.info("All done!")


if __name__ == "__main__":
    main()
