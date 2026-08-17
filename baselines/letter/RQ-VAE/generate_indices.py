import collections
import json
import logging
import math
import datetime

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
import argparse
import os

def setup_key_logger(log_file, dataset):
    if log_file is None:
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        module_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(module_dir, "log", f"{dataset}_generate_indices_{run_ts}.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = logging.getLogger("generate_indices")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_file

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset')
    parser.add_argument("--root_path", type=str,default="../checkpoint/", help='root path')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='full checkpoint path')
    parser.add_argument('--output_file', type=str, default=None, help='full output index json path')
    parser.add_argument('--device', type=str, default='auto', help='cuda:0, cpu, or auto')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='override dataloader workers')
    parser.add_argument('--log_file', type=str, default=None, help='key log file path')


    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
logger, key_log_file = setup_key_logger(args_setting.log_file, dataset)
if args_setting.checkpoint_path:
    ckpt_path = args_setting.checkpoint_path
else:
    ckpt_path = os.path.join(
        args_setting.root_path,
        f'alpha{args_setting.alpha}-beta{args_setting.beta}',
        args_setting.checkpoint,
    )

output_dir = f"./data/{dataset}/"
output_file = f"{dataset}.index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
output_file = os.path.join(output_dir,output_file)
if args_setting.output_file:
    output_file = args_setting.output_file
output_dirname = os.path.dirname(output_file)
if output_dirname:
    os.makedirs(output_dirname, exist_ok=True)

if args_setting.device == "auto":
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
elif args_setting.device.startswith("cuda") and not torch.cuda.is_available():
    print(f"Requested device={args_setting.device}, but CUDA is not available. Falling back to CPU.")
    device_name = "cpu"
else:
    device_name = args_setting.device
device = torch.device(device_name)
logger.info("generate_indices started [dataset=%s, checkpoint=%s, output=%s, device=%s]", dataset, ckpt_path, output_file, device_name)

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path)
logger.info("Loaded embedding data [items=%d, dim=%d, data_path=%s]", len(data), data.dim, args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  beta=getattr(args, "beta", 0.001),
                  use_diversity=not getattr(args, "no_diversity", False),
                  )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()

num_workers = args_setting.num_workers if args_setting.num_workers is not None else args.num_workers
data_loader = DataLoader(data,num_workers=num_workers,
                             batch_size=args_setting.batch_size, shuffle=False,
                             pin_memory=(device.type == "cuda"))

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    # x = data.cpu().detach().numpy()
    # data = self.embedding.weight.cpu().detach().numpy()
    x = data
    n_samples = len(data)
    size_min = min(max(n_samples // (n_clusters * 2), 1), 10)
    size_max = max(size_min * 4, math.ceil(n_samples / n_clusters))
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

labels = {str(i): [] for i in range(len(model.rq.vq_layers))}
if model.rq.use_diversity and model.rq.beta > 0:
    embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]
    for idx, emb in enumerate(embs):
        centers, label = constrained_km(emb)
        labels[str(idx)] = label
for d in tqdm(data_loader):
    d, emb_idx = d[0], d[1]
    d = d.to(device)
    
    # indices = model.get_indices(d, use_sk=False)
    indices = model.get_indices(d, labels,use_sk=False)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

# model.rq.vq_layers[-1].sk_epsilon = 0.1
tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    for collision_items in collision_item_groups:
        d = data[collision_items]
        d = d[0].to(device)
        indices = model.get_indices(d, labels, use_sk=True)

        # indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)
logger.info("All indices number: %d", len(all_indices))
logger.info("Max number of conflicts: %d", max(get_indices_count(all_indices_str).values()))
logger.info("Collision Rate %s", (tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)



with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
logger.info("Saved index file: %s", output_file)
