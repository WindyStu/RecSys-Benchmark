import sys
import copy
import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos(time_seq, time_span):
    
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]
    
        idx = maxlen - 1
        ts = set(map(lambda x: x[0],user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10,n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set: # float as map key?
        time_map[time] = int(round(float(time-time_min)))
    return time_map

def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1
    
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)

def data_partition(fname):
    letter_inter_path = os.path.join("data", fname, f"{fname}.inter.json")
    if os.path.exists(letter_inter_path):
        return data_partition_letter(fname, letter_inter_path)

    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1
        item_count[i]+=1
    f.close()
    f = open('data/%s.txt' % fname, 'r') # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u]<5 or item_count[i]<5: # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]


def data_partition_letter(fname, inter_path):
    item_path = os.path.join("data", fname, f"{fname}.item.json")
    with open(inter_path, "r", encoding="utf-8") as f:
        inter = json.load(f)

    if os.path.exists(item_path):
        with open(item_path, "r", encoding="utf-8") as f:
            item2feature = json.load(f)
        itemnum = max(int(item_id) for item_id in item2feature.keys()) + 1
    else:
        itemnum = 0
        for seq in inter.values():
            if seq:
                itemnum = max(itemnum, max(int(item) for item in seq) + 1)

    user_train = {}
    user_valid = {}
    user_test = {}
    max_time = 0

    for uid, raw_seq in sorted(inter.items(), key=lambda x: int(x[0])):
        user = int(uid) + 1
        events = [[int(item) + 1, idx + 1] for idx, item in enumerate(raw_seq)]
        max_time = max(max_time, len(events))
        if len(events) < 3:
            user_train[user] = events
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = events[:-2]
            user_valid[user] = [events[-2]]
            user_test[user] = [events[-1]]

    usernum = len(user_train)
    print(f"Preparing LETTER data from {inter_path}")
    print(f"usernum={usernum}, itemnum={itemnum}, timenum={max_time}")
    return [user_train, user_valid, user_test, usernum, itemnum, max_time]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x[0],train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix],item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix],item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def _context_to_arrays(context, maxlen, time_span):
    seq = np.zeros([maxlen], dtype=np.int32)
    time_seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for item, timestamp in reversed(context):
        seq[idx] = item
        time_seq[idx] = timestamp
        idx -= 1
        if idx == -1:
            break
    time_matrix = computeRePos(time_seq, time_span)
    return seq, time_matrix


@torch.no_grad()
def evaluate_full(model, dataset, args, split="valid"):
    train, valid, test, usernum, itemnum, timenum = copy.deepcopy(dataset)
    model.eval()

    topks = sorted(set(getattr(args, "topk", [5, 10])))
    metrics = {}
    for k in topks:
        metrics[f"HR@{k}"] = 0.0
        metrics[f"NDCG@{k}"] = 0.0

    users = list(range(1, usernum + 1))
    batch_size = getattr(args, "eval_batch_size", 256)
    valid_user = 0

    for start in tqdm(range(0, len(users), batch_size), desc=f"eval-{split}", ncols=100):
        batch_users = []
        seqs = []
        time_matrices = []
        targets = []
        rated_sets = []

        for u in users[start : start + batch_size]:
            if split == "valid":
                if len(train[u]) < 1 or len(valid[u]) < 1:
                    continue
                context = train[u]
                target = valid[u][0][0]
                rated = set(map(lambda x: x[0], train[u]))
            elif split == "test":
                if len(train[u]) < 1 or len(valid[u]) < 1 or len(test[u]) < 1:
                    continue
                context = train[u] + valid[u]
                target = test[u][0][0]
                rated = set(map(lambda x: x[0], train[u] + valid[u]))
            else:
                raise ValueError(f"Unknown split: {split}")

            seq, time_matrix = _context_to_arrays(context, args.maxlen, args.time_span)
            batch_users.append(u)
            seqs.append(seq)
            time_matrices.append(time_matrix)
            targets.append(target)
            rated.add(0)
            rated_sets.append(rated)

        if not batch_users:
            continue

        log_feats = model.seq2feats(
            np.array(batch_users),
            np.array(seqs),
            np.array(time_matrices),
        )
        final_feat = log_feats[:, -1, :]
        item_emb = model.item_emb.weight[1:]
        scores = torch.matmul(final_feat, item_emb.transpose(0, 1))

        for row, rated in enumerate(rated_sets):
            mask_items = [item - 1 for item in rated if item > 0]
            if mask_items:
                scores[row, torch.LongTensor(mask_items).to(args.device)] = -float("inf")

        target_idx = torch.LongTensor([target - 1 for target in targets]).to(args.device)
        target_scores = scores.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1).add(1).detach().cpu().numpy()

        for rank in ranks:
            valid_user += 1
            for k in topks:
                if rank <= k:
                    metrics[f"HR@{k}"] += 1.0
                    metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 1)

    for key in metrics:
        metrics[key] = float(metrics[key] / max(valid_user, 1))
    metrics["users"] = int(valid_user)
    return metrics
