import logging
import json
import math
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
import wandb
from datasets import EmbDataset
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}


    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            # optimizer = optim.AdamW([
            # {'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay':weight_decay}, 
            # {'params': self.awl.parameters(), 'weight_decay':0}
            # ])
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
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
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path)
        message = "VQ initialization dataset loaded [items=%d, dim=%d, data_path=%s]" % (
            len(original_data),
            original_data.dim,
            self.args.data_path,
        )
        print(message, flush=True)
        self.logger.info(message)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            message = "VQ initialization full batch ready [batch=%d, device=%s]" % (len(data), self.device)
            print(message, flush=True)
            self.logger.info(message)
            self.model.vq_initialization(data)
            print("VQ initialization finished", flush=True)
            self.logger.info("VQ initialization finished")

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        use_diversity = self.model.rq.use_diversity and self.model.rq.beta > 0
        if use_diversity:
            embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]

            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels

        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices, dense_out = self.model(data, self.labels)

            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(out, rq_loss, emb_idx, dense_out, xs=data)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cf_loss += (cf_loss.item() if cf_loss != 0 else cf_loss)
            total_quant_loss += quant_loss.item()

        return total_loss, total_recon_loss, total_cf_loss, total_quant_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()

        num_sample = 0
        use_diversity = self.model.rq.use_diversity and self.model.rq.beta > 0
        if use_diversity:
            embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):

            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data, self.labels)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set))/num_sample
        # balance_score = self.balance_overall(tokens_appearance)
        # wandb.log({"collision_rate": collision_rate, "balance_score": 0})


        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(
        self,
        epoch_idx,
        s_time,
        e_time,
        loss,
        recon_loss,
        cf_loss,
        quant_loss,
        num_batches,
    ):
        epoch_no = epoch_idx + 1
        progress = 100.0 * epoch_no / self.epochs
        avg_loss = loss / max(num_batches, 1)
        avg_recon_loss = recon_loss / max(num_batches, 1)
        avg_cf_loss = cf_loss / max(num_batches, 1)
        avg_quant_loss = quant_loss / max(num_batches, 1)
        if epoch_no % self.eval_step == 0 or epoch_no == self.epochs:
            eval_info = "eval_now"
        else:
            next_eval = min(((epoch_no // self.eval_step) + 1) * self.eval_step, self.epochs)
            eval_info = "next_eval=%d" % next_eval
        return (
            "epoch %d/%d training "
            "[progress=%.2f%%, batches=%d, time=%.2fs, avg_loss=%.6f, avg_recon=%.6f, avg_quant=%.6f, avg_cf=%.6f, %s]"
            % (
                epoch_no,
                self.epochs,
                progress,
                num_batches,
                e_time - s_time,
                avg_loss,
                avg_recon_loss,
                avg_quant_loss,
                avg_cf_loss,
                eval_info,
            )
        )

    def fit(self, data):

        no_improve_eval_count = 0
        self.logger.info(
            "RQ-VAE training started "
            "[epochs=%d, batches_per_epoch=%d, eval_step=%d, patience=%d, ckpt_dir=%s]"
            % (self.epochs, len(data), self.eval_step, self.args.patience, self.ckpt_dir)
        )
        self.vq_init()
        for epoch_idx in range(self.epochs):
            is_eval_epoch = (epoch_idx + 1) % self.eval_step == 0 or epoch_idx + 1 == self.epochs
            # train
            training_start_time = time()
            train_loss, train_recon_loss, cf_loss, quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            if is_eval_epoch:
                train_loss_output = self._generate_train_loss_output(
                    epoch_idx,
                    training_start_time,
                    training_end_time,
                    train_loss,
                    train_recon_loss,
                    cf_loss,
                    quant_loss,
                    len(data),
                )
                self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if is_eval_epoch:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                improved = (self.best_collision_rate - collision_rate) > self.args.early_stop_min_delta
                if improved:
                    self.best_collision_rate = collision_rate
                    no_improve_eval_count = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    no_improve_eval_count += 1

                valid_end_time = time()
                valid_score_output = (
                    "epoch %d/%d evaluating "
                    "[time=%.2fs, collision_rate=%f, best_collision_rate=%f, early_stop=%d/%d]"
                ) % (
                    epoch_idx + 1,
                    self.epochs,
                    valid_end_time - valid_start_time,
                    collision_rate,
                    self.best_collision_rate,
                    no_improve_eval_count,
                    self.args.patience,
                )

                self.logger.info(valid_score_output)

                if epoch_idx>2500:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)

                if self.args.patience > 0 and no_improve_eval_count >= self.args.patience:
                    self.logger.info(
                        "Early stopping: epoch %d/%d, no collision-rate improvement for %d eval rounds"
                        % (epoch_idx + 1, self.epochs, no_improve_eval_count)
                    )
                    break

        self.logger.info(
            "RQ-VAE training finished [best_loss=%.6f, best_collision_rate=%f]"
            % (self.best_loss, self.best_collision_rate)
        )

        return self.best_loss, self.best_collision_rate
