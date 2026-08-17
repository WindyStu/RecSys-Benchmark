# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc
from typing import Dict

import torch
from generative_recommenders.research.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        # pyrefly: ignore [bad-index]
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class FeatureAugmentedEmbeddingModule(EmbeddingModule):
    """
    Item embedding that augments the main item ID embedding with side features.

    For each item, the total embedding = id_emb[item_id] + sum_i(proj_i(feat_i_emb[item_id])).

    Each of the 4 feature types (a, b, c, d) has its own embedding table (16-dim),
    which is projected to the main item_embedding_dim and summed in.
    """

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        feat_vocab_sizes: Dict[str, int],
        item_to_feat_a: torch.Tensor,
        item_to_feat_b: torch.Tensor,
        item_to_feat_c: torch.Tensor,
        item_to_feat_d: torch.Tensor,
        feat_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim

        # Main item ID embedding
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )

        # Feature embedding tables
        self._feat_a_emb = torch.nn.Embedding(
            feat_vocab_sizes.get("a", 1), feat_embedding_dim, padding_idx=0
        )
        self._feat_b_emb = torch.nn.Embedding(
            feat_vocab_sizes.get("b", 1), feat_embedding_dim, padding_idx=0
        )
        self._feat_c_emb = torch.nn.Embedding(
            feat_vocab_sizes.get("c", 1), feat_embedding_dim, padding_idx=0
        )
        self._feat_d_emb = torch.nn.Embedding(
            feat_vocab_sizes.get("d", 1), feat_embedding_dim, padding_idx=0
        )

        # Linear projections: feat_embedding_dim -> item_embedding_dim
        self._feat_proj_a = torch.nn.Linear(feat_embedding_dim, item_embedding_dim)
        self._feat_proj_b = torch.nn.Linear(feat_embedding_dim, item_embedding_dim)
        self._feat_proj_c = torch.nn.Linear(feat_embedding_dim, item_embedding_dim)
        self._feat_proj_d = torch.nn.Linear(feat_embedding_dim, item_embedding_dim)

        # Register feature index tensors as buffers (not parameters, no grad)
        self.register_buffer("_item_to_feat_a", item_to_feat_a)
        self.register_buffer("_item_to_feat_b", item_to_feat_b)
        self.register_buffer("_item_to_feat_c", item_to_feat_c)
        self.register_buffer("_item_to_feat_d", item_to_feat_d)

        self.reset_params()

    def debug_str(self) -> str:
        return f"feat_aug_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        from generative_recommenders.research.modeling.initialization import (
            truncated_normal,
        )

        for name, params in self.named_parameters():
            if "_item_emb" in name or "_feat_" in name and "_proj" not in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            elif "_feat_proj_" in name and "weight" in name:
                torch.nn.init.xavier_uniform_(params)
            elif "_feat_proj_" in name and "bias" in name:
                params.data.fill_(0.0)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        # Main item ID embedding: [..., item_embedding_dim]
        emb = self._item_emb(item_ids)

        # Feature embeddings: look up indices, embed, project, sum
        feat_a_idx = self._item_to_feat_a[item_ids]  # [...]
        feat_b_idx = self._item_to_feat_b[item_ids]
        feat_c_idx = self._item_to_feat_c[item_ids]
        feat_d_idx = self._item_to_feat_d[item_ids]

        emb = emb + self._feat_proj_a(self._feat_a_emb(feat_a_idx))
        emb = emb + self._feat_proj_b(self._feat_b_emb(feat_b_idx))
        emb = emb + self._feat_proj_c(self._feat_c_emb(feat_c_idx))
        emb = emb + self._feat_proj_d(self._feat_d_emb(feat_d_idx))

        return emb

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class DenseFeatureEmbeddingModule(EmbeddingModule):
    """
    Item embedding that augments the main item ID embedding with pre-computed
    dense features (e.g., text embeddings from Qwen, 2048-dim).

    For each item: emb = id_emb[item_id] + proj(pretrained_emb[item_id])
    where proj is a learned Linear(dense_dim, item_embedding_dim).
    The pretrained embeddings are stored as a frozen buffer.
    """

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        pretrained_emb: torch.Tensor,  # [num_items + 1, dense_dim], row 0 = padding
        dense_dim: int,
    ) -> None:
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim

        # Main item ID embedding
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )

        # Dense projection: pretrained_dim -> item_embedding_dim
        self._dense_proj = torch.nn.Linear(dense_dim, item_embedding_dim)

        # Store pretrained embeddings as frozen buffer
        self.register_buffer("_pretrained_emb", pretrained_emb)

        self.reset_params()

    def debug_str(self) -> str:
        return f"dense_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        from generative_recommenders.research.modeling.initialization import (
            truncated_normal,
        )

        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            elif "_dense_proj" in name and "weight" in name:
                torch.nn.init.xavier_uniform_(params)
            elif "_dense_proj" in name and "bias" in name:
                params.data.fill_(0.0)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        # Main item ID embedding
        emb = self._item_emb(item_ids)

        # Dense feature: lookup + project + sum
        dense_feat = self._pretrained_emb[item_ids]  # [..., dense_dim]
        emb = emb + self._dense_proj(dense_feat)  # [..., item_embedding_dim]

        return emb

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
