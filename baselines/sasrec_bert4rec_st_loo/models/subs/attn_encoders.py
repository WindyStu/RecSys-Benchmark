import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_latent, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_latent, d_latent, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(d_latent, d_latent, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(F.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = inputs + outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class SelfAttentionEncoderLayer(nn.Module):
    def __init__(self, d_latent, n_head, dropout):
        super().__init__()
        self.d_latent = d_latent
        self.n_head = n_head

        self.norm_attn = nn.LayerNorm(self.d_latent)
        self.mha = nn.MultiheadAttention(self.d_latent, self.n_head, batch_first=True)
        self.norm_ffn = nn.LayerNorm(self.d_latent)
        self.ffn = PointWiseFeedForward(self.d_latent, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, mask_causal):
        h = self.norm_attn(h)
        h = h + self.dropout(self.mha(h, h, h, attn_mask=mask_causal)[0])
        h = h + self.dropout(self.ffn(self.norm_ffn(h)))
        return h


class SelfAttentionEncoder(nn.Module):
    def __init__(self, d_latent, n_attn, n_head, dropout, len_trim):
        super().__init__()
        self.idx_pad = 0
        self.d_latent = d_latent
        self.n_attn = n_attn
        self.n_head = n_head
        self.dropout = dropout
        self.len_trim = len_trim

        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_attn):
            self.layers.append(SelfAttentionEncoderLayer(self.d_latent, self.n_head, self.dropout))

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((self.len_trim, self.len_trim), float('-inf')), diagonal=1))

    def forward(self, h, mask_seq):
        for layer in self.layers:
            h = h * mask_seq
            h = layer(h, self.mask_causal)
        return h
