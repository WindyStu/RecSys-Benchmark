import torch
import torch.nn as nn

from models.subs.layers import Gate, Expert


class PLE(nn.Module):
    def __init__(self, d_latent):
        super().__init__()

        self.d_latent = d_latent

        self.expert_m = Expert(d_latent * 2, d_latent)
        self.expert_1 = Expert(d_latent * 2, d_latent)
        self.expert_2 = Expert(d_latent * 2, d_latent)

        self.gate_1 = Gate(d_latent * 2, 2)
        self.gate_2 = Gate(d_latent * 2, 2)

    def forward(self, h_1, h_2, mask_seq):
        inputs = torch.cat((h_1, h_2), dim=-1)

        ex_m = self.expert_m(inputs).unsqueeze(-2)
        ex_1 = self.expert_1(inputs).unsqueeze(-2)
        ex_2 = self.expert_2(inputs).unsqueeze(-2)

        g_1 = self.gate_1(inputs).unsqueeze(-1)
        g_2 = self.gate_2(inputs).unsqueeze(-1)

        h_1 = (g_1 * torch.cat((ex_1, ex_m), dim=-2)).sum(-2) * mask_seq
        h_2 = (g_2 * torch.cat((ex_2, ex_m), dim=-2)).sum(-2) * mask_seq

        return h_1, h_2


class CrossDomainAggregation(nn.Module):
    """ Cross-Domain Mixture-of-Experts Aggregation Module """
    def __init__(self, d_latent):
        super().__init__()
        self.d_latent = d_latent
        self.PLE_a = PLE(self.d_latent)
        self.PLE_b = PLE(self.d_latent)

    def forward(self, h_s, h_a, h_b, mask_seq_a, mask_seq_b):
        h_a2s, h_a = self.PLE_a(h_s, h_a, mask_seq_a)
        h_b2s, h_b = self.PLE_b(h_s, h_b, mask_seq_b)
        return h_a, h_b, h_a2s, h_b2s
