import torch
from torch import nn
from torch.nn import functional as F

from utils.misc import init_linear


class LlamaRMSNorm(nn.Module):
    """ From Llama """
    def __init__(self, d_latent, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_latent))
        self.var_epsilon = eps

    def forward(self, h):
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.var_epsilon)
        return self.weight * h


class FeedForward(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.layer = Expert(d_latent, d_latent)

    def forward(self, h):
        return self.layer(h)


class Expert(nn.Module):
    """ SwishGLU """
    def __init__(self, d_in, d_out):
        super().__init__()
        d_ffn = 4 * int(2 * d_in / 3)

        self.fc_1 = nn.Linear(d_in, d_ffn, bias=False)
        self.fc_2 = nn.Linear(d_in, d_ffn, bias=False)
        self.fc_3 = nn.Linear(d_ffn, d_out, bias=False)
        init_linear(self.fc_1)
        init_linear(self.fc_2)
        init_linear(self.fc_3)

    def forward(self, h):
        h = self.fc_3(F.silu(self.fc_2(h)) * self.fc_1(h))
        return h


class Gate(nn.Module):
    def __init__(self, d_input, n_expert):
        super().__init__()
        self.fc = nn.Linear(d_input, n_expert, bias=False)
        init_linear(self.fc)

    def forward(self, h):
        h = F.softmax(self.fc(h), dim=-1)
        return h
