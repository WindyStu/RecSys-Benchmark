import torch
from torch import nn
from torch.nn import functional as F

from utils.misc import init_linear


class DomainExtraction(nn.Module):
    """ Domain Extraction Module """
    def __init__(self, d_latent):
        super().__init__()
        self.fc1 = nn.Linear(d_latent, d_latent, bias=True)
        self.fc2 = nn.Linear(d_latent, d_latent, bias=True)
        init_linear(self.fc1)
        init_linear(self.fc1)

    def forward(self, x):
        return self.fc1(F.relu(self.fc1(x.detach())))


class AttentionModuleSingle(nn.Module):
    """
    Attention Module for Single-Domain
    Bug: the $H_e^x$ and $H_e^y$ are XOR, since they are domain-specific sequences
    E.g., if $H_e^x(k)$ ≠ masking pad, $H_e^y(k)$ must be masking pad, and vice versa.
    As a result the attention value here will all be zero, and the weight will be both 0.5 after sigmoid on 0.
    """
    def __init__(self):
        super().__init__()

    def forward(self, h_a, h_b):
        return (h_a + h_b) / 2


class AttentionModuleCross(nn.Module):
    """ Attention Module for Cross-Domain """
    def __init__(self):
        super().__init__()

    def forward(self, h_s, h_c):
        att = torch.sigmoid((h_s * h_c))
        return h_s * att + h_c * (1 - att)
