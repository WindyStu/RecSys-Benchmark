import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout_gnn = args.dropout_gnn
        self.n_gnn = args.n_gnn

        self.bias = nn.Parameter(torch.FloatTensor(args.d_latent))
        stdv = 1. / math.sqrt(self.bias.size(0))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        h_sum = [h]
        for _ in range(self.n_gnn):
            h = F.dropout(h, self.dropout_gnn, training=self.training)
            h = torch.spmm(adj, h) + self.bias
            h_sum.append(h)
        return torch.stack(h_sum, dim=1).mean(dim=1)
