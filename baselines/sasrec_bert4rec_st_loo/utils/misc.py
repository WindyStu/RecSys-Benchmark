import torch.nn as nn


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
