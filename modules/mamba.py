# 文件：modules/mamba.py
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int = 1  # ECG 不使用 vocab，这里只是占位
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.scale * x / (norm + self.eps)


class SimpleMambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = RMSNorm(args.d_model)
        self.linear1 = nn.Linear(args.d_model, args.d_inner)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(args.d_inner, args.d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x + residual


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_name = 'mamba'
        self.args = args
        self.layers = nn.ModuleList([SimpleMambaBlock(args) for _ in range(args.n_layer)])
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):  # x: [B, L, d_model]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MambaForECGClassification(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, n_layer=6):
        super().__init__()
        args = ModelArgs(d_model=d_model, n_layer=n_layer, vocab_size=1)
        self.backbone = Mamba(args)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)        # [B, L, d_model]
        x = self.backbone(x)          # [B, L, d_model]
        x = x.transpose(1, 2)         # [B, d_model, L]
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        return self.classifier(x)     # [B, num_classes]
