import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):       
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias, **self.factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **self.factory_kwargs
        )

        self.activation = 'silu'
        self.act = nn.SiLU()
        self.norm = RMSnorm(dim=d_model)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=bias, **self.factory_kwargs)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **self.factory_kwargs)

        # 初始化dt
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # 初始化 dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **self.factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        # softplus逆函数
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # 初始化S4
        A = repeat(
            torch.arange(1, 1 + self.d_state, dtype=torch.float32, device=device),
            pattern='n -> d n',
            d=self.d_inner
        ).contiguous()

        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log.to(device))
        self.A_log._no_weight_decay = True

        # D “skip”
        self.D = nn.Parameter(torch.ones(self.d_inner, **self.factory_kwargs))  # (D, )
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **self.factory_kwargs)


    def forward(self, hidden_states):   # (B, L, d_model)
        print(1)
        B, L, D = hidden_states.shape

        device = self.factory_kwargs['device']

        hidden_states_norm = self.norm(hidden_states)   # (B, L, d_model)

        # 分为主通道x和门控通道z
        xz = self.in_proj(hidden_states_norm)    # (B, L, 2D)
        x, z = torch.chunk(xz, 2, dim=-1)    # (B, L, D)

        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.conv1d(x)[..., : L]    # (B, D, L)
        x = rearrange(x, pattern='b d l -> b l d')  # (B, L, D)
        x = self.act(x)

        # 将x分为3个部分，dt,B,C
        x_dbl = self.x_proj(x)   # [B, L, dt_rank + 2*d_state]
        x_dbl = rearrange(x_dbl, pattern='b l d -> (b l) d')
        dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 时间尺度 Δt 映射 → dt: [B*L, d_inner]
        dt = self.dt_proj.weight @ dt.t() # (D, B*L)
        dt = rearrange(dt, pattern='d (b l) -> b l d', l=L).contiguous().to(device)
        B_param = rearrange(B_param, pattern='(b l) d -> b l d', l=L).contiguous().to(device)
        C_param = rearrange(C_param, pattern='(b l) d -> b l d', l=L).contiguous().to(device)  # (B, L, d_state)

        # 构建SSM输入参数dt,A,B,C,D
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype, device=device))
        A = -torch.exp(self.A_log.float()).to(device)  # (D, d_state)
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A)) # (B, L, D, d_state)
        dB = dt.unsqueeze(-1) * rearrange(B_param, pattern='b l n -> b l 1 n') # (B, L, D, d_state)
        dB = dB.to(device=device)

        # 初始化状态 S = 0，逐步迭代状态更新
        s = torch.zeros(B, self.d_inner, self.d_state, **self.factory_kwargs)   # (B, D, d_state)
        x = x.to(device)
        ys = []

        for t in range(L):
            s = dA[:, t] * s + dB[:, t] * x[:, t].unsqueeze(-1) # update state  (B, D, d_state)
            y = torch.einsum('bdn,bn->bd', s, C_param[:, t])    # (B, D)
            # y = torch.sum(s * C_param[:, t].unsqueeze(-1), dim=-1)  # (B, D)
            y = y + self.D * x[:, t]    # (B, D)
            y = y * self.act(z[:, t])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D)

        out = self.out_proj(y)  # (B, L, d_model)

        out = out + hidden_states   # 残差结构 (N, L, d_model)

        return out


class RMSnorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.scale * x / (norm + self.eps)


class MambaECGClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, d_state=16, d_conv=4, expand=2, num_layers=6, device=None, dtype=None):
        super().__init__()
        self.model_name = 'mamba_simple'
        factory_kwargs = {
            'device': device,
            'dtype': dtype
        }
        self.input_proj = nn.Linear(input_dim, d_model, **factory_kwargs)
        block_config = {
            'd_model': d_model,
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            **factory_kwargs
        }
        self.modelList = nn.ModuleList([Mamba(**block_config) for _ in range(num_layers)])

        self.norm = RMSnorm(dim=d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.output_proj = nn.Linear(d_model, num_classes, **factory_kwargs)
    
    def forward(self, x):   # (B, L, input_dim)
        x = self.input_proj(x)  # (B, L, d_model)
        for block in self.modelList:
            x = block(x)    # (B, L, d_model)

        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)    # (B, d_model)
        out = self.output_proj(x)   # (B, num_classes)
        return out

