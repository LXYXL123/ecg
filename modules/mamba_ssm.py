import torch
import torch.nn as nn
from transformers import MambaModel

class Mamba_SSM(nn.Module):
    def __init__(self, pretrained_name: str, num_classes: int):
        super().__init__()
        self.backbone = MambaModel.from_pretrained(pretrained_name)
        self.input_proj = nn.Linear(12, self.backbone.config.hidden_size)  # 必须是2048
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):  # x: [B, L, 12]
        x = self.input_proj(x)  # [B, L, 2048]
        output = self.backbone(inputs_embeds=x)
        x = output.last_hidden_state  # [B, L, 2048]
        x = x.transpose(1, 2)         # [B, 2048, L]
        x = self.pool(x).squeeze(-1)  # [B, 2048]
        return self.classifier(x)     # [B, num_classes]
