import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score

from dataset_precessing import ECGDatasetLoader
from modules.mamba_ssm import Mamba_SSM

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')
# 参数
path = "./data/"
sampling_rate = 100
batch_size = 16
num_epochs = 10
pretrained_name = "state-spaces/mamba-130m"
num_classes = 5  # 与数据集中多标签维度保持一致

# 加载数据
train_loader = DataLoader(ECGDatasetLoader(path, sampling_rate, 'train'), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ECGDatasetLoader(path, sampling_rate, 'val'), batch_size=batch_size)

# 初始化模型
model = Mamba_SSM(pretrained_name, num_classes=num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    train_loop = tqdm(train_loader, desc=f"[Train Epoch {epoch}]", leave=False)
    for signals, labels in train_loop:
        signals, labels = signals.to(device), labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loop.set_postfix(loss=loss.item())

    # 验证
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"[Val Epoch {epoch}]", leave=False)
        for signals, labels in val_loop:
            signals, labels = signals.to(device), labels.to(device)

            logits = model(signals)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            correct += (preds == labels).float().sum().item()
            total += labels.numel()
            val_loop.set_postfix(acc=correct/total)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = correct / total

    print(f"Epoch {epoch} — F1: {f1:.4f} | Binary Accuracy: {acc:.4f}")
