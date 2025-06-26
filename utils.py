from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
import os


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs,
          model_save_path: str = None):

    # 训练循环
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for signals, labels in progress_bar:
            signals = signals.to(device)
            labels = labels.to(device).float()

            logits = model(signals)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Binary accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == labels).float().sum()
            total = torch.numel(labels)

            train_loss += loss.item()
            train_correct += correct.item()
            train_total += total

            progress_bar.set_postfix(loss=loss.item(), binary_acc = train_correct / train_total)

        train_binary_acc = train_correct / train_total

        # ---------- Validate ----------
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        with torch.no_grad():
            for signals, labels in val_loop:
                signals = signals.to(device)
                labels = labels.to(device).float()

                logits = model(signals)
                preds = (torch.sigmoid(logits) > 0.5).float()

                correct = (preds == labels).float().sum()
                total = torch.numel(labels)

                val_correct += correct.item()
                val_total += total

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                val_loop.set_postfix(binary_acc=val_correct / val_total)

        val_binary_acc = val_correct / val_total
        f1 = f1_score(torch.cat(all_labels), torch.cat(all_preds), average='macro')

        print(f"Epoch {epoch} Summary: "
            f"Train Acc = {train_binary_acc:.4f} | "
            f"Val Acc = {val_binary_acc:.4f} | "
            f"F1 = {f1:.4f}")

    if model_save_path is None: # 默认保存地址
        model_save_path = './save_model'

    if not os.path.exists(os.path.dirname(model_save_path)):    # 文件路径不存咋，创建
        os.makedirs(os.path.dirname(model_save_path))
    
    if not os.path.isdir(model_save_path):  # 确保保存地址为目录
        model_save_path = os.path.dirname(model_save_path)

    torch.save(model.state_dict(), os.path.join(model_save_path, f'{model.model_name}_epochs{num_epochs}.pth'))
    print(f'模型训练完成，保存在{model_save_path}')
    # # 加载
    # model.load_state_dict(torch.load("mamba_epoch10.pth"))

def test(model, test_loader, load_model_path):
    model.load_state_dict(torch.load(load_model_path))
    model.to(device)
    model.eval()

    test_correct = 0
    test_total = 0
    all_preds, all_labels = [], []

    test_loop = tqdm(test_loader, desc='[test model]', leave=False)
    with torch.no_grad():
        for signals, labels in test_loop:
            signals = signals.to(device)
            labels = labels.to(device).float()

            logits = model(signals)
            preds = (torch.sigmoid(logits) > 0.5).float()

            correct = (preds == labels).float().sum()
            total = torch.numel(labels)

            test_correct += correct.item()
            test_total += total

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')

            test_loop.set_postfix(binary_acc=correct/total, f1=f1)
        test_f1 = f1_score(torch.cat(all_labels), torch.cat(all_preds), average='macro') 
        print(f'结果为：test_binary_acc={test_correct/test_total:.4f} | '
              f'f1={test_f1:.4f}')


