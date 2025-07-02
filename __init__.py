# from torch.utils.data import DataLoader
# from dataset_precessing import ECGDatasetLoader
# from modules.Mymamba import MambaForECGClassification
# import torch
# from torch.nn import BCEWithLogitsLoss
# from utils import train, test
# from torchinfo import summary
# from modules.mamba_simple import MambaECGClassifier

# path = 'data/'
# sampling_rate = 100
# input_dim = 12
# num_classes = 5

# if torch.backends.mps.is_available():
#     device = torch.device('mps')
# elif torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')


# # 加载数据
# trainDataset = ECGDatasetLoader(path, sampling_rate, type='train', use_cache=True)
# valDataset = ECGDatasetLoader(path, sampling_rate, type='val', use_cache=True)
# testDataset = ECGDatasetLoader(path, sampling_rate, type='test', use_cache=True)

# train_loader = DataLoader(trainDataset, batch_size=16)
# val_loader = DataLoader(valDataset, batch_size=16)
# test_loader = DataLoader(testDataset, batch_size=16)

# # 初始化
# model1 = MambaForECGClassification(
#     input_dim=input_dim,
#     num_classes=num_classes
# )

# model2 = MambaECGClassifier(
#     input_dim=input_dim,
#     num_classes=num_classes,
#     device=device,
#     dtype=torch.float32
# )

# summary(model1, input_size=(32, 1000, 12))
# # summary(model2, input_size=(32, 1000, 12))



# optimizer_m1 = torch.optim.Adam(model1.parameters(), lr=1e-4)
# optimizer_m2 = torch.optim.Adam(model2.parameters(), lr=1e-4)

# criterion = BCEWithLogitsLoss()

# # # 训练
# # print(f'正在训练模型{model1.model_name}')
# # train(model1, train_loader, val_loader, optimizer_m1, criterion, num_epochs=1)
# # print(f'模型{model1.model_name}训练完毕\n\n\n')

# print(f'正在训练模型{model2.model_name}')
# train(model2, train_loader, val_loader, optimizer_m2, criterion, num_epochs=1)
# print(f'模型{model2.model_name}训练完毕\n\n\n')


# # 测试
# # load_model_path = './save_model/mamba_epoch50.pth'
# # test(model, test_loader, load_model_path)


import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from modules.mamba_simple import MambaECGClassifier
from utils import train
# 模拟数据配置
path = 'data/'
sampling_rate = 100
input_dim = 12
num_classes = 5

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 随机生成信号（模拟 ECG）
dummy_signals = torch.randn(32, 1000, 12)
# 随机生成多标签 [0, 1] 标签，float 用于 BCELoss
dummy_labels = torch.randint(0, 2, (32, num_classes)).float()

# 切分训练/验证集
train_dataset = TensorDataset(dummy_signals[:48], dummy_labels[:48])
val_dataset = TensorDataset(dummy_signals[48:], dummy_labels[48:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)




model = MambaECGClassifier(
    input_dim=input_dim,
    num_classes=num_classes,
    device=device,
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
print(f'正在训练模型{model.model_name}')
train(model, train_loader, val_loader, optimizer, criterion, num_epochs=1)
print(f'模型{model.model_name}训练完毕\n\n\n')