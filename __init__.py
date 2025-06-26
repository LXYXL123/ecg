from torch.utils.data import DataLoader
from dataset_precessing import ECGDatasetLoader
from modules.Mymamba import MambaForECGClassification
import torch
from torch.nn import BCEWithLogitsLoss
from utils import train, test
from torchinfo import summary

path = 'data/'
sampling_rate = 100
input_dim = 12
num_classes = 5
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# 加载数据
trainDataset = ECGDatasetLoader(path, sampling_rate, type='train', use_cache=True)
valDataset = ECGDatasetLoader(path, sampling_rate, type='val', use_cache=True)
testDataset = ECGDatasetLoader(path, sampling_rate, type='test', use_cache=True)

train_loader = DataLoader(trainDataset, batch_size=32)
val_loader = DataLoader(valDataset, batch_size=32)
test_loader = DataLoader(testDataset, batch_size=32)

# 初始化
model = MambaForECGClassification(
    input_dim=input_dim,
    num_classes=num_classes
)

summary(model, input_size=(32, 1000, 12))


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = BCEWithLogitsLoss()

# 训练
train(model, train_loader, val_loader, optimizer, criterion, num_epochs=1)

# 测试
load_model_path = './save_model/mamba_epoch50.pth'
test(model, test_loader, load_model_path)
