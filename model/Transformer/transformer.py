import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
data_path = '../../dataset/ETTh1.csv'
data = pd.read_csv(data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 提取需要的特征列
features = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values

# 数据标准化
scaler = MinMaxScaler(feature_range=(-1, 1))
features_normalized = scaler.fit_transform(features)


# 准备数据集
def prepare_data(data, seq_length):
    input_seq, target_seq = [], []
    for i in range(len(data) - seq_length * 2):
        input_seq.append(data[i:i + seq_length])
        target_seq.append(data[i + seq_length:i + seq_length * 2])
    return torch.FloatTensor(input_seq).to(device), torch.FloatTensor(target_seq).to(device)


# 构建Transformer模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 调整形状以适应Transformer的输入要求
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # 恢复原始形状
        x = self.fc(x)
        return x


# 设置超参数
input_size = 7  # 输入特征数
hidden_size = 64  # Transformer隐藏层大小
output_size = 7  # 输出特征数
num_layers = 2  # Transformer层数
num_heads = 8  # Transformer头数
dropout = 0.1
seq_length = 96  # 输入序列长度

# 准备训练数据
input_seq, target_seq = prepare_data(features_normalized, seq_length)

# 划分数据集
train_size = int(0.6 * len(input_seq))
val_size = int(0.2 * len(input_seq))
test_size = len(input_seq) - train_size - val_size

train_dataset = TensorDataset(input_seq[:train_size], target_seq[:train_size])
val_dataset = TensorDataset(input_seq[train_size:train_size + val_size], target_seq[train_size:train_size + val_size])
test_dataset = TensorDataset(input_seq[train_size + val_size:], target_seq[train_size + val_size:])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = TimeSeriesTransformer(input_size, hidden_size, output_size, num_layers, num_heads, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)
# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 在验证集上进行验证
    model.eval()
    with torch.no_grad():
        val_losses = []
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_losses.append(val_loss.item())

        average_val_loss = np.mean(val_losses)
        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {average_val_loss:.4f}')

# 在测试集上进行测试
model.eval()
with torch.no_grad():
    test_losses = []
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        test_losses.append(test_loss.item())

# 打印测试损失
average_test_loss = np.mean(test_losses)
print(f'Test Loss: {average_test_loss:.4f}')
