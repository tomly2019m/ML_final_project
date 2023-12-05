import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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


# 定义模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out)
        return predictions


# 参数设置
input_size = 7  # 输入特征数
hidden_size = 64  # 隐藏层大小
output_size = 7  # 输出特征数
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器，并将它们移动到 GPU
model = LSTM(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
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
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_losses.append(val_loss.item())

        average_val_loss = np.mean(val_losses)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {average_val_loss:.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_losses = []
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        test_losses.append(test_loss.item())

# 计算均方根误差
average_test_loss = np.mean(test_losses)
print(f'Mean Squared Error on Test Data: {average_test_loss:.4f}')
