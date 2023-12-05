import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器，并将它们移动到 GPU
model = LSTM(input_size, hidden_size, output_size).to(device)
MSE = nn.MSELoss()
MAE = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 1000
train_MSE_losses = []
val_avg_MSE_losses = []
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        MSE_loss = MSE(outputs, targets)
        MAE_loss = MAE(outputs, targets)
        MSE_loss.backward()
        optimizer.step()

    # 在验证集上进行验证
    model.eval()
    with torch.no_grad():
        val_MSE_losses = []
        val_MAE_losses = []
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_MSE_loss = MSE(val_outputs, val_targets)
            val_MAE_loss = MAE(val_outputs, val_targets)
            val_MSE_losses.append(val_MSE_loss.item())
            val_MAE_losses.append(val_MAE_loss.item())

        average_val_MSE_loss = np.mean(val_MSE_losses)
        average_val_MAE_loss = np.mean(val_MAE_losses)
        train_MSE_losses.append(MSE_loss.item())
        val_avg_MSE_losses.append(average_val_MSE_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], MSE Loss: {MSE_loss.item():.4f}, Val MSE Loss: {average_val_MSE_loss:.4f}, MAE Loss: {MAE_loss.item():.4f}, Val MAE Loss: {average_val_MAE_loss:.4f}')

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_MSE_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_avg_MSE_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 寻找loss最小的一组数据 作为绘图数据
min_loss = 10000
draw_inputs = None
draw_targets = None
draw_prediction = None

# 测试模型
model.eval()
with torch.no_grad():
    test_losses = []
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        test_loss = MSE(test_outputs, test_targets)
        test_losses.append(test_loss.item())
        if test_loss.item() < min_loss:
            min_loss = test_loss.item()
            draw_inputs = test_inputs
            draw_targets = test_targets
            draw_prediction = test_outputs.cpu().numpy()

# 计算均方根误差
average_test_loss = np.mean(test_losses)
print(f'Mean Squared Error on Test Data: {average_test_loss:.4f}')


# 获取最后一组预测结果
predicted_outputs = draw_prediction

# 反标准化预测结果
predicted_outputs = scaler.inverse_transform(predicted_outputs.reshape(-1, output_size))

# 反标准化测试集目标数据
actual_outputs = draw_targets.cpu().numpy()
actual_outputs = scaler.inverse_transform(actual_outputs.reshape(-1, output_size))

# 反标准化测试集输入数据（前96小时已知数据）
input_data = draw_inputs.cpu().numpy().reshape(-1, input_size)
input_data = scaler.inverse_transform(input_data)

# 创建时间轴，只显示最后192小时的数据
time_axis = np.arange(0, 192)


# 分别绘制每个特征的图表
for i in range(output_size):
    plt.figure(figsize=(12, 6))
    merged_actual = np.concatenate([input_data[-96:, i], actual_outputs[-96:, i]])
    plt.plot(time_axis[-192:], merged_actual, label=f'Feature {i+1} (Actual 0-96h)')
    plt.plot(time_axis[-96:], actual_outputs[-96:, i], label=f'Feature {i+1} (Actual 96h-192h)')
    plt.plot(time_axis[-96:], predicted_outputs[-96:, i], label=f'Feature {i+1} (Predicted)', linestyle='dashed')

    plt.title(f'Feature {i+1} - Time Series Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel(f'Feature {i+1} Values')
    plt.legend()
    plt.show()
