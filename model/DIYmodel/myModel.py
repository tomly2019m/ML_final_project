import os
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
import torch.nn.functional as F

# 0 1024

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# 读取数据
train_path = '../../dataset/train_set.csv'

valid_path = '../../dataset/validation_set.csv'

test_path = '../../dataset/test_set.csv'

train_data = pd.read_csv(train_path)

valid_data = pd.read_csv(valid_path)

test_data = pd.read_csv(test_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据表头
data_head = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# 提取需要的特征列
train_features = train_data[data_head].values
valid_features = valid_data[data_head].values
test_features = test_data[data_head].values

features = np.concatenate((train_features, valid_features, test_features), axis=0)
# 数据标准化
scaler = MinMaxScaler(feature_range=(-1, 1))
features_normalized = scaler.fit_transform(features)

# 长时预测还是短时预测
predict_type = "long"

factor = 1 if predict_type == "short" else 3.5


# 准备数据集
def prepare_data(data, seq_length):
    input_seq, target_seq = [], []
    for i in range(len(data) - int(seq_length * (factor + 1))):
        input_seq.append(data[i:i + seq_length])
        target_seq.append(data[i + seq_length:i + int(seq_length * (factor + 1))])
    return torch.FloatTensor(input_seq).to(device), torch.FloatTensor(target_seq).to(device)


class ftat_BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ftat_BiLSTM, self).__init__()

        self.hidden_dim = hidden_size

        self.conv = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)

        self.feature_v = nn.Linear(input_size, input_size)
        self.feature_k = nn.Linear(input_size, input_size)
        self.feature_q = nn.Linear(input_size, input_size)

        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=9, padding=4)

        self.temporal_v = nn.Linear(hidden_size, hidden_size)
        self.temporal_k = nn.Linear(hidden_size, hidden_size)
        self.temporal_q = nn.Linear(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size, int(16 * factor))
        self.fc = nn.Linear(16, output_size)

        self.dropout = nn.Dropout(0.1)

    def feature_attn(self, x):
        # x: B, T, F

        # single head
        value = self.feature_v(x)
        key = self.feature_k(x)
        query = self.feature_q(x)

        attention = torch.matmul(query.transpose(1, 2), key)  # B, F, F
        attention = attention / (input_size ** 0.5)
        attention = F.softmax(attention, dim=-1)
        attn_output = torch.matmul(value, attention)  # B, T, F
        return F.tanh(attn_output + x)

    def temporal_attn(self, lstm_output):
        # single head
        value = self.temporal_v(lstm_output)
        key = self.temporal_k(lstm_output)
        query = self.temporal_q(lstm_output)

        attention = torch.matmul(query, key.transpose(1, 2))  # B, T, T
        attention = attention / (hidden_size ** 0.5)
        attention = F.softmax(attention, dim=-1)
        attn_output = torch.matmul(attention, value)  # B, T, C

        return attn_output + lstm_output

    def forward(self, x):
        # 以p的概率引入高斯噪声
        noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
        noise = torch.FloatTensor(noise).to(device)
        p = 0.1
        mask = torch.rand(*x.shape, device='cuda') < p
        x = x + mask * noise / 5
        x = torch.clamp(x, -1, 1)

        # x: B, T, F
        conv_out = x.permute(0, 2, 1)
        conv_out = self.conv(conv_out)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.dropout(conv_out)
        # 特征维度上进行attention
        # 只使用feature_attn时，在训练集上都表现出较慢的收敛速度
        # feature_attn_output = x
        feature_attn_output = self.feature_attn(x + conv_out)  # B, T, F

        lstm_output, _ = self.bilstm(feature_attn_output)  # B, T, 2 * C
        batch_size, seq_len, _ = lstm_output.shape
        lstm_output = lstm_output.contiguous().view(batch_size, seq_len, 2, -1)  # B, T, 2, C
        lstm_output = torch.mean(lstm_output, dim=2)  # 关键步骤

        # 时间维度上进行attention
        # 感觉还是有必要留着，不然会有明显的滞后现象
        conv2_output = lstm_output.permute(0, 2, 1)
        conv2_output = self.conv2(conv2_output)
        conv2_output = conv2_output.permute(0, 2, 1)
        conv2_output = self.dropout(conv2_output)

        temporal_attn_output = self.temporal_attn(conv2_output)  # B, T, C
        # temporal_attn_output = lstm_output  # B, T, C
        # temporal_attn_output = self.dropout(temporal_attn_output)
        out = self.linear(temporal_attn_output)  # B, T_out
        out = self.fc(out.reshape(x.size(0), int(factor * seq_length), 16))
        return out


# 设置超参数
input_size = 7  # 输入特征数
hidden_size = 512  # 隐藏层大小
output_size = 1  # 输出特征数
dropout = 0.1
seq_length = 96  # 输入序列长度

train_input, train_target = prepare_data(features_normalized[:8640], seq_length)

valid_input, valid_target = prepare_data(features_normalized[8640: 8640 + 2976], seq_length)

test_input, test_target = prepare_data(features_normalized[8640 + 2976:], seq_length)

train_dataset = TensorDataset(train_input, train_target)
val_dataset = TensorDataset(valid_input, valid_target)
test_dataset = TensorDataset(test_input, test_target)


def add_gaussian_noise(sample, mean=0, std=0.02):
    noise = torch.randn_like(sample) * std + mean
    return sample + noise


augmented_data = []
select_index = set()
for i in range(int(len(train_dataset) * 0.2)):
    while True:
        rand_index = np.random.randint(0, len(train_dataset))
        if not rand_index in select_index:
            select_index.add(rand_index)
            break
    input, target = train_dataset[rand_index]
    input = add_gaussian_noise(input)
    target = add_gaussian_noise(target)
    augmented_data.append((input, target))

train_dataset = ConcatDataset([train_dataset, augmented_data])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型、损失函数和优化器，并将它们移动到 GPU
model = ftat_BiLSTM(input_size, output_size, hidden_size).to(device)
MSE = nn.MSELoss()
MAE = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 保存模型的路径
output_path = f'./output/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}/'
os.makedirs(output_path)

# 训练模型
epochs = 500 if predict_type == 'short' else 1000
best_epoch = 0
best_val_loss = 1000
train_MSE_losses = []
val_avg_MSE_losses = []
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        MSE_loss = MSE(outputs[:, :, -1:], targets[:, :, -1:])
        MAE_loss = MAE(outputs[:, :, -1:], targets[:, :, -1:])
        MSE_loss.backward()
        optimizer.step()

    # 在验证集上进行验证
    model.eval()
    with torch.no_grad():
        val_MSE_losses = []
        val_MAE_losses = []
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_MSE_loss = MSE(val_outputs[:, :, -1:], val_targets[:, :, -1:])
            val_MAE_loss = MAE(val_outputs[:, :, -1:], val_targets[:, :, -1:])
            val_MSE_losses.append(val_MSE_loss.item())
            val_MAE_losses.append(val_MAE_loss.item())

        average_val_MSE_loss = np.mean(val_MSE_losses)
        average_val_MAE_loss = np.mean(val_MAE_losses)
        train_MSE_losses.append(MSE_loss.item())
        val_avg_MSE_losses.append(average_val_MSE_loss)
        print(
            f'Epoch [{epoch + 1}/{epochs}], MSE Loss: {MSE_loss.item():.6f}, Val MSE Loss: {average_val_MSE_loss:.6f}, MAE Loss: {MAE_loss.item():.6f}, Val MAE Loss: {average_val_MAE_loss:.6f}')
        if average_val_MSE_loss < best_val_loss and epoch >= (400 if predict_type == 'short' else 500):
            torch.save(model, f"{output_path}DIY_model_{seq_length}h_best.pt")
            best_val_loss = average_val_MSE_loss
            best_epoch = epoch

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_MSE_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_avg_MSE_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(output_path + 'loss.png')

# 寻找loss最小的一组数据 作为绘图数据
min_loss = 10000
draw_inputs = None
draw_targets = None
draw_prediction = None

print(best_epoch)

# 测试模型
model = torch.load(f"{output_path}DIY_model_{seq_length}h_best.pt")
model.eval()
with torch.no_grad():
    test_losses = []
    MAE_losses = []
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        test_loss = MSE(test_outputs[:, :, -1:], test_targets[:, :, -1:])
        draw_loss = MSE(test_outputs[:, :, -1:], test_targets[:, :, -1:])
        MAE_loss = MAE(test_outputs[:, :, -1:], test_targets[:, :, -1:])
        test_losses.append(test_loss.item())
        MAE_losses.append(MAE_loss.item())
        if draw_loss.item() < min_loss:
            min_loss = draw_loss.item()
            draw_inputs = test_inputs
            draw_targets = test_targets
            draw_prediction = test_outputs.cpu().numpy()

# 计算均方根误差
average_test_loss = np.mean(test_losses)
average_MAE_loss = np.mean(MAE_losses)
print(f'Mean Squared Error on Test Data: {average_test_loss:.4f}')
print(f'Mean Absolute Error on Test Data: {average_MAE_loss:.4f}')
std_test_loss = np.std(test_losses)
std_MAE_loss = np.std(MAE_losses)
print(f'standard Deviation of Mean Squared Error on Test Data:{std_test_loss:.4f}' )
print(f'standard Deviation of Mean Absolute Error on Test Data:{std_MAE_loss: .4f}')


# 使用开头数据作为绘图输入
iterator = iter(test_loader)
draw_inputs, draw_targets = next(iterator)

with torch.no_grad():
    draw_prediction = model(draw_inputs).cpu().numpy()
# 获取最后一组预测结果
predicted_outputs = draw_prediction

padded_outputs = np.zeros((predicted_outputs.shape[0], predicted_outputs.shape[1], input_size))
padded_outputs[:, :, -1] = predicted_outputs[:, :, -1]
predicted_outputs = padded_outputs

# 获取最后一组预测结果
predicted_outputs = draw_prediction

padded_outputs = np.zeros((predicted_outputs.shape[0], predicted_outputs.shape[1], input_size))
padded_outputs[:, :, -1] = predicted_outputs[:, :, -1]
predicted_outputs = padded_outputs

# 反标准化预测结果
predicted_outputs = scaler.inverse_transform(predicted_outputs.reshape(-1, input_size))

# 反标准化测试集目标数据
actual_outputs = draw_targets.cpu().numpy()
actual_outputs = scaler.inverse_transform(actual_outputs.reshape(-1, input_size))

# 反标准化测试集输入数据（前96小时已知数据）
input_data = draw_inputs.cpu().numpy().reshape(-1, input_size)
input_data = scaler.inverse_transform(input_data)

# 创建时间轴
time_axis = np.arange(0, int(seq_length * (factor + 1)))

# 分别绘制每个特征的图表
for i in range(6, 7):
    plt.figure(figsize=(12, 6))
    merged_actual = np.concatenate([input_data[-seq_length:, i], actual_outputs[-int(factor * seq_length):, i]])
    plt.plot(time_axis[-int((factor + 1) * seq_length):], merged_actual,
             label=f'Feature: {data_head[i]} (Actual 0-{seq_length}h)')
    plt.plot(time_axis[-int(factor * seq_length):], actual_outputs[-int(factor * seq_length):, i],
             label=f'Feature: {data_head[i]} (Actual {seq_length}h-{int((factor + 1) * seq_length)}h)')
    plt.plot(time_axis[-int(factor * seq_length):], predicted_outputs[-int(factor * seq_length):, i],
             label=f'Feature: {data_head[i]} (Predicted)',
             linestyle='dashed')

    plt.title(f'Feature: {data_head[i]} - Time Series Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel(f'Feature: {data_head[i]} Values')
    plt.legend()
    plt.show()
    plt.savefig(output_path + 'ot.png')

with open(output_path + "parameter.txt", 'w') as f:
    f.write(f"hidden_size={hidden_size}\n")
    f.write(f'batch_size=64\n')
    f.write(f'lr=0.001\n')
    f.write(f'noise_std=0.02\n')
    f.write(f'percent=0.2\n')
    f.write(f'MSE={average_test_loss}\n')
    f.write(f'MAE={average_MAE_loss}\n')
    f.write(f'DR={dropout}\n')
    f.write(f'mseSTD={std_test_loss}\n')
    f.write(f'MAEStd={std_MAE_loss}\n')
    f.write(f'seed={seed}')