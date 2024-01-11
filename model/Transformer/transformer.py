import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split

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
scaler = MinMaxScaler(feature_range=(0, 1))
features_normalized = scaler.fit_transform(features)

# 长时预测还是短时预测
predict_type = "short"

factor = 1 if predict_type == "short" else 3.5


# 准备数据集
def prepare_data(data, seq_length):
    input_seq, target_seq = [], []
    for i in range(len(data) - int(seq_length * (factor + 1))):
        input_seq.append(data[i:i + seq_length])
        target_seq.append(data[i + seq_length:i + int(seq_length * (factor + 1))])
    return torch.FloatTensor(input_seq).to(device), torch.FloatTensor(target_seq).to(device)


# 构建Transformer模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc = nn.Linear(hidden_size, int(16 * factor))
        self.linear = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 调整形状以适应Transformer的输入要求
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # 恢复原始形状
        x = self.fc(x)
        x = self.linear(x.view(x.size(0), int(factor * seq_length), 16))
        return x


# 设置超参数
input_size = 7  # 输入特征数
hidden_size = 64  # Transformer隐藏层大小
output_size = 7  # 输出特征数
num_layers = 2  # Transformer层数
num_heads = 8  # Transformer头数
dropout = 0.1
seq_length = 96  # 输入序列长度

train_input, train_target = prepare_data(features_normalized[:8640], seq_length)

valid_input, valid_target = prepare_data(features_normalized[8640: 8640 + 2976], seq_length)

test_input, test_target = prepare_data(features_normalized[8640 + 2976:], seq_length)

train_dataset = TensorDataset(train_input, train_target)
val_dataset = TensorDataset(valid_input, valid_target)
test_dataset = TensorDataset(test_input, test_target)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型、损失函数和优化器，并将它们移动到 GPU
model = TimeSeriesTransformer(input_size, hidden_size, output_size, num_layers, num_heads, dropout).to(device)
MSE = nn.MSELoss()
MAE = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 保存模型的路径
output_path = './output/'

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
            torch.save(model, f"{output_path}transformer_{seq_length}h_best.pt")
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

# 寻找loss最小的一组数据 作为绘图数据
min_loss = 500
draw_inputs = None
draw_targets = None
draw_prediction = None


print(best_epoch)
# 测试模型
model = torch.load(f"{output_path}transformer_{seq_length}h_best.pt")
model.eval()
with torch.no_grad():
    test_losses = []
    MAE_losses = []
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        test_loss = MSE(test_outputs[:, :, -1:], test_targets[:, :, -1:])
        draw_loss = MAE(test_outputs[:, :, -1:], test_targets[:, :, -1:])
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

# 创建时间轴
time_axis = np.arange(0, int(seq_length * (factor + 1)))

# 分别绘制每个特征的图表
for i in range(6, 7):
    plt.figure(figsize=(12, 6))
    merged_actual = np.concatenate([input_data[-seq_length:, i], actual_outputs[-int(factor * seq_length):, i]])
    plt.plot(time_axis[-int((factor + 1) * seq_length):], merged_actual, label=f'Feature: {data_head[i]} (Actual 0-{seq_length}h)')
    plt.plot(time_axis[-int(factor * seq_length):], actual_outputs[-int(factor * seq_length):, i],
             label=f'Feature: {data_head[i]} (Actual {seq_length}h-{int((factor + 1) * seq_length)}h)')
    plt.plot(time_axis[-int(factor * seq_length):], predicted_outputs[-int(factor * seq_length):, i], label=f'Feature: {data_head[i]} (Predicted)',
             linestyle='dashed')

    plt.title(f'Feature: {data_head[i]} - Time Series Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel(f'Feature: {data_head[i]} Values')
    plt.legend()
    plt.show()
