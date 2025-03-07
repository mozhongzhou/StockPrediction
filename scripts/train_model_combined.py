# train_model_combined.py
# 综合训练 4 只股票（AAPL, NVDA, TSLA, GOOGL）的多模态 Transformer

# 导入必要的库
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 设置数据和模型路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data", "processed", "multimodal")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# 定义多模态 Transformer 模型
class MultimodalTransformer(nn.Module):
    def __init__(self, kline_input_dim=5, report_input_dim=768, hidden_dim=256, num_layers=4, num_heads=2):
        super(MultimodalTransformer, self).__init__()
        # K 线 Transformer，数据多，头数加到 2
        self.kline_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=kline_input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.2),
            num_layers=num_layers
        )
        # 财报全连接层，压缩到 256 维
        self.report_fc = nn.Linear(report_input_dim, hidden_dim)
        self.fc = nn.Linear(kline_input_dim + hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, kline, report):
        kline_out = self.kline_transformer(kline)[:, -1, :]  # 取最后一周，(batch, 5)
        report_out = self.report_fc(report.squeeze(1))  # (batch, 256)
        combined = torch.cat((kline_out, report_out), dim=1)  # (batch, 5 + 256)
        out = self.fc(combined)
        return self.sigmoid(out)

# 加载 4 个股票的数据
stocks = ["AAPL", "NVDA", "TSLA", "GOOGL"]
all_kline, all_report, all_labels = [], [], []
for stock in stocks:
    kline = np.load(os.path.join(data_dir, f"{stock}_kline.npy"))  # (532, 104, 5)
    report = np.load(os.path.join(data_dir, f"{stock}_report.npy"))  # (532, 1, 768)
    labels = np.load(os.path.join(data_dir, f"{stock}_labels.npy"))  # (532,)
    all_kline.append(kline)
    all_report.append(report)
    all_labels.append(labels)
    print(f"Loaded {stock}: kline {kline.shape}, report {report.shape}, labels {labels.shape}")

# 拼接数据
kline = np.concatenate(all_kline, axis=0)  # (2128, 104, 5)
report = np.concatenate(all_report, axis=0)  # (2128, 1, 768)
labels = np.concatenate(all_labels, axis=0)  # (2128,)
print(f"Combined data: kline {kline.shape}, report {report.shape}, labels {labels.shape}")

# 转 PyTorch 张量
kline = torch.FloatTensor(kline)
report = torch.FloatTensor(report)
labels = torch.FloatTensor(labels).unsqueeze(1)  # (2128, 1)

# 分训练和测试集
train_kline, test_kline, train_report, test_report, train_labels, test_labels = train_test_split(
    kline, report, labels, test_size=0.2, random_state=42
)
print(f"Train samples: {len(train_kline)}, Test samples: {len(test_kline)}")

# 初始化模型、损失和优化器
model = MultimodalTransformer()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 训练循环
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_kline, train_report)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 测试
model.eval()
with torch.no_grad():
    test_outputs = model(test_kline, test_report)
    test_preds = (test_outputs > 0.5).float()
    accuracy = (test_preds == test_labels).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")

# 保存模型
torch.save(model.state_dict(), os.path.join(model_dir, "combined_model.pth"))
print("Model saved to combined_model.pth")