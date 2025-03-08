# 导入必要的库
import os  # 处理文件路径，加载数据和保存模型
import numpy as np  # 处理数组，加载 K 线、财报、标签
import torch  # PyTorch 框架，建模和训练用
import torch.nn as nn  # 神经网络模块，比如 Transformer 和全连接层
import torch.optim as optim  # 优化器，调整模型参数
from sklearn.model_selection import train_test_split  # 分训练和测试集

# 设置数据路径
# 用相对路径，保持代码灵活性
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))  # 项目根目录
data_dir = os.path.join(base_dir, "data", "processed", "multimodal")  # 对齐数据和标签的目录
model_dir = os.path.join(base_dir, "models")  # 模型保存目录
os.makedirs(model_dir, exist_ok=True)  # 如果 models 目录不存在就创建

# 定义多模态 Transformer 模型
class MultimodalTransformer(nn.Module):
    def __init__(self, kline_input_dim=5, report_input_dim=768, hidden_dim=128, num_layers=2, num_heads=1):
        super(MultimodalTransformer, self).__init__()
        # K 线 Transformer 编码器
        # 处理 (104, 5) 的时间序列，nhead=1 因为特征少，dim_feedforward 是中间层大小
        self.kline_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=kline_input_dim, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers  # 两层 Transformer，学时间规律
        )
        # 财报全连接层
        # 把 768 维财报向量压缩到 hidden_dim（128），方便融合
        self.report_fc = nn.Linear(report_input_dim, hidden_dim)
        # 融合后的分类器
        # K 线输出 (5) + 财报输出 (128) = 133 维，压缩到 1 维概率
        self.fc = nn.Linear(kline_input_dim + hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()  # 输出 0-1 概率，表示涨的信心

    def forward(self, kline, report):
        # kline: (batch, 104, 5), report: (batch, 1, 768)
        kline_out = self.kline_transformer(kline)[:, -1, :]  # 取最后一周输出，(batch, 5)
        report_out = self.report_fc(report.squeeze(1))  # 去掉 1 维，(batch, 128)
        combined = torch.cat((kline_out, report_out), dim=1)  # 拼接成 (batch, 5 + 128)
        out = self.fc(combined)  # (batch, 1)
        return self.sigmoid(out)  # 概率值 (batch, 1)

# 加载 NVDA 数据
kline = np.load(os.path.join(data_dir, "NVDA_kline.npy"))  # (532, 104, 5)
report = np.load(os.path.join(data_dir, "NVDA_report.npy"))  # (532, 1, 768)
labels = np.load(os.path.join(data_dir, "NVDA_labels.npy"))  # (532,)

# 转成 PyTorch 张量
kline = torch.FloatTensor(kline)  # K 线转浮点张量
report = torch.FloatTensor(report)  # 财报转浮点张量
labels = torch.FloatTensor(labels).unsqueeze(1)  # 标签加一维，(532, 1)，方便算损失

# 分训练集和测试集
# 80% 训练，20% 测试，random_state 固定随机种子
train_kline, test_kline, train_report, test_report, train_labels, test_labels = train_test_split(
    kline, report, labels, test_size=0.2, random_state=42
)
print(f"Train samples: {len(train_kline)}, Test samples: {len(test_kline)}")  # 比如 425 和 107

# 初始化模型、损失函数和优化器
model = MultimodalTransformer()  # 创建模型实例
criterion = nn.BCELoss()  # 二分类交叉熵损失，适合 0/1 预测
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器，学习率 0.001

# 训练循环
epochs = 50  # 跑 50 轮，让模型学透
for epoch in range(epochs):
    model.train()  # 训练模式
    optimizer.zero_grad()  # 清空梯度
    outputs = model(train_kline, train_report)  # 前向传播，(425, 1)
    loss = criterion(outputs, train_labels)  # 计算损失
    loss.backward()  # 反向传播，算梯度
    optimizer.step()  # 更新参数
    if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 测试
model.eval()  # 测试模式，不更新参数
with torch.no_grad():  # 不算梯度，节省内存
    test_outputs = model(test_kline, test_report)  # 测试集预测，(107, 1)
    test_preds = (test_outputs > 0.5).float()  # 概率 > 0.5 算涨 (1)，否则跌 (0)
    accuracy = (test_preds == test_labels).float().mean()  # 准确率
    print(f"Test Accuracy: {accuracy.item():.4f}")

# 保存模型
torch.save(model.state_dict(), os.path.join(model_dir, "nvda_model.pth"))  # 存权重
print("Model saved to nvda_model.pth")