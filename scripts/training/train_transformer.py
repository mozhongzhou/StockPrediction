# scripts/training/train_transformer.py
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import PyPDF2
from datetime import datetime

# 自定义数据集
class StockDataset(Dataset):
    def __init__(self, ticker, seq_len=30):
        self.df = pd.read_csv(f"data/processed/labeled/{ticker}_labeled.csv", parse_dates=["date"])
        self.seq_len = seq_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.report_dir = "data/raw/reports"
        self.ticker = ticker
    
    def __len__(self):
        return len(self.df) - self.seq_len
    
    def __getitem__(self, idx):
        # K 线序列
        kline = self.df.iloc[idx:idx+self.seq_len][["open", "high", "low", "close", "volume"]].values
        kline = torch.tensor(kline, dtype=torch.float32)
        
        # 10-K 文本
        report_date = self.df.iloc[idx]["10k_date"]
        report_file = os.path.join(self.report_dir, f"{self.ticker}_10-K_{report_date}.pdf")
        if not os.path.exists(report_file):
            raise FileNotFoundError(f"10-K 文件不存在: {report_file}")
        with open(report_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() or "" for page in reader.pages[:10])
        text_inputs = self.tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}  # 移除 batch 维度
        
        # 标签
        label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return kline, text_inputs, label

# Transformer 模型
class StockPredictionTransformer(nn.Module):
    def __init__(self, kline_dim=5, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.kline_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=kline_dim, nhead=1, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=num_layers
        )
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fusion = nn.Linear(kline_dim + 768, hidden_dim)  # 768 是 BERT 输出维度
        self.fc = nn.Linear(hidden_dim, 2)  # 二分类：涨/跌
        self.dropout = nn.Dropout(dropout)

    def forward(self, kline, text_inputs):
        kline_out = self.kline_encoder(kline)[:, -1, :]  # 取最后一个时间步
        text_out = self.text_encoder(**text_inputs).pooler_output  # (batch, 768)
        combined = torch.cat([kline_out, text_out], dim=-1)
        combined = self.fusion(combined)
        combined = self.dropout(torch.relu(combined))
        logits = self.fc(combined)
        return logits

# 训练函数
def train_model(ticker, seq_len=30, batch_size=16, num_epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据加载
    dataset = StockDataset(ticker, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 模型、损失函数和优化器
    model = StockPredictionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for kline, text_inputs, labels in dataloader:
            kline = kline.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(kline, text_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    model_path = os.path.join("models", f"{ticker}_transformer_stock.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"模型保存至: {model_path}")

if __name__ == "__main__":
    ticker = "AAPL"  # 可改为参数化
    train_model(ticker)