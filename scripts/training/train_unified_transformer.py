# scripts/training/train_unified_transformer.py
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 设置路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
labeled_dir = os.path.join(base_dir, "data", "processed", "labeled")
report_dir = os.path.join(base_dir, "data", "raw", "reports")
model_dir = os.path.join(base_dir, "models", "unified")
os.makedirs(model_dir, exist_ok=True)

# 数据集
class UnifiedStockDataset(Dataset):
    def __init__(self, stocks, kline_type="weekly", seq_len=30, max_text_len=512):
        dfs = []
        for stock in stocks:
            file = os.path.join(labeled_dir, f"{stock}_{kline_type}_labeled.csv")
            if os.path.exists(file):
                df = pd.read_csv(file, parse_dates=["Date"])
                df["ticker"] = stock  # 添加 ticker 列以区分股票
                dfs.append(df)
            else:
                logger.warning(f"No {kline_type} labeled file for {stock}")
        self.df = pd.concat(dfs, ignore_index=True)
        self.seq_len = seq_len
        self.max_text_len = max_text_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def __len__(self):
        return len(self.df) - self.seq_len
    
    def __getitem__(self, idx):
        kline = self.df.iloc[idx:idx+self.seq_len][["Open", "High", "Low", "Close", "Volume"]].values
        kline = torch.tensor(kline, dtype=torch.float32)
        
        ticker = self.df.iloc[idx]["ticker"]
        report_date = self.df.iloc[idx]["10k_date"]
        report_file = os.path.join(report_dir, f"{ticker}_10-K_{report_date}.txt")
        if not os.path.exists(report_file):
            logger.warning(f"10-K file not found: {report_file}, using empty text")
            text = ""
        else:
            with open(report_file, "r", encoding="utf-8") as f:
                text = f.read()[:100000]
        
        text_inputs = self.tokenizer(text, max_length=self.max_text_len, truncation=True, padding="max_length", return_tensors="pt")
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return kline, text_inputs, label

# Transformer 模型
class MultiModalTransformer(nn.Module):
    def __init__(self, kline_dim=5, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.kline_proj = nn.Linear(kline_dim, hidden_dim)
        self.kline_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, dropout=dropout),
            num_layers=num_layers
        )
        self.text_embedding = nn.Embedding(BertTokenizer.from_pretrained("bert-base-uncased").vocab_size, hidden_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, dropout=dropout),
            num_layers=num_layers
        )
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, kline, text_inputs):
        kline = self.kline_proj(kline)
        kline_out = self.kline_encoder(kline)[:, -1, :]
        text_ids = text_inputs["input_ids"]
        text_embed = self.text_embedding(text_ids)
        text_out = self.text_encoder(text_embed)[:, 0, :]
        combined = torch.cat([kline_out, text_out], dim=-1)
        combined = self.fusion(combined)
        combined = self.dropout(torch.relu(combined))
        logits = self.fc(combined)
        return logits

# 训练函数
def train_model(stocks, kline_type="weekly", seq_len=30, batch_size=16, num_epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training unified model on {device}")
    
    dataset = UnifiedStockDataset(stocks, kline_type, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MultiModalTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    model_path = os.path.join(model_dir, f"unified_{kline_type}_transformer.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    kline_type = "weekly"
    
    logger.info("Training unified model for all stocks")
    train_model(stocks, kline_type)