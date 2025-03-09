import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertModel
import logging
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 设置路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
labeled_dir = os.path.join(base_dir, "data", "processed", "labeled")
report_dir = os.path.join(base_dir, "data", "raw", "reports")
model_dir = os.path.join(base_dir, "models", "individual")
os.makedirs(model_dir, exist_ok=True)

# 数据预处理：标准化函数
def normalize_df(df, columns):
    """对指定列进行标准化（零均值，单位方差）"""
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:  # 防止除以零
            df[col] = (df[col] - mean) / std
    return df

# 数据集类
class StockDataset(Dataset):
    def __init__(self, ticker, kline_type="weekly", seq_len=30, max_text_len=512):
        """初始化数据集，加载股票数据和文本"""
        try:
            self.df = pd.read_csv(
                os.path.join(labeled_dir, f"{ticker}_{kline_type}_labeled.csv"),
                parse_dates=["Date"],
                dtype={"10k_date": str}
            )
            self.df = normalize_df(self.df, ["Open", "High", "Low", "Close", "Volume"])
        except FileNotFoundError:
            logger.error(f"File {ticker}_{kline_type}_labeled.csv not found")
            raise
        
        self.seq_len = seq_len
        self.max_text_len = max_text_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ticker = ticker
        
        if len(self.df) < seq_len:
            raise ValueError(f"Data length {len(self.df)} is less than sequence length {seq_len}")
    
    def __len__(self):
        """返回数据集长度（考虑序列长度）"""
        return len(self.df) - self.seq_len
    
    def __getitem__(self, idx):
        """获取单个样本：K 线序列、文本输入和标签"""
        kline = self.df.iloc[idx:idx + self.seq_len][["Open", "High", "Low", "Close", "Volume"]].values
        kline = torch.tensor(kline, dtype=torch.float32)
        
        report_date = str(self.df.iloc[idx]["10k_date"]).replace(".0", "")
        if pd.isna(report_date) or not report_date:
            text = ""
        else:
            report_file = os.path.join(report_dir, f"{self.ticker}_10-K_{report_date}.txt")
            text = ""
            if os.path.exists(report_file):
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        text = f.read()[:100000]
                except Exception as e:
                    logger.warning(f"Error reading {report_file}: {e}")
        
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_text_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return kline, text_inputs, label

# 多模态 Transformer 模型
class MultiModalTransformer(nn.Module):
    def __init__(self, kline_dim=5, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        """初始化模型，包含 K 线和文本的双 Transformer 结构"""
        super().__init__()
        self.kline_proj = nn.Linear(kline_dim, hidden_dim)
        self.kline_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(768, hidden_dim)
        
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, kline, text_inputs):
        """前向传播：处理 K 线和文本并融合"""
        kline = self.kline_proj(kline)
        kline_out = self.kline_encoder(kline)[:, -1, :]
        
        with torch.no_grad():
            text_out = self.text_model(**text_inputs).last_hidden_state[:, 0, :]
        text_out = self.text_proj(text_out)
        
        combined = torch.cat([kline_out, text_out], dim=-1)
        combined = self.fusion(combined)
        combined = self.dropout(torch.relu(combined))
        logits = self.fc(combined)
        return logits

# 训练和评估函数
def train_and_evaluate(ticker, kline_type="weekly", seq_len=30, batch_size=16, num_epochs=20, lr=5e-5):
    """训练模型并评估性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {ticker} on {device} (CUDA: {torch.cuda.get_device_name(0)})")
    
    try:
        dataset = StockDataset(ticker, kline_type, seq_len)
    except Exception as e:
        logger.error(f"Failed to load dataset for {ticker}: {e}")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = MultiModalTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')

    train_losses, val_losses = [], []
    val_accuracies, val_f1_scores = [], []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for kline, text_inputs, labels in train_loader:
            kline = kline.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(kline, text_inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for kline, text_inputs, labels in val_loader:
                kline = kline.to(device)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                labels = labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(kline, text_inputs)
                    loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        val_accuracies.append(accuracy)
        val_f1_scores.append(f1)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # 保存模型
    model_path = os.path.join(model_dir, f"{ticker}_{kline_type}_transformer.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ticker} Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Accuracy")
    plt.plot(val_f1_scores, label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{ticker} Metrics Curve")
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(model_dir, f"{ticker}_{kline_type}_training_plot.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training plot saved to: {plot_path}")  # 修复后的日志语句

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    kline_type = "weekly"
    
    logger.info("Training individual models for each stock")
    for stock in stocks:
        labeled_file = os.path.join(labeled_dir, f"{stock}_{kline_type}_labeled.csv")
        if os.path.exists(labeled_file):
            train_and_evaluate(stock, kline_type)
        else:
            logger.warning(f"No {kline_type} labeled file for {stock}, skipping")