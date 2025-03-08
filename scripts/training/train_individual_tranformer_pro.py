import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
import time
from datetime import datetime

# 设置日志
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
labeled_dir = os.path.join(base_dir, "data", "processed", "labeled")
report_dir = os.path.join(base_dir, "data", "raw", "reports")
model_dir = os.path.join(base_dir, "models", "individual")
viz_dir = os.path.join(base_dir, "outputs", "visualizations")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

# 数据集
class StockDataset(Dataset):
    def __init__(self, ticker, kline_type="weekly", seq_len=30, max_text_len=512):
        try:
            self.df = pd.read_csv(
                os.path.join(labeled_dir, f"{ticker}_{kline_type}_labeled.csv"),
                parse_dates=["Date"],
                dtype={"10k_date": str}  # 强制 10k_date 为字符串
            )
            logger.info(f"Loaded dataset for {ticker} with {len(self.df)} rows")
        except Exception as e:
            logger.error(f"Failed to load dataset for {ticker}: {str(e)}")
            raise
        
        self.seq_len = seq_len
        self.max_text_len = max_text_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ticker = ticker
    
    def __len__(self):
        return len(self.df) - self.seq_len
    
    def __getitem__(self, idx):
        try:
            kline = self.df.iloc[idx:idx+self.seq_len][["Open", "High", "Low", "Close", "Volume"]].values
            # 数据归一化处理，提高模型稳定性
            kline = self.normalize_kline(kline)
            kline = torch.tensor(kline, dtype=torch.float32)
            
            report_date = str(self.df.iloc[idx]["10k_date"]).replace(".0", "")  # 转换为字符串并移除 .0
            if report_date == 'nan' or not report_date:
                logger.debug(f"Invalid 10-K date for {self.ticker} at index {idx}, using empty text")
                text = ""
            else:
                report_file = os.path.join(report_dir, f"{self.ticker}_10-K_{report_date}.txt")
                if not os.path.exists(report_file):
                    logger.debug(f"10-K file not found: {report_file}, using empty text")
                    text = ""
                else:
                    with open(report_file, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()[:100000]
            
            text_inputs = self.tokenizer(text, max_length=self.max_text_len, truncation=True, padding="max_length", return_tensors="pt")
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
            
            label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
            return kline, text_inputs, label
        except Exception as e:
            logger.error(f"Error getting item {idx} for {self.ticker}: {str(e)}")
            # 返回一个有效但可能为空的项，防止加载器崩溃
            empty_kline = torch.zeros((self.seq_len, 5), dtype=torch.float32)
            empty_text = self.tokenizer("", max_length=self.max_text_len, truncation=True, 
                                       padding="max_length", return_tensors="pt")
            empty_text = {k: v.squeeze(0) for k, v in empty_text.items()}
            return empty_kline, empty_text, torch.tensor(0, dtype=torch.long)
    
    def normalize_kline(self, kline):
        """对K线数据进行归一化处理，提高模型稳定性"""
        # 按列归一化 OHLC
        for i in range(4):  # 0,1,2,3 分别是 Open, High, Low, Close
            mean = np.mean(kline[:, i])
            std = np.std(kline[:, i])
            if std > 0:
                kline[:, i] = (kline[:, i] - mean) / std
        
        # 对Volume单独归一化，使用最大-最小缩放法
        vol_max = np.max(kline[:, 4])
        vol_min = np.min(kline[:, 4])
        if vol_max > vol_min:
            kline[:, 4] = (kline[:, 4] - vol_min) / (vol_max - vol_min)
        
        return kline

# Transformer 模型
class MultiModalTransformer(nn.Module):
    def __init__(self, kline_dim=5, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.kline_proj = nn.Linear(kline_dim, hidden_dim)
        self.kline_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.text_embedding = nn.Embedding(BertTokenizer.from_pretrained("bert-base-uncased").vocab_size, hidden_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重，提高模型稳定性
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, kline, text_inputs):
        kline = self.kline_proj(kline)
        kline_out = self.kline_encoder(kline)[:, -1, :]
        text_ids = text_inputs["input_ids"]
        attention_mask = text_inputs.get("attention_mask", None)
        
        text_embed = self.text_embedding(text_ids)
        text_out = self.text_encoder(text_embed)[:, 0, :]
        
        combined = torch.cat([kline_out, text_out], dim=-1)
        combined = self.fusion(combined)
        combined = self.dropout(torch.relu(combined))
        logits = self.fc(combined)
        return logits

# 训练函数
def train_model(ticker, kline_type="weekly", seq_len=30, batch_size=64, num_epochs=20, lr=1e-4, val_ratio=0.2):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training {ticker} on {device}")
        
        # 加载数据集
        try:
            full_dataset = StockDataset(ticker, kline_type, seq_len)
            
            # 划分训练集和验证集
            val_size = int(len(full_dataset) * val_ratio)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        except Exception as e:
            logger.error(f"Error preparing dataset for {ticker}: {str(e)}")
            return
            
        # 初始化模型、优化器和损失函数
        model = MultiModalTransformer().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = GradScaler()  # 混合精度训练
        
        # 添加学习率调度器
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # 记录训练过程
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # 早停机制
        best_val_loss = float('inf')
        best_model_path = os.path.join(model_dir, f"{ticker}_{kline_type}_transformer_best.pth")
        patience = 5
        counter = 0
        
        # 开始训练
        start_time = time.time()
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            epoch_loss = 0
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            
            for kline, text_inputs, labels in train_progress:
                kline = kline.to(device)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(kline, text_inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                # 梯度裁剪，提高训练稳定性
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                epoch_loss += loss.item()
                train_progress.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_train_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            with torch.no_grad():
                for kline, text_inputs, labels in val_progress:
                    kline = kline.to(device)
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    labels = labels.to(device)
                    
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(kline, text_inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_progress.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f}, "
                       f"Val F1: {val_f1:.4f}")
            
            # 检查早停条件
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # 训练完成后保存最终模型
        final_model_path = os.path.join(model_dir, f"{ticker}_{kline_type}_transformer.pth")
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # 可视化训练过程
        visualize_training(ticker, kline_type, history)
        
        # 记录训练总时间
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        
        return history
    
    except Exception as e:
        logger.error(f"Error in training {ticker}: {str(e)}", exc_info=True)
        return None

# 可视化训练过程
def visualize_training(ticker, kline_type, history):
    try:
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{ticker} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率和F1分数
        plt.subplot(2, 1, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
        plt.plot(history['val_f1'], label='Validation F1 Score', color='purple')
        plt.title(f'{ticker} Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图形
        viz_path = os.path.join(viz_dir, f"{ticker}_{kline_type}_training_viz.png")
        plt.savefig(viz_path)
        logger.info(f"Training visualization saved to: {viz_path}")
        
        # 关闭图形以释放内存
        plt.close()
    except Exception as e:
        logger.error(f"Error in visualization for {ticker}: {str(e)}")

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    kline_type = "weekly"
    
    logger.info(f"Starting training process with {len(stocks)} stocks")
    
    # 创建股票训练进度条
    stock_progress = tqdm(stocks, desc="Training stocks")
    
    for stock in stock_progress:
        stock_progress.set_description(f"Training {stock}")
        labeled_file = os.path.join(labeled_dir, f"{stock}_{kline_type}_labeled.csv")
        
        if os.path.exists(labeled_file):
            history = train_model(stock, kline_type)
            if history:
                logger.info(f"Successfully trained model for {stock}")
        else:
            logger.warning(f"No {kline_type} labeled file for {stock}, skipping")
    
    logger.info("All training completed!")