import pandas as pd
import numpy as np
import os

# 输入输出路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_dir = os.path.join(base_dir, "data", "raw", "kline")
output_dir = os.path.join(base_dir, "data", "processed", "kline")
os.makedirs(output_dir, exist_ok=True)
print(f"Reading from: {input_dir}")
print(f"Saving to: {output_dir}")

# 股票列表
stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# 预处理函数
def preprocess_kline(stock):
    # 读取数据
    file_path = os.path.join(input_dir, f"{stock}_weekly.csv")
    # 读取 CSV，跳过前两行（Price 和 Ticker），用第三行作为列名
    df = pd.read_csv(file_path, skiprows=2, header=0)
    print(f"Loaded {stock} data: {len(df)} weeks")
    print(f"Columns: {df.columns.tolist()}")  # 调试列名
    
    # 重命名列，确保一致性
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)  # 转换为日期格式

    # 清洗缺失值
    df = df.dropna()
    if len(df) < 104:
        print(f"Warning: {stock} has insufficient data ({len(df)} weeks), skipping")
        return
    
    # 检查异常值
    df = df[(df["Volume"] >= 0) & (df["Open"] >= 0) & (df["High"] >= 0) & 
            (df["Low"] >= 0) & (df["Close"] >= 0)]
    print(f"After cleaning {stock}: {len(df)} weeks")

    # 保存清洗后的 CSV
    cleaned_file = os.path.join(output_dir, f"{stock}_weekly_cleaned.csv")
    df.to_csv(cleaned_file)
    print(f"Saved cleaned {stock} data to: {cleaned_file}")

    # 归一化
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # 构造序列（104 周）
    seq_length = 104
    sequences = []
    for i in range(len(df) - seq_length + 1):
        seq = df.iloc[i:i + seq_length].values  # 104 周 × 5 维
        sequences.append(seq)
    
    # 保存序列为 .npy
    output_file = os.path.join(output_dir, f"{stock}_sequences.npy")
    np.save(output_file, np.array(sequences))
    print(f"Saved {stock} sequences: {len(sequences)} samples, shape: {np.array(sequences).shape}")

# 执行
for stock in stocks:
    preprocess_kline(stock)