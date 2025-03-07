# prepare_labels.py
# 生成标签，从 K 线数据算未来 4 周涨跌
# 支持 AAPL、NVDA、TSLA、GOOGL 四个股票

# 导入必要的库
import os  # 用于处理文件路径，确保读写路径正确
import numpy as np  # 处理数组，保存标签到 .npy 文件
import pandas as pd  # 读取 CSV，提取收盘价和日期

# 设置输入输出路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
kline_dir = os.path.join(base_dir, "data", "processed", "kline")
output_dir = os.path.join(base_dir, "data", "processed", "multimodal")
os.makedirs(output_dir, exist_ok=True)
print(f"K-line from: {kline_dir}")
print(f"Saving to: {output_dir}")

# 定义要处理的股票列表
stocks = ["AAPL", "NVDA", "TSLA", "GOOGL"]

# 定义生成标签函数
def prepare_labels(stock, future_weeks=4):
    # 加载清洗后的 K 线数据
    csv_file = os.path.join(kline_dir, f"{stock}_weekly_cleaned.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found, skipping {stock}")
        return
    df = pd.read_csv(csv_file)
    close_prices = df["Close"].values  # 收盘价数组，比如 635 个
    dates = pd.to_datetime(df["Date"])  # 日期，调试用
    
    # 加载 K 线序列
    sequences_file = os.path.join(kline_dir, f"{stock}_sequences.npy")
    if not os.path.exists(sequences_file):
        print(f"Error: {sequences_file} not found, skipping {stock}")
        return
    sequences = np.load(sequences_file)  # 形状 (532, 104, 5)
    print(f"Loaded {stock} sequences: {sequences.shape}")
    
    # 生成标签：未来 4 周涨跌
    labels = []
    valid_samples = len(close_prices) - 104 - future_weeks + 1  # 有效样本数，比如 528
    for i in range(len(sequences)):
        current_end = i + 104 - 1  # 序列最后一周索引（比如 103）
        future_end = current_end + future_weeks  # 未来 4 周索引（比如 107）
        
        if future_end < len(close_prices):  # 有未来数据
            current_price = close_prices[current_end]
            future_price = close_prices[future_end]
            label = 1 if future_price > current_price else 0
            labels.append(label)
        else:
            break  # 数据不够，跳出
    
    # 补齐到序列数
    if len(labels) < len(sequences):
        last_label = labels[-1] if labels else 0  # 用最后有效标签填充
        labels.extend([last_label] * (len(sequences) - len(labels)))
    labels = np.array(labels, dtype=int)  # 转成整数数组
    print(f"Generated {stock} labels: {len(labels)} samples, valid up to {valid_samples}")
    
    # 保存标签
    np.save(os.path.join(output_dir, f"{stock}_labels.npy"), labels)
    print(f"Saved {stock} labels: {labels.shape}")
    print(f"----- Finished labeling {stock} -----\n")

# 主程序：循环处理 4 个股票
for stock in stocks:
    print(f"----- Starting labeling for {stock} -----\n")
    prepare_labels(stock, future_weeks=4)