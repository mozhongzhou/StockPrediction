# 导入必要的库
import os  # 用于处理文件路径，确保读写路径正确
import numpy as np  # 处理数组，保存标签到 .npy 文件
import pandas as pd  # 读取 CSV，提取收盘价和日期

# 设置输入输出路径
# 用相对路径，让代码跨平台跑
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # 项目根目录，从 scripts/ 跳到上一级
kline_dir = os.path.join(base_dir, "data", "processed", "kline")  # 清洗后的 K 线数据目录
output_dir = os.path.join(base_dir, "data", "processed", "multimodal")  # 标签和对齐数据的保存目录
os.makedirs(output_dir, exist_ok=True)  # 如果 multimodal 目录不存在就创建，exist_ok=True 避免报错
print(f"K-line from: {kline_dir}")  # 打印 K 线数据路径，调试用
print(f"Saving to: {output_dir}")  # 打印保存路径，确认位置

# 定义生成标签函数，只处理 NVDA
def prepare_labels(stock, future_weeks=4):
    # 加载清洗后的 K 线数据
    # 用 "NVDA_weekly_cleaned.csv"，因为它有日期和原始收盘价（没归一化）
    df = pd.read_csv(os.path.join(kline_dir, f"{stock}_weekly_cleaned.csv"))
    close_prices = df["Close"].values  # 提取收盘价数组，比如 635 个值
    dates = pd.to_datetime(df["Date"])  # 提取日期，调试和对齐用
    
    # 加载序列化后的 K 线数据
    # "NVDA_sequences.npy" 有 532 个样本，确定标签数量
    sequences = np.load(os.path.join(kline_dir, f"{stock}_sequences.npy"))  # 形状 (532, 104, 5)
    print(f"Loaded {stock} sequences: {sequences.shape}")  # 确认序列数量
    
    # 生成标签：未来 4 周的涨跌
    labels = []  # 存每个样本的标签（0 或 1）
    valid_samples = len(close_prices) - 104 - future_weeks + 1  # 计算有效样本数，比如 528
    for i in range(len(sequences)):  # 循环 532 个样本
        current_end = i + 104 - 1  # 当前序列的最后一周索引（0-based，比如 103）
        future_end = current_end + future_weeks  # 未来 4 周的索引（比如 107）
        
        # 只处理有未来数据的样本
        if future_end < len(close_prices):  # 如果未来 4 周有数据
            current_price = close_prices[current_end]  # 当前序列最后一周的收盘价
            future_price = close_prices[future_end]  # 未来 4 周后的收盘价
            label = 1 if future_price > current_price else 0  # 涨=1，跌=0
            labels.append(label)
        else:
            # 数据不够的样本，后面填充
            break
    
    # 填满到 532 个样本
    if len(labels) < len(sequences):
        last_label = labels[-1] if labels else 0  # 用最后一个有效标签填充，或默认 0
        labels.extend([last_label] * (len(sequences) - len(labels)))  # 补齐到 532
    labels = np.array(labels, dtype=int)  # 转成纯整数数组，避免对象类型
    print(f"Generated {stock} labels: {len(labels)} samples, valid up to {valid_samples}")
    
    # 保存标签
    # 存成 "NVDA_labels.npy"，形状 (532,)，全是非空整数
    np.save(os.path.join(output_dir, f"{stock}_labels.npy"), labels)
    print(f"Saved {stock} labels: {len(labels)} samples, shape: {labels.shape}")

# 只处理 NVDA，预测未来 4 周涨跌
prepare_labels("NVDA", future_weeks=4)