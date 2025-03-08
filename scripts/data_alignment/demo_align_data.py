# align_data.py
# 对齐 K 线和财报数据，生成多模态输入
# 支持任意 K 线范围和年报数量，动态匹配时间

# 导入必要的库
import os
import numpy as np
import pandas as pd

# 设置输入输出路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))  # 项目根目录
kline_dir = os.path.join(base_dir, "data", "processed", "kline")
report_dir = os.path.join(base_dir, "data", "processed", "reports")
output_dir = os.path.join(base_dir, "data", "processed", "multimodal")
os.makedirs(output_dir, exist_ok=True)
print(f"K-line from: {kline_dir}")
print(f"Reports from: {report_dir}")
print(f"Saving to: {output_dir}")

# 定义要处理的股票列表
stocks = ["AAPL", "NVDA", "TSLA", "GOOGL"]

# 定义对齐函数，处理单个股票
def align_data(stock):
    # 加载 K 线序列
    kline_file = os.path.join(kline_dir, f"{stock}_sequences.npy")
    if not os.path.exists(kline_file):
        print(f"Error: {kline_file} not found, skipping {stock}")
        return
    sequences = np.load(kline_file)
    print(f"Loaded {stock} kline: {sequences.shape}")
    
    # 加载年报嵌入
    report_file = os.path.join(report_dir, f"{stock}_embeddings.npy")
    if not os.path.exists(report_file):
        print(f"Error: {report_file} not found, skipping {stock}")
        return
    embeddings = np.load(report_file)
    print(f"Loaded {stock} report: {embeddings.shape}")
    
    # 加载 K 线日期
    csv_file = os.path.join(kline_dir, f"{stock}_weekly_cleaned.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found, skipping {stock}")
        return
    df = pd.read_csv(csv_file)
    dates = pd.to_datetime(df["Date"])
    
    # 动态获取年报年份
    # 假设 embeddings 的顺序跟 10-K 年份对应，从文件名或配置推算
    # 这里简化为 2013 开始，实际可从 preprocess_report.py 传参
    report_years = list(range(2013, 2013 + len(embeddings)))  # 比如 [2013, 2014, ..., 2024]
    print(f"{stock} report years: {report_years}")
    
    # 对齐年报到 K 线序列
    aligned_embeddings = []
    for i in range(len(sequences)):
        start_date = dates[i]
        year = start_date.year
        # 找最近的年报年份
        closest_year_idx = min(range(len(report_years)), key=lambda j: abs(report_years[j] - year))
        if year < report_years[0]:
            # 早于最早年报，用第一份
            aligned_embeddings.append(embeddings[0])
            print(f"Warning: {stock} sequence {i} (year {year}) before range, using {report_years[0]}")
        elif year > report_years[-1]:
            # 晚于最后年报，用最后一份
            aligned_embeddings.append(embeddings[-1])
            print(f"Warning: {stock} sequence {i} (year {year}) after range, using {report_years[-1]}")
        else:
            # 在范围内，用最近的
            aligned_embeddings.append(embeddings[closest_year_idx])
    aligned_embeddings = np.array(aligned_embeddings)
    
    # 保存对齐后的数据
    np.save(os.path.join(output_dir, f"{stock}_kline.npy"), sequences)
    np.save(os.path.join(output_dir, f"{stock}_report.npy"), aligned_embeddings)
    print(f"Saved {stock} multimodal data: kline {sequences.shape}, report {aligned_embeddings.shape}")
    print(f"----- Finished aligning {stock} -----\n")

# 主程序：循环处理股票
for stock in stocks:
    print(f"----- Starting alignment for {stock} -----\n")
    align_data(stock)