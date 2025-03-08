# 导入必要的库
import os  # 用于处理文件路径，确保读写路径正确
import numpy as np  # 处理数组，加载和保存 K 线、财报数据
import pandas as pd  # 读取 CSV，提取 K 线日期用

# 设置输入输出路径
# 用相对路径，让代码在不同机器上都能跑
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))  # 项目根目录，从 scripts/ 跳到上一级
kline_dir = os.path.join(base_dir, "data", "processed", "kline")  # 清洗后的 K 线数据目录
report_dir = os.path.join(base_dir, "data", "processed", "reports")  # 清洗后的财报嵌入目录
output_dir = os.path.join(base_dir, "data", "processed", "multimodal")  # 对齐后的多模态数据目录
os.makedirs(output_dir, exist_ok=True)  # 如果 multimodal 目录不存在就创建，exist_ok=True 避免重复创建报错
print(f"K-line from: {kline_dir}")  # 打印 K 线数据路径，调试用，确认读对地方
print(f"Reports from: {report_dir}")  # 打印财报数据路径，确认来源
print(f"Saving to: {output_dir}")  # 打印保存路径，确认输出位置

# 定义对齐函数，只处理指定股票（NVDA）
def align_data(stock):
    # 加载 K 线序列
    # 文件比如 "NVDA_sequences.npy"，存了 532 个 104 周的序列
    kline_file = os.path.join(kline_dir, f"{stock}_sequences.npy")
    sequences = np.load(kline_file)  # 加载成数组，形状 (532, 104, 5)，5 是 OHLCV 特征
    print(f"Loaded {stock} kline: {sequences.shape}")  # 打印形状，确认数据没问题
    
    # 加载年报嵌入
    # 文件比如 "NVDA_embeddings.npy"，存了 12 个年度的 768 维向量
    report_file = os.path.join(report_dir, f"{stock}_embeddings.npy")
    embeddings = np.load(report_file)  # 加载成数组，形状 (12, 1, 768)
    print(f"Loaded {stock} report: {embeddings.shape}")  # 打印形状，确认加载正确
    
    # 加载 K 线日期
    # 从清洗后的 CSV "NVDA_weekly_cleaned.csv" 拿日期，匹配每个序列的起始时间
    df = pd.read_csv(os.path.join(kline_dir, f"{stock}_weekly_cleaned.csv"))
    dates = pd.to_datetime(df["Date"])  # 转成日期格式，比如 "2013-01-01"，长度是 635 周
    
    # 对齐年报到 K 线序列
    aligned_embeddings = []  # 存对齐后的财报向量，最后要有 532 个
    for i in range(len(sequences)):  # 循环 532 个 K 线样本
        start_date = dates[i]  # 第 i 个序列的起始日期，比如 "2013-01-01"
        year = start_date.year  # 提取年份，比如 2013
        idx = year - 2013  # 计算对应财报的索引，2013 是 0，2014 是 1，依次类推
        # 检查索引是否在范围内
        if 0 <= idx < len(embeddings):
            aligned_embeddings.append(embeddings[idx])  # 用对应年份的财报向量
        else:
            # 如果超范围（比如 2025 年没财报），用最后一份财报填充
            aligned_embeddings.append(embeddings[-1])
            print(f"Warning: {stock} sequence {i} (year {year}) out of report range, using last embedding")
    aligned_embeddings = np.array(aligned_embeddings)  # 转成数组，形状 (532, 1, 768)
    
    # 保存对齐后的数据
    # K 线数据不变，财报数据扩展到 532 个样本
    np.save(os.path.join(output_dir, f"{stock}_kline.npy"), sequences)  # 保存 K 线，(532, 104, 5)
    np.save(os.path.join(output_dir, f"{stock}_report.npy"), aligned_embeddings)  # 保存财报，(532, 1, 768)
    print(f"Saved {stock} multimodal data: kline {sequences.shape}, report {aligned_embeddings.shape}")

# 只处理 NVDA
align_data("NVDA")  # 调用函数，针对 NVDA 跑对齐