# preprocess_kline.py
# 预处理 K 线数据，清洗和序列化成模型输入
# 读取原始 K 线 CSV，清洗缺失值和异常值，归一化，切成 104 周序列，存成 .npy 文件
# 用相对路径，确保代码在任何机器上都能跑


# 导入必要的库
import pandas as pd  # 处理表格数据，读 CSV、清洗、切片都靠它
import numpy as np  # 处理数组，存序列化的 K 线数据到 .npy 文件
import os  # 管理文件路径和目录，确保读写路径正确

# 设置输入输出路径
# 用相对路径，让代码在任何机器上都能跑，不写死绝对路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 项目根目录，从 scripts/ 跳到上一级
input_dir = os.path.join(base_dir, "data", "raw", "kline")  # 原始 K 线 CSV 文件的目录
output_dir = os.path.join(base_dir, "data", "processed", "kline")  # 清洗后数据的目录（CSV 和 .npy）
os.makedirs(output_dir, exist_ok=True)  # 如果 output_dir 不存在就创建，exist_ok=True 避免重复创建报错
print(f"Reading from: {input_dir}")  # 打印输入路径，调试用，确认读对地方
print(f"Saving to: {output_dir}")  # 打印输出路径，确认保存位置

# 定义要处理的股票列表
# 这 5 支股票是预测目标，跟财报数据对齐
stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# 定义预处理函数，清洗和序列化每支股票的 K 线数据
def preprocess_kline(stock):
    # 读取 K 线数据
    # 文件名比如 "NVDA_weekly.csv"，存每周的 OHLCV 数据
    file_path = os.path.join(input_dir, f"{stock}_weekly.csv")
    # 读 CSV，跳过前两行（可能是标题或无用行），第三行是列名
    df = pd.read_csv(file_path, skiprows=2, header=0)
    print(f"Loaded {stock} data: {len(df)} weeks")  # 打印加载的周数，比如 635 周，检查数据量
    print(f"Columns: {df.columns.tolist()}")  # 打印列名，调试用，看原始数据格式对不对

    # 重命名列，确保一致性
    # 原始 CSV 列名可能乱（比如 "O", "H"），强制改成标准名
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df.set_index("Date", inplace=True)  # 把 "Date" 设为索引，方便按时间操作
    df.index = pd.to_datetime(df.index)  # 转成日期格式（比如 "2013-01-01"），后续对齐用

    # 清洗缺失值
    # 如果某行有空值（NaN），直接删掉，保证数据完整
    df = df.dropna()
    # 检查数据够不够长，少于 104 周没法切序列，就跳过
    if len(df) < 104:
        print(f"Warning: {stock} has insufficient data ({len(df)} weeks), skipping")
        return  # 退出函数，不处理这支股票
    
    # 检查异常值
    # K 线数据不该有负值（价格、成交量都得 ≥ 0），去掉不符合的行
    df = df[(df["Volume"] >= 0) & (df["Open"] >= 0) & (df["High"] >= 0) & 
            (df["Low"] >= 0) & (df["Close"] >= 0)]
    print(f"After cleaning {stock}: {len(df)} weeks")  # 清洗后剩多少周，确认没丢太多数据

    # 保存清洗后的 CSV
    # 存一份中间结果，方便检查，比如 "NVDA_weekly_cleaned.csv"
    cleaned_file = os.path.join(output_dir, f"{stock}_weekly_cleaned.csv")
    df.to_csv(cleaned_file)  # 保存带索引的 CSV，日期也在里面
    print(f"Saved cleaned {stock} data to: {cleaned_file}")  # 提示保存成功

    # 归一化
    # 把每列（Open, High, Low, Close, Volume）标准化成均值 0、标准差 1
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()  # Z-score 公式，消除量级差异
    # 为啥归一化？模型（Transformer）对数据大小敏感，原始值差别太大（比如 Volume 是百万，Close 是百）

    # 构造序列（104 周）
    seq_length = 104  # 定窗口大小 104 周（约 2 年），跟财报时间尺度搭
    sequences = []  # 存所有序列，最后转成数组
    # 滑动窗口切片，步长 1 周
    for i in range(len(df) - seq_length + 1):
        seq = df.iloc[i:i + seq_length].values  # 取 104 周的 5 列数据，形状 (104, 5)
        sequences.append(seq)  # 加到列表里
    # 比如 635 周，切成 635 - 104 + 1 = 532 个序列

    # 保存序列为 .npy
    # 转成 numpy 数组，存成二进制文件，方便模型读取
    output_file = os.path.join(output_dir, f"{stock}_sequences.npy")
    np.save(output_file, np.array(sequences))  # 形状比如 (532, 104, 5)
    print(f"Saved {stock} sequences: {len(sequences)} samples, shape: {np.array(sequences).shape}")  # 确认保存和形状

# 主程序：循环处理每支股票
for stock in stocks:
    preprocess_kline(stock)  # 对每支股票跑一遍清洗和序列化