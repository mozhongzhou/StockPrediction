# scripts/data_alignment/align_kline_10k.py
import os  # 用于文件路径操作
import pandas as pd  # 处理表格数据（如 CSV）
from datetime import datetime  # 处理日期时间

# 定义数据目录
kline_dir = "data/processed/kline"  # K 线数据存储目录，例如 daily 或 weekly K 线 CSV 文件
report_dir = "data/raw/reports"     # 10-K 年报存储目录，包含 PDF 文件

def align_kline_and_10k(ticker):
    """
    将指定股票的 K 线数据与 10-K 年报按时间对齐。
    参数:
        ticker (str): 股票代码，例如 "AAPL"（苹果公司）
    """
    # 构建 K 线文件路径，例如 "data/processed/kline/AAPL_kline.csv"
    kline_file = os.path.join(kline_dir, f"{ticker}_kline.csv")
    
    # 读取 K 线数据，假设 CSV 包含 "date" 列，自动解析为 datetime 类型
    # 例如：date,open,high,low,close,volume
    kline_df = pd.read_csv(kline_file, parse_dates=["date"])
    
    # 获取 10-K 文件列表，筛选以 "{ticker}_10-K_" 开头且以 ".pdf" 结尾的文件
    # 示例：["AAPL_10-K_20201231.pdf", "AAPL_10-K_20211231.pdf"]
    reports = [f for f in os.listdir(report_dir) if f.startswith(f"{ticker}_10-K_") and f.endswith(".pdf")]
    
    # 从文件名中提取 10-K 的财年结束日期，并排序
    # 文件名格式：{ticker}_10-K_YYYYMMDD.pdf
    # split("_")[2] 取 "YYYYMMDD.pdf"，split(".")[0] 取 "YYYYMMDD"，转为 datetime 对象
    report_dates = sorted([datetime.strptime(f.split("_")[2].split(".")[0], "%Y%m%d") for f in reports])
    # 示例：report_dates = [2020-12-31, 2021-12-31]
    
    # 在 K 线数据框中添加 "10k_date" 列，初始值为 None，表示尚未分配 10-K 日期
    kline_df["10k_date"] = None
    
    # 遍历所有 10-K 日期，分配到对应的 K 线时间段
    for i, report_date in enumerate(report_dates):
        # 设置时间段的结束日期：
        # - 如果有下一份 10-K，则结束于下一份的日期
        # - 如果是最后一份，则结束于当前日期 (datetime.now())
        end_date = report_dates[i + 1] if i + 1 < len(report_dates) else datetime.now()
        
        # 创建掩码，筛选 K 线数据中日期在 [report_date, end_date) 之间的记录
        # 注意：>= 是包含开始日期，< 是不包含结束日期（半开区间）
        mask = (kline_df["date"] >= report_date) & (kline_df["date"] < end_date)
        
        # 将符合条件的 K 线记录的 "10k_date" 列设置为当前 10-K 的日期（格式化为 YYYYMMDD）
        kline_df.loc[mask, "10k_date"] = report_date.strftime("%Y%m%d")
    
    # 构建输出文件路径，例如 "data/processed/aligned/AAPL_aligned.csv"
    output_file = os.path.join("data/processed/aligned", f"{ticker}_aligned.csv")
    
    # 创建输出目录（如果不存在），exist_ok=True 避免重复创建时出错
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存对齐后的数据到 CSV 文件，不保存索引列
    kline_df.to_csv(output_file, index=False)
    
    # 打印完成信息
    print(f"对齐完成: {output_file}")

if __name__ == "__main__":
    # 主程序入口，测试用 AAPL 股票
    ticker = "AAPL"  # 可通过命令行参数化，例如 argparse
    align_kline_and_10k(ticker)