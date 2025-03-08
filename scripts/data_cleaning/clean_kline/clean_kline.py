# scripts/data_preprocessing/clean_kline.py
# 预处理 K 线数据，清洗并规范化，同时支持日线和周线，增强健壮性

# 导入必要的库
import pandas as pd  # 处理表格数据，读取和清洗 CSV
import os  # 管理文件路径和目录
import glob  # 查找匹配模式的文件
import logging  # 添加日志记录，追踪错误和过程
from typing import List, Tuple, Optional  # 类型提示，提升代码可读性

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 设置输入输出路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))  # 项目根目录
input_dir = os.path.join(base_dir, "data", "raw", "kline")  # 原始 K 线数据目录
output_dir = os.path.join(base_dir, "data", "processed", "kline")  # 清洗后数据目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Reading from: {input_dir}")
logger.info(f"Saving to: {output_dir}")

# 定义股票列表
stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

def find_kline_files(stock: str) -> List[Tuple[str, str]]:
    """
    查找匹配的 K 线文件，返回所有日线和周线文件。
    
    参数:
        stock (str): 股票代码
    
    返回:
        List[Tuple[str, str]]: 包含 (文件路径, 文件类型) 的列表，例如 [(daily_file, "daily"), (weekly_file, "weekly")]
    """
    pattern = os.path.join(input_dir, f"{stock}_*.csv").replace("\\", "/")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        logger.error(f"No matching files found for {stock} with pattern {pattern}")
        return []
    
    files = []
    for file in matching_files:
        file_lower = file.lower()
        if "daily" in file_lower:
            files.append((file, "daily"))
        elif "weekly" in file_lower:
            files.append((file, "weekly"))
        else:
            logger.warning(f"No 'daily' or 'weekly' in filename for {stock}, treating {file} as unknown")
            files.append((file, "unknown"))
    
    logger.info(f"Found {len(files)} files for {stock}: {[f[0] for f in files]}")
    return files

def clean_kline(stock: str, file_path: str, file_type: str) -> None:
    """
    清洗指定股票的 K 线数据，确保格式规范并保存。
    
    参数:
        stock (str): 股票代码
        file_path (str): 输入文件路径
        file_type (str): 文件类型 ("daily", "weekly", "unknown")
    """
    logger.info(f"Processing {file_path} as {file_type} data")
    
    try:
        # 读取 CSV，动态跳过前两行或无效行
        df = pd.read_csv(file_path, skiprows=lambda x: x < 2 or pd.isna(pd.read_csv(file_path, nrows=1).iloc[0, 0]))
        logger.info(f"Loaded {stock} data: {len(df)} rows")
        logger.debug(f"Original columns: {df.columns.tolist()}")
        
        # 检查列数并规范化列名
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if len(df.columns) != len(expected_cols):
            logger.error(f"Unexpected column count for {stock}: {len(df.columns)} (expected {len(expected_cols)})")
            return
        
        df.columns = expected_cols
        
        # 转换日期格式，处理无效日期
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = df["Date"].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropped {invalid_dates} rows with invalid dates for {stock}")
        df = df.dropna(subset=["Date"]).set_index("Date")
        
        # 转换为数值类型，处理非数值
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 清洗缺失值和异常值
        original_len = len(df)
        df = df.dropna()  # 删除任何包含 NaN 的行
        df = df[(df["Volume"] >= 0) & (df["Open"] >= 0) & (df["High"] >= 0) & 
                (df["Low"] >= 0) & (df["Close"] >= 0) & (df["High"] >= df["Low"])]
        
        cleaned_len = len(df)
        logger.info(f"After cleaning {stock}: {cleaned_len} rows (removed {original_len - cleaned_len} rows)")
        
        if cleaned_len == 0:
            logger.error(f"No valid data remains for {stock} after cleaning")
            return
        
        # 保存清洗后的数据
        cleaned_file = os.path.join(output_dir, f"{stock}_{file_type}_cleaned.csv").replace("\\", "/")
        df.to_csv(cleaned_file)
        logger.info(f"Saved cleaned {stock} data to: {cleaned_file}")
    
    except Exception as e:
        logger.error(f"Error processing {stock}: {str(e)}", exc_info=True)

def main():
    """主程序：循环清洗每支股票的 K 线数据，同时处理日线和周线"""
    for stock in stocks:
        logger.info(f"----- Cleaning data for {stock} -----")
        files = find_kline_files(stock)
        if not files:
            logger.info(f"No files to process for {stock}")
            continue
        for file_path, file_type in files:
            clean_kline(stock, file_path, file_type)
        logger.info(f"----- Finished cleaning {stock} -----\n")

if __name__ == "__main__":
    main()