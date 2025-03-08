import yfinance as yf
import pandas as pd
import os

# 股票代码列表
stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
start_date = "2013-01-01"
end_date = "2025-03-09"

# 创建保存路径（使用相对路径）
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))  # 项目根目录
output_dir = os.path.join(base_dir, "data", "raw", "kline")
os.makedirs(output_dir, exist_ok=True)
print(f"Data will be saved to: {output_dir}")

# 下载数据
for stock in stocks:
    print(f"Fetching data for {stock}...")
    df = yf.download(stock, start=start_date, end=end_date, interval="1d")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    # "开" "高" "低" "收" "量"
    # 文件保存路径
    file_path = os.path.join(output_dir, f"{stock}_daily.csv")
    
    # 保存数据
    df.to_csv(file_path)
    print(f"Saved {stock} data: {len(df)} days to {file_path}")