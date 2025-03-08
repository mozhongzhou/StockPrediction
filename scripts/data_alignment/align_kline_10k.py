# scripts/data_alignment/align_kline_10k.py
import os
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
kline_dir = os.path.join(base_dir, "data", "processed", "kline")
report_dir = os.path.join(base_dir, "data", "raw", "reports")
output_dir = os.path.join(base_dir, "data", "processed", "aligned")
os.makedirs(output_dir, exist_ok=True)

def align_kline_and_10k(ticker, kline_type):
    kline_file = os.path.join(kline_dir, f"{ticker}_{kline_type}_cleaned.csv")
    if not os.path.exists(kline_file):
        logger.error(f"K-line file not found: {kline_file}")
        return
    
    kline_df = pd.read_csv(kline_file, parse_dates=["Date"])
    logger.info(f"Loaded {kline_file}: {len(kline_df)} rows")
    
    reports = [f for f in os.listdir(report_dir) if f.startswith(f"{ticker}_10-K_") and f.endswith(".txt")]
    if not reports:
        logger.error(f"No 10-K reports found for {ticker} in {report_dir}")
        return
    
    report_dates = sorted([datetime.strptime(f.split("_")[2].split(".")[0], "%Y%m%d") for f in reports])
    logger.info(f"Found {len(report_dates)} 10-K reports for {ticker}")
    
    kline_df["10k_date"] = None
    for i, report_date in enumerate(report_dates):
        end_date = report_dates[i + 1] if i + 1 < len(report_dates) else datetime.now()
        mask = (kline_df["Date"] >= report_date) & (kline_df["Date"] < end_date)
        kline_df.loc[mask, "10k_date"] = report_date.strftime("%Y%m%d")
    
    kline_df["10k_date"] = kline_df["10k_date"].astype(str)  # 强制转为字符串
    output_file = os.path.join(output_dir, f"{ticker}_{kline_type}_aligned.csv")
    kline_df.to_csv(output_file, index=False)
    logger.info(f"Alignment completed: {output_file}")

def main():
    stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    kline_types = ["daily", "weekly"]
    
    for stock in stocks:
        logger.info(f"Aligning data for {stock}")
        for kline_type in kline_types:
            kline_file = os.path.join(kline_dir, f"{stock}_{kline_type}_cleaned.csv")
            if os.path.exists(kline_file):
                align_kline_and_10k(stock, kline_type)
            else:
                logger.warning(f"No {kline_type} K-line file found for {stock}, skipping")

if __name__ == "__main__":
    main()