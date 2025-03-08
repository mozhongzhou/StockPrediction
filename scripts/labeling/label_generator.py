# scripts/labeling/label_generator.py
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
input_dir = os.path.join(base_dir, "data", "processed", "aligned")
output_dir = os.path.join(base_dir, "data", "processed", "labeled")
os.makedirs(output_dir, exist_ok=True)

def label_data(ticker, kline_type, days_ahead=None):
    aligned_file = os.path.join(input_dir, f"{ticker}_{kline_type}_aligned.csv")
    if not os.path.exists(aligned_file):
        logger.error(f"Aligned file not found: {aligned_file}")
        return
    
    df = pd.read_csv(aligned_file, parse_dates=["Date"], dtype={"10k_date": str})  # 强制 10k_date 为字符串
    if days_ahead is None:
        days_ahead = 7 if kline_type == "daily" else 1
    
    df["future_close"] = df["Close"].shift(-days_ahead)
    df["label"] = (df["future_close"] > df["Close"]).astype(int)
    df = df.dropna(subset=["future_close", "label"])
    
    df["10k_date"] = df["10k_date"].astype(str)  # 确保保存时为字符串
    labeled_file = os.path.join(output_dir, f"{ticker}_{kline_type}_labeled.csv")
    df.to_csv(labeled_file, index=False)
    logger.info(f"Labeled data saved: {labeled_file}, rows: {len(df)}, days_ahead: {days_ahead}")

def main():
    stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    kline_types = ["daily", "weekly"]
    
    for stock in stocks:
        logger.info(f"Labeling data for {stock}")
        for kline_type in kline_types:
            aligned_file = os.path.join(input_dir, f"{stock}_{kline_type}_aligned.csv")
            if os.path.exists(aligned_file):
                label_data(stock, kline_type)
            else:
                logger.warning(f"No {kline_type} aligned file found for {stock}, skipping")

if __name__ == "__main__":
    main()