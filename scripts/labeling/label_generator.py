# scripts/labeling/label_generator.py
import os
import pandas as pd

def label_data(ticker, days_ahead=5):
    aligned_file = os.path.join("data/processed/aligned", f"{ticker}_aligned.csv")
    df = pd.read_csv(aligned_file, parse_dates=["date"])
    
    df["future_close"] = df["close"].shift(-days_ahead)
    df["label"] = (df["future_close"] > df["close"]).astype(int)
    
    df = df.dropna(subset=["future_close", "label"])
    
    labeled_file = os.path.join("data/processed/labeled", f"{ticker}_labeled.csv")
    os.makedirs(os.path.dirname(labeled_file), exist_ok=True)
    df.to_csv(labeled_file, index=False)
    print(f"标注完成: {labeled_file}")

if __name__ == "__main__":
    ticker = "AAPL"  # 可改为参数化
    label_data(ticker)