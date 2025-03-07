from PyPDF2 import PdfReader
import os

# 获取项目根目录（假设脚本在 scripts/ 下）
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 相对路径到 data/raw/reports/
input_dir = os.path.join(base_dir, "data", "raw", "reports")
os.makedirs(input_dir, exist_ok=True)  # 确保目录存在

stocks = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
for stock in stocks:
    for year in range(2013, 2025):
        pdf_path = os.path.join(input_dir, f"{stock}_10-K_{year}.pdf")
        txt_path = os.path.join(input_dir, f"{stock}_10-K_{year}.txt")
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved {txt_path}")
        else:
            print(f"Warning: {pdf_path} not found")