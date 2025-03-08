from PyPDF2 import PdfReader
import os
import glob

# 获取项目根目录
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# 相对路径到 data/raw/reports/
input_dir = os.path.join(base_dir, "data", "raw", "reports")
os.makedirs(input_dir, exist_ok=True)  # 确保目录存在

# 获取目录中所有PDF文件
pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))

for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    
    # 生成对应的txt文件路径，保持与pdf文件名一致
    txt_path = os.path.join(input_dir, filename.replace(".pdf", ".txt"))
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {txt_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")