from PyPDF2 import PdfReader
import os
import glob
from concurrent.futures import ProcessPoolExecutor

# 获取项目根目录
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
# 相对路径到 data/raw/reports/
input_dir = os.path.join(base_dir, "data", "raw", "reports")
os.makedirs(input_dir, exist_ok=True)  # 确保目录存在

def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
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

if __name__ == '__main__':
    # 获取目录中所有PDF文件
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    # 使用多进程执行
    with ProcessPoolExecutor() as executor:
        list(executor.map(process_pdf, pdf_files))