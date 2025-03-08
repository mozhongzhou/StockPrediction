import os
import re
from datetime import datetime
import PyPDF2

directory = "data/raw/reports"  # 你的数据路径

DATE_PATTERN = re.compile(
    r"(?:for\s+the\s+)?fiscal\s+year\s+(?:ended|ending|ended\s+on)?\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE
)

MONTHS_MAPPING = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May", "Jun": "June",
    "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"
}

def normalize_date(date_str):
    parts = date_str.split()
    if len(parts) == 3 and parts[0] in MONTHS_MAPPING:
        parts[0] = MONTHS_MAPPING[parts[0]]
        date_str = " ".join(parts)
    return date_str

def extract_date(text):
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    match = DATE_PATTERN.search(text)
    if match:
        date_str = normalize_date(match.group(1))
        try:
            date_obj = datetime.strptime(date_str, "%B %d, %Y")
            return date_obj.strftime("%Y%m%d")
        except ValueError:
            return None
    return None

def extract_date_from_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() or "" for page in reader.pages[:5]])
            return extract_date(text)
    except Exception:
        return None

def extract_date_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read(5000)
        return extract_date(text)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            text = f.read(5000)
        return extract_date(text)
    except Exception:
        return None

def rename_10k_files(directory):
    for filename in os.listdir(directory):
        match = re.match(r"([A-Z]+)_10-K_.*\.(pdf|txt)$", filename, re.IGNORECASE)
        if not match:
            continue
        
        ticker = match.group(1).upper()
        file_path = os.path.join(directory, filename)
        
        date = extract_date_from_pdf(file_path) if filename.endswith(".pdf") else extract_date_from_txt(file_path)
        if not date:
            print(f"无法提取日期: {filename}")
            continue
        
        new_filename = f"{ticker}_10-K_{date}{os.path.splitext(filename)[1]}"
        new_file_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(file_path, new_file_path)
            print(f"重命名: {filename} -> {new_filename}")
        except Exception:
            print(f"重命名失败: {filename}")

if __name__ == "__main__":
    rename_10k_files(directory)