import re
import spacy
from dateutil.parser import parse
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd

# ------------------
# 配置部分
# ------------------
# 加载英文NLP模型（先运行：python -m spacy download en_core_web_sm）
nlp = spacy.load("en_core_web_sm")

# 英文日期正则表达式（覆盖更多格式）
DATE_PATTERNS = [
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",  # Mar 15, 2023
    r"\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b",  # 15 March 2023
    r"\b\d{4}-\d{2}-\d{2}\b",          # ISO格式 2023-03-15
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",      # MM/DD/YYYY 或 DD/MM/YYYY
    r"\b\d{4}_\d{2}_\d{2}\b",          # 2023_03_15（某些PDF转换后可能出现）
    r"\b(?:Q1|Q2|Q3|Q4) \d{4}\b",      # 季报格式 Q4 2023
    r"\b\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b"  # 15-Mar-2023
]

COMBINED_DATE_REGEX = re.compile(
    "|".join(DATE_PATTERNS), 
    flags=re.IGNORECASE
)

# 英文关键词上下文（优先级从高到低）
KEYWORD_CONTEXTS = [
    {"keywords": ["release date", "report date", "filing date"], "window": 50},
    {"keywords": ["published", "issued", "dated"], "window": 30},
    {"keywords": ["quarter ended", "year ended"], "window": 100}
]

# ------------------
# 核心函数
# ------------------
def extract_dates_with_context(text):
    """用spacy提取日期实体及其上下文"""
    doc = nlp(text)
    date_entities = []
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            # 获取上下文窗口
            start = max(0, ent.start_char - 50)
            end = min(len(text), ent.end_char + 50)
            context = text[start:end].replace("\n", " ")
            date_entities.append((ent.text, context))
    
    return date_entities

def is_valid_date(date_str):
    """验证是否为合理日期（过滤明显错误）"""
    try:
        date = parse(date_str, fuzzy=True)
        # 假设财报年份在1990-当前年份之间
        return 2013 <= date.year <= datetime.now().year
    except:
        return False

def find_keyword_based_date(text):
    """基于关键词上下文的日期提取"""
    for context_group in KEYWORD_CONTEXTS:
        for keyword in context_group["keywords"]:
            # 构建动态正则表达式
            pattern = re.compile(
                fr"\b{keyword}\b.*?(\d{{4}}[-/_]\d{{1,2}}[-/_]\d{{1,2}}|\b(?:{'|'.join(DATE_PATTERNS)}))",
                re.IGNORECASE | re.DOTALL
            )
            match = pattern.search(text)
            if match:
                return match.group(1)
    return None

def prioritize_dates(candidates):
    """日期优先级排序"""
    valid_dates = []
    for date_str in candidates:
        if is_valid_date(date_str):
            try:
                dt = parse(date_str)
                valid_dates.append((dt, date_str))
            except:
                continue
    
    if not valid_dates:
        return None
    
    # 按以下优先级排序：
    # 1. 最接近文档开头
    # 2. 最近日期（如果是年报）
    # 3. 格式最明确的日期（包含月份名称的格式）
    sorted_dates = sorted(
        valid_dates,
        key=lambda x: (
            -x[0].year,  # 优先新年报
            "AMJFMASONDJ" in x[1].lower(),  # 包含月份名称的格式更可靠
            len(x[1])  # 长格式更可靠
        ),
        reverse=True
    )
    
    return sorted_dates[0][1]

def extract_report_date(text):
    # 策略1：关键词上下文匹配
    keyword_date = find_keyword_based_date(text)
    if keyword_date:
        return keyword_date
    
    # 策略2：spacy实体提取
    spacy_dates = extract_dates_with_context(text)
    if spacy_dates:
        # 提取前10%文本中的日期（假设日期在开头）
        early_text = text[:len(text)//10]
        early_dates = COMBINED_DATE_REGEX.findall(early_text)
        if early_dates:
            return prioritize_dates(early_dates)
        
        # 使用所有找到的日期
        all_dates = [date[0] for date in spacy_dates]
        return prioritize_dates(all_dates)
    
    # 策略3：正则全局匹配
    all_matches = COMBINED_DATE_REGEX.findall(text)
    if all_matches:
        return prioritize_dates(all_matches)
    
    return "NOT_FOUND"


def reports_data_extract(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []
    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue
        # 得到当前文件的路径
        file_path = os.path.join(input_folder, filename)
        # 读取txt文件到text中
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        date = extract_report_date(text)
        results.append({
            "filename": filename,
            "extracted_date": date,
            "validation_status": "待验证"  # 供人工检查
        })
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"处理完成！结果保存至 {output_csv}")
    print("建议执行以下操作：")
    print("1. 随机抽样检查结果准确性")
    print("2. 分析错误案例改进正则表达式")
    print("3. 对持续错误的格式添加定制规则")


current_script_path = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.join(current_script_path, '..', '..', 'data', 'processed', 'reports')
input_dir = os.path.join(base_dir, 'txtfile')
output_dir = os.path.join(base_dir, 'report_time')
# 使用示例
# process_reports_batch("txt_output")