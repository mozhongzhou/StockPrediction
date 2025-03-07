"""
PDF财报关键信息提取模块
功能：将PDF财报转换为精简版TXT文件，聚焦报告发布日期相关信息
协作规范：
1. 输出文件体积需控制在10KB以下
2. 必须保留可能包含日期的上下文信息
3. 兜底策略需保留至少10行原始文本
"""

import os
import pdfplumber
import re

def pdf_to_txt(input_folder, output_folder, max_pages=5):
    """
    将PDF文件转换为精简版TXT文件
    
    参数：
    input_folder  : 输入文件夹路径，包含待处理的PDF文件
    output_folder : 输出文件夹路径，用于保存处理后的TXT文件
    max_pages     : 最大读取页数，默认前5页（根据财报特征优化为4页）
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入目录下的所有PDF文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            # 构建完整文件路径
            pdf_path = os.path.join(input_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)
            
            # PDF文件处理
            with pdfplumber.open(pdf_path) as pdf:
                extracted_texts = []
                
                # 第一层过滤：仅读取前N页（根据max_pages参数控制）
                for page in pdf.pages[:max_pages]:
                    text = page.extract_text()
                    if text:
                        # 基础文本清洗（单页级别处理）
                        lines = [
                            line.strip() for line in text.split('\n')
                            if line.strip()  # 移除空行（减少30%体积）
                            and not line.strip().isdigit()  # 过滤纯数字行（页码）
                            and 'page' not in line.lower()  # 过滤包含页码说明的行
                        ]
                        # 重组处理后的文本（保留原始行结构）
                        processed_text = '\n'.join(lines)
                        extracted_texts.append(processed_text)
                
                # 合并多页文本（页间用换行分隔）
                full_text = '\n'.join(extracted_texts)
                
                # 第二层过滤：基于关键字的智能筛选
                filtered_text = []
                # 日期特征正则表达式（匹配包含日期关键词的整行）
                # 匹配模式说明：
                # 1. 包含日期关键词：reported, date, filed 等
                # 2. 包含4位数字的年份标识
                date_pattern = re.compile(
                    r'(\b(?:reported?|date|filed|as of|quarterly|annual|published)\b.*?\b\d{4}\b)', 
                    re.IGNORECASE  # 不区分大小写
                )
                
                # 逐行扫描筛选
                for line in full_text.split('\n'):
                    if date_pattern.search(line):
                        filtered_text.append(line)  # 保留匹配行
                
                # 兜底策略：如果未找到日期特征，保留前10行文本
                # （考虑不同财报封面格式差异，10行可覆盖大部分情况）
                if not filtered_text:
                    filtered_text = full_text.split('\n')[:10]
                
                # 生成最终文本内容
                final_text = '\n'.join(filtered_text)
                
                # 写入输出文件（UTF-8编码确保特殊字符正确处理）
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
            print(f"转换完成: {filename}")

# ------------------ 路径配置 ------------------ 
# 重要：路径结构遵循项目规范
# base_dir -> raw/reports 存放原始PDF
# base_dir -> processed/reports/txt_only_N 存放处理结果

# 设置最大读取页数（根据经验4页可覆盖99%情况）
max_pages = 4

# 动态构建输出目录（包含页数参数，便于结果版本管理）
current_script_path = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.join(current_script_path, '..', '..', 'data')  # 项目数据根目录
input_dir = os.path.join(base_dir, 'raw', 'reports')  # 原始PDF存放路径
output_dir = os.path.join(base_dir, 'processed', 'reports', f'txt_only_{max_pages}')  # 输出路径包含页数参数

# 执行转换任务
pdf_to_txt(input_dir, output_dir, max_pages=max_pages)