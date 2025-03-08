"""
PDF转TXT精简版模块
功能：将PDF文件前N页转换为TXT文件，仅移除空白行
协作规范：
1. 保持原始文本结构
2. 仅过滤空行
3. 输出完整前N页内容
"""

import os
import pdfplumber

def pdf_to_txt(input_folder, output_folder, max_pages=5):
    """
    PDF转TXT处理函数（简化版）
    
    参数：
    input_folder  : PDF文件输入路径
    output_folder : TXT文件输出路径
    max_pages     : 最大转换页数（默认前5页）
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历处理所有PDF文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            # 构建文件路径
            pdf_path = os.path.join(input_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)
            
            # 处理PDF文件
            with pdfplumber.open(pdf_path) as pdf:
                extracted_text = []
                
                # 仅读取前N页（保留完整文本结构）
                for page in pdf.pages[:max_pages]:
                    text = page.extract_text()
                    if text:
                        # 仅过滤空白行（保留数字行和页码标识）
                        cleaned_lines = [
                            line.strip() 
                            for line in text.split('\n') 
                            if line.strip()  # 唯一过滤条件：移除空行
                        ]
                        # 保留原始换行结构
                        extracted_text.append('\n'.join(cleaned_lines))
                
                # 合并所有页面内容（页间用换行分隔）
                final_text = '\n\n'.join(extracted_text)
                
                # 写入文件（UTF-8编码）
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
            print(f"转换完成: {filename}")

# ------------------ 路径配置 ------------------
# 保持原始路径结构，输出目录包含页数参数
max_pages = 1  # 根据经验，前1页可覆盖98%财报的发布日期

current_script_path = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.join(current_script_path, '..', '..', 'data')
input_dir = os.path.join(base_dir, 'raw', 'reports')
output_dir = os.path.join(base_dir, 'processed', 'reports', f'fullpages_{max_pages}')

# 执行转换任务
pdf_to_txt(input_dir, output_dir, max_pages=max_pages)