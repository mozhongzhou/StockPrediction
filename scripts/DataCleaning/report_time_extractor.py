import os
import re
import requests
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm

# DeepSeek API配置
DEEPSEEK_API_KEY = "你的API密钥"  # 替换为实际密钥
API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

def extract_report_date(text, max_retries=3):
    """
    使用DeepSeek API提取财报发布日期
    返回格式：YYYY-MM-DD 或 "Not Found"
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Accept": "application/json"
    }

    # 优化后的系统提示词
    system_prompt = """你是一个专业的财务文档分析助手。请执行以下操作：
1. 在用户提供的文本中准确识别公司财报的官方发布日期
2. 日期必须满足：
   - 来自报告标题或发布声明
   - 排除历史数据日期、表格日期等无关日期
3. 输出格式严格遵守YYYY-MM-DD，如未找到返回"Not Found"
不要任何解释，只需返回日期或"Not Found"
"""

    # 文本预处理函数
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # 合并空白字符
        return text[:8000]  # 控制token用量

    processed_text = preprocess_text(text)

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": processed_text}
        ],
        "temperature": 0.1,
        "max_tokens": 20
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()['choices'][0]['message']['content'].strip()
            
            # 验证日期格式
            if validate_date_format(result):
                return result
            return "Invalid Format"
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # 速率限制处理
                wait_time = int(response.headers.get('Retry-After', 30))
                print(f"达到速率限制，等待 {wait_time}秒...")
                time.sleep(wait_time)
            else:
                print(f"API错误: {e}")
                time.sleep(2)
        except Exception as e:
            print(f"其他错误: {e}")
            time.sleep(1)
    
    return "API Error"

def validate_date_format(date_str):
    """验证日期格式和合理性"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return 1990 <= date.year <= datetime.now().year
    except:
        return False

def reports_data_extract(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 设置输出CSV路径
    output_csv = os.path.join(output_folder, "report_dates.csv")
    results = []
    
    # 获取文件列表
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    # 添加进度条
    with tqdm(total=len(files), desc="Processing Reports") as pbar:
        for filename in files:
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            date = extract_report_date(text)
            results.append({
                "filename": filename,
                "extracted_date": date,
                "validation_status": "待验证"
            })
            
            pbar.update(1)
            time.sleep(1)  # 控制请求频率

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n处理完成！结果保存至 {output_csv}")
    print("建议执行以下操作：")
    print("1. 随机抽样检查结果准确性")
    print("2. 分析错误案例改进流程")
    print("3. 使用以下代码查看示例：")
    print("   import pandas as pd")
    print(f'   df = pd.read_csv("{output_csv}")')
    print("   print(df.sample(3))")

# 路径配置
if __name__ == "__main__":
    current_script_path = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.join(current_script_path, '..', '..', 'data', 'processed', 'reports')
    
    input_dir = os.path.join(base_dir, 'txtfile')
    output_dir = os.path.join(base_dir, 'report_time')
    
    # 验证输入路径是否存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    reports_data_extract(input_dir, output_dir)