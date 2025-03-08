import os
import re
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

# 初始化OpenAI客户端（DeepSeek兼容API）
client = OpenAI(
    api_key='',  # 替换为实际API密钥
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # DeepSeek API端点
)

def extract_report_date(text, max_retries=3):
    """
    使用DeepSeek大模型提取财报发布日期
    返回格式：YYYY-MM-DD 或 "Not Found"
    
    参数：
    text (str): 财报文本内容
    max_retries (int): 最大重试次数
    
    返回：
    str: 提取到的日期或错误信息
    """
    # 系统提示词（中英双语增强准确性）
    system_prompt = """你是一个专业的财务文档分析助手。请执行以下操作：
1. 仔细阅读用户提供的英文财报文本
2. 准确识别官方发布日期，注意：
   - 优先查找标题附近的日期
   - 排除历史数据和表格中的日期
3. 输出格式必须为YYYY-MM-DD
4. 如果没有找到明确日期，返回"Not Found"

请严格遵循以下规则：
- 不要添加任何解释
- 不要返回多个日期
- 不要返回日期范围"""

    # 文本预处理（优化API调用效率）
    def preprocess_text(t):
        # 移除多余空白字符和换行符
        t = re.sub(r'\s+', ' ', t)
        # 截取前6000字符（平衡准确率和token消耗）
        return t[:6000]

    processed_text = preprocess_text(text)
    
    # 构建对话消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": processed_text}
    ]

    for attempt in range(max_retries):
        try:
            # 调用DeepSeek API
            response = client.chat.completions.create(
                model="ep-20250308095954-46dbs",  # 实际的ModelEndPointID
                messages=messages,
                temperature=0.1,        # 低随机性保证稳定性
                max_tokens=20,          # 限制输出长度
                timeout=15              # 超时时间（秒）
            )
            
            # 提取结果内容
            result = response.choices[0].message.content.strip()
            
            # 验证日期格式
            if validate_date_format(result):
                return result
            return "Not Found"
            
        except Exception as e:
            print(f"API请求失败（尝试 {attempt+1}/{max_retries}）: {str(e)}")
            time.sleep(2 ** attempt)  # 指数退避策略
    
    return "API Error"

def validate_date_format(date_str):
    """
    验证日期格式和合理性
    
    参数：
    date_str (str): 待验证的日期字符串
    
    返回：
    bool: 是否有效
    """
    try:
        # 严格匹配ISO 8601格式
        date = datetime.strptime(date_str, "%Y-%m-%d")
        # 验证年份范围（1990-当前年份）
        return 1990 <= date.year <= datetime.now().year
    except ValueError:
        return False

def reports_data_extract(input_folder, output_folder):
    """
    批量处理财报文件
    
    参数：
    input_folder (str): 输入文件夹路径（包含txt文件）
    output_folder (str): 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 设置输出CSV路径
    output_csv = os.path.join(output_folder, "report_dates.csv")
    results = []
    
    # 获取文件列表
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    # 处理文件（带进度条）
    with tqdm(total=len(files), desc="分析财报日期", unit="文件") as pbar:
        for filename in files:
            file_path = os.path.join(input_folder, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # 提取日期
                date = extract_report_date(text)
                print(f'file:{filename}, date:{date}')
                results.append({
                    "filename": filename,
                    "extracted_date": date,
                    # "validation_status": "待验证"
                })
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                results.append({
                    "filename": filename,
                    "extracted_date": "处理失败",
                    "validation_status": "错误"
                })
            
            pbar.update(1)
            time.sleep(1)  # 控制请求频率
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ 处理完成！结果已保存至: {output_csv}")
    print("后续建议操作：")
    print("1. 使用 pandas 查看结果示例:")
    print(f'   import pandas as pd\n   pd.read_csv("{output_csv}").sample(3)')
    print("2. 检查 'validation_status' 列为'错误'的记录")
    print("3. 人工抽检部分结果准确性")

# 主程序入口
if __name__ == "__main__":

    # # 在调用前添加测试
    # test_response = client.chat.completions.create(
    #     model="ep-20250308095954-46dbs",
    #     messages=[{"role": "user", "content": "今天的日期是？"}]
    # )
    # print("API连接测试结果:", test_response.choices[0].message.content)
    # 路径配置（根据项目结构调整）
    current_script_path = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.join(current_script_path, '..', '..', 'data', 'processed', 'reports')
    
    input_dir = os.path.join(base_dir, 'txt_only_4')    # 输入目录
    output_dir = os.path.join(base_dir, 'report_time')  # 输出目录
    
    # 验证输入路径
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 启动处理流程
    reports_data_extract(input_dir, output_dir)