# preprocess_report.py
# 预处理财报数据，清洗和嵌入成模型输入
# 读取原始财报文本，清洗标点和停用词，用 FinBert 提取嵌入，存成 .npy 文件
# 支持 NVDA、TSLA、GOOGL、AAPL、MSFT 五支股票，加分割线分开输出

# 导入必要的库
import os  # 用于处理文件路径和目录操作，比如创建文件夹、拼接路径
import numpy as np  # 用于处理数组和保存数据，财报的嵌入会存成 .npy 文件
from transformers import AutoTokenizer, AutoModel  # 从 Hugging Face 导入 FinBert 的工具，分词器和模型
import torch  # PyTorch 框架，用于运行 FinBert 模型，处理张量计算
import spacy  # NLP 库，用于清洗财报文本（去标点、停用词等）

# 加载 FinBert 和 spaCy 模型
# FinBert 是专门为金融领域微调的 BERT，能理解财报里的术语和情绪
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")  # 分词器，把文本转成数字 token
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")  # 模型，把 token 转成 768 维嵌入
nlp = spacy.load("en_core_web_sm")  # spaCy 的英文模型，清洗文本用，"sm" 是小模型，速度快

# 设置输入输出路径
# 用相对路径，确保代码在不同机器上都能跑
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 项目根目录
input_dir = os.path.join(base_dir, "data", "raw", "reports")  # 原始财报 .txt 文件的目录
output_dir = os.path.join(base_dir, "data", "processed", "reports")  # 处理后的嵌入 .npy 文件的目录
os.makedirs(output_dir, exist_ok=True)  # 如果 output_dir 不存在就创建，exist_ok=True 避免重复创建报错
print(f"Reading from: {input_dir}")  # 打印输入路径，方便调试
print(f"Saving to: {output_dir}")  # 打印输出路径，确认保存位置

# 定义要处理的股票列表
# 包括已有 NVDA、TSLA、GOOGL、AAPL，加上新加的 MSFT
stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# 定义预处理函数，处理每支股票的年报
def preprocess_report(stock):
    embeddings = []  # 存每年的嵌入向量，最后凑成一个大数组
    # 循环处理 2013-2024 年的 10-K 财报（12 年）
    for year in range(2013, 2025):
        # 拼接年报文件路径，比如 "MSFT_10-K_2013.txt"
        file_path = os.path.join(input_dir, f"{stock}_10-K_{year}.txt")
        
        # 检查文件是否存在，不存在就跳过
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping")  # 提醒用户缺文件，继续跑其他年份
            continue
        
        # 读取财报文本
        # 用 UTF-8 编码打开，兼容特殊字符（财报可能有表格符号等）
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()  # 读全部内容，可能几十万字
        print(f"Loaded {stock}_10-K_{year}, text length: {len(text)} chars")  # 调试：看文本多长
        
        # 用 spaCy 清洗文本
        # 财报太长，FinBert 最多吃 512 个 token，截取前 10 万字符
        doc = nlp(text[:100000])  # 前 10 万字符（约 2 万词），抓摘要和关键部分
        # 去标点和停用词，只留核心词，比如 "revenue growth 2023"
        cleaned_text = " ".join(token.text for token in doc if not token.is_punct and not token.is_stop)
        print(f"Cleaned {stock}_10-K_{year}, cleaned length: {len(cleaned_text)} chars")  # 调试：清洗后多长

        # 用 FinBert 分词和提取嵌入
        # 转成 token IDs，限制 512 token，多截少补
        inputs = tokenizer(cleaned_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        # 用模型提取特征
        with torch.no_grad():  # 不算梯度，节省内存，只推理
            outputs = model(**inputs)  # 输出隐藏状态
            # 取 [CLS] 向量，代表整个财报的核心信息
            embedding = outputs.last_hidden_state[:, 0, :].numpy()  # 形状 (1, 768)
        
        # 加到嵌入列表
        embeddings.append(embedding)
        print(f"Processed {stock}_10-K_{year}")  # 提示处理完成

    # 保存所有年份的嵌入
    if embeddings:  # 有数据才保存，避免空列表
        output_file = os.path.join(output_dir, f"{stock}_embeddings.npy")
        embeddings_array = np.array(embeddings)  # 转成数组，形状 (年份数, 1, 768)
        np.save(output_file, embeddings_array)
        print(f"Saved {stock} embeddings: {len(embeddings)} reports, shape: {embeddings_array.shape}")
    else:
        print(f"No embeddings saved for {stock}, check input files!")
    print(f"----- Finished {stock} -----\n")  # 加结束分隔线

# 主程序：循环处理每支股票
for stock in stocks:
    print(f"----- Starting preprocessing for {stock} -----\n")  # 加开始分隔线
    preprocess_report(stock)