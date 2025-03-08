# 项目架构

```
StockPrediction/
├── data/ # 存放原始和处理后的数据
│ ├── raw/ # 原始数据（如 K 线、10-K 财报）
│ └── processed/ # 处理后的数据（如序列、嵌入）
├── models/ # 存放模型代码和权重
├── scripts/ # 数据处理和训练脚本
│ ├── data_acquisition/ # 数据获取阶段
│ │ ├── fetch_kline.py # 获取 K 线数据
│ │ ├── fetch_10k.py # 获取 10-K 财报
│ │ └── utils/ # 通用工具
│ │ ├── api_client.py # API 调用工具
│ │ └── download_helper.py # 下载辅助函数
│ ├── data_cleaning/ # 数据清洗阶段
│ │ ├── clean_kline.py # 清洗 K 线数据
│ │ ├── clean_10k.py # 清洗 10-K 财报（如重命名）
│ │ └── utils/ # 清洗工具
│ │ ├── date_parser.py # 日期解析工具
│ │ └── text_extractor.py # 文本提取工具
│ ├── data_alignment/ # 多模态数据对齐阶段
│ │ ├── align_kline_10k.py # 对齐 K 线和财报数据
│ │ └── utils/ # 对齐工具
│ │ ├── time_sync.py # 时间同步函数
│ │ └── format_converter.py # 数据格式转换
│ ├── labeling/ # 打标签阶段
│ │ ├── label_generator.py # 生成标签
│ │ └── utils/ # 标签工具
│ │ ├── rule_engine.py # 规则引擎
│ │ └── label_validator.py # 标签验证
│ ├── training/ # 训练 Transformer 阶段
│ │ ├── train_transformer.py # 训练主脚本
│ │ ├── model_config.yaml # 模型配置文件
│ │ └── utils/ # 训练工具
│ │ ├── data_loader.py # 数据加载
│ │ └── metrics.py # 评估指标
│ └── config/ # 配置文件
│ ├── api_keys.yaml # API 密钥
│ ├── proxy_settings.yaml # 代理配置
│ └── cleaning_rules.yaml # 清洗规则
├── outputs/ # 存放结果（图表、预测、日志）
├── requirements.txt # 依赖包列表
├── main.py # 主入口脚本（可选）
└── README.md # 项目说明
```
