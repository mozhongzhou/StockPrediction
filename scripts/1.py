import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.current_device())  # 应输出 0（或可用 GPU 编号）
print(torch.cuda.get_device_name(0))  # 应输出 GPU 名称，例如 "NVIDIA GeForce RTX 3080"