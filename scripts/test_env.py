import torch
print(torch.__version__)  # 打印PyTorch版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.version.cuda)  # 打印PyTorch支持的CUDA版本