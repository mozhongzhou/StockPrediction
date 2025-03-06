import torch
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import AutoTokenizer, AutoModel
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("yfinance imported successfully")
print("Transformers imported successfully")