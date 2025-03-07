import numpy as np
labels = np.load("E:/Coding/NLP-Project/StockPrediction/data/processed/multimodal/NVDA_labels.npy", allow_pickle=True)
print(labels.shape, labels[:5])