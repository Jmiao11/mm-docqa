# emb_probe.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
Q_PREFIX = "为这个句子生成表示以用于检索相关文章："

query = "向量数据库有什么用"
passages = [
    "向量数据库用来存储和检索高维向量，做近似最近邻搜索，找出语义最相近的文档块。",  # 相关
    "今天天气晴朗，适合出门散步，公园里樱花开得正好。",                            # 不相关
]

def embed(texts):
    return model.encode(texts, normalize_embeddings=True)

def cos(a, b):                       # 归一化后 cosine == 点积
    return float(np.dot(a, b))

q_plain = embed([query])[0]              # query 不加前缀
q_inst = embed([Q_PREFIX + query])[0]   # query 加检索前缀
P = embed(passages)                      # passage 一律不加前缀

print("== query 不加前缀 ==")
for i, p in enumerate(P):
    print(f"  passage[{i}] cos = {cos(q_plain, p):.4f}")
print("== query 加检索前缀 ==")
for i, p in enumerate(P):
    print(f"  passage[{i}] cos = {cos(q_inst, p):.4f}")

print("\n模长 |q_inst| =", round(float(np.linalg.norm(q_inst)), 4))  # 应≈1.0