import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import chromadb
from chromadb.utils import embedding_functions

print("正在读取 test.txt 的最新内容...")
with open("test.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 自动分块（不需要手动复制粘贴了）
chunk_size = 300
overlap = 50
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunks.append(text[i:i + chunk_size])

print(f"读取成功！自动切分为 {len(chunks)} 块。")
print("正在连接本地向量模型...")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)
client = chromadb.PersistentClient(path="./chroma_db")


try:
    client.delete_collection("knowledge_base")
except:
    pass

collection = client.create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef
)

print("正在把最新文字变成向量，并存入数据库...")
collection.add(
    documents=chunks,
    ids=[f"id_{i}" for i in range(len(chunks))]
)

print(f"✅ 成功更新！现在数据库里有 {collection.count()} 块新数据了！")