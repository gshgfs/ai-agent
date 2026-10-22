print("正在读取文档...")

# 1. 读取 txt 文档内容（注意你的文件名可能是 test.txt.txt，如果是，请把下面这一行改成 test.txt.txt）
with open("test.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"读取成功！文档总共有 {len(text)} 个字符。\n")

# 2. 文本分块（Chunking）
chunk_size = 300   # 每块大概 300 个字符
overlap = 50       # 相邻两块重叠 50 个字符（防止句子被生硬切断）

chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

print(f"切分完成！一共切成了 {len(chunks)} 块。\n")

# 3. 看一眼切出来的成果
for i, chunk in enumerate(chunks[:3]):  # 只看前 3 块
    print(f"--- 第 {i+1} 块 ---")
    print(chunk)
    print("-" * 30)