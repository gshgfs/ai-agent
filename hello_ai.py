import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL = os.getenv("MODEL_NAME")

messages = [
    {"role": "system", "content": "你是一个简洁有用的 AI 助手。"}
]

print("开始聊天，输入 exit 退出。\n")

while True:
    user_input = input("你：")

    if user_input.strip().lower() == "exit":
        print("再见！")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )

    print("AI：", end="")
    full_answer = ""

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_answer += delta

    print("\n")
    messages.append({"role": "assistant", "content": full_answer})