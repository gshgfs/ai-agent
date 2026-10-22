import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME")


# 1. 我们自己写一个真实的工具函数（查天气）
def get_weather(city: str):
    # 真实情况下这里会去调天气接口，这里为了测试，我们写死
    if "杭州" in city:
        return "杭州今天天气晴朗，气温 25-30 度，适合穿短袖。"
    elif "北京" in city:
        return "北京今天暴雨，气温 15-20 度，记得带伞。"
    else:
        return f"抱歉，暂时没有 {city} 的天气数据。"


# 2. 用 JSON 格式向 AI 描述这个工具（相当于给 AI 递一张说明书）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：杭州、北京",
                    }
                },
                "required": ["city"],
            },
        }
    }
]


async def main():
    # 我们故意问一个 AI 不知道实时情况的问题
    user_question = "今天杭州天气怎么样？我应该穿什么衣服？"
    print(f"👤 用户: {user_question}\n")

    messages = [{"role": "user", "content": user_question}]

    # 第一次请求 AI：把问题和工具说明书一起发给它
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,  # 关键：把工具说明书传给 AI
        tool_choice="auto"  # 让 AI 自己决定用不用工具
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 检查 AI 是不是决定要调用工具
    if tool_calls:
        print("🤖 AI 决定调用工具！")
        # 把 AI 的决定加入记忆（这一步必须做，否则 AI 后面会失忆）
        messages.append(response_message)

        # 遍历 AI 决定调用的所有工具（可能一次调用多个）
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)  # 提取参数

            print(f"   -> 准备执行函数: {function_name}，参数: {function_args}")

            # 执行我们本地的真实函数
            if function_name == "get_weather":
                function_response = get_weather(city=function_args.get("city"))
                print(f"   -> 函数执行结果: {function_response}")

                # 把工具执行的结果，作为一个新的消息发给 AI
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

        # 第二次请求 AI：告诉它工具执行完了，让它根据结果回答用户
        print("\n🤖 AI 正在根据工具结果生成最终回答...")
        second_response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        print(f"✨ AI 最终回复: {second_response.choices[0].message.content}")
    else:
        # 如果 AI 觉得不需要用工具，直接回答
        print(f"✨ AI 直接回复: {response_message.content}")


if __name__ == "__main__":
    asyncio.run(main())







