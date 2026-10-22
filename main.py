import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import httpx
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
app = FastAPI()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME")

# ================= 1. 数据库配置 =================
SQLALCHEMY_DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)


Base.metadata.create_all(bind=engine)

# ================= 2. 向量库配置 =================
print("正在加载向量模型...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef
)


# ================= 3. 工具定义（真实高德天气） =================
async def get_weather(city: str):
    api_key = os.getenv("AMAP_API_KEY")
    if not api_key:
        return "未配置高德 API Key，无法查询天气。"

    async with httpx.AsyncClient() as http_client:
        try:
            # 1. 地理编码：城市名转 adcode
            geo_url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&key={api_key}"
            geo_res = await http_client.get(geo_url)
            geo_data = geo_res.json()

            if not geo_data.get("geocodes"):
                return f"找不到城市：{city}"

            adcode = geo_data["geocodes"][0]["adcode"]

            # 2. 查天气
            weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={api_key}"
            weather_res = await http_client.get(weather_url)
            weather_data = weather_res.json()

            if weather_data.get("lives"):
                live = weather_data["lives"][0]
                weather = live["weather"]
                temp = live["temperature"]
                wind = live["winddirection"] + "风"
                return f"{city}当前天气：{weather}，气温 {temp} 度，{wind}。"
            else:
                return f"抱歉，暂时查不到 {city} 的天气数据。"
        except Exception as e:
            return f"天气查询失败，错误信息：{str(e)}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如：杭州"}
                },
                "required": ["city"],
            },
        }
    }
]


class ChatRequest(BaseModel):
    session_id: str
    message: str


# ================= 4. 历史记录与话题管理接口 =================

# 获取历史话题列表（带标题）
@app.get("/api/sessions")
def get_sessions():
    db = SessionLocal()
    try:
        # 获取所有不重复的 session_id
        sessions = db.query(Message.session_id).distinct().all()
        result = []
        for (s_id,) in sessions:
            # 找这个会话里的第一句话作为标题
            first_msg = db.query(Message).filter(
                Message.session_id == s_id,
                Message.role == "user"
            ).order_by(Message.id).first()

            # 截取前 15 个字作为标题，如果没有就用“新对话”
            title = (first_msg.content[:15] + "...") if first_msg else "空白对话"
            result.append({"session_id": s_id, "title": title})

        # 按照会话 id 倒序（通常 id 包含时间戳或字母，新会话排在前面）
        result.reverse()
        return result
    finally:
        db.close()


# 获取某个话题的具体聊天记录
@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    db = SessionLocal()
    try:
        history = db.query(Message).filter(Message.session_id == session_id).order_by(Message.id).all()
        return [{"role": msg.role, "content": msg.content} for msg in history]
    finally:
        db.close()


# 删除特定的历史话题
@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    db = SessionLocal()
    try:
        db.query(Message).filter(Message.session_id == session_id).delete()
        db.commit()
        return {"message": "话题已删除"}
    finally:
        db.close()


# ================= 5. 终极聊天接口 =================
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        db = SessionLocal()
        try:
            # 查历史
            history = db.query(Message).filter(Message.session_id == req.session_id).order_by(Message.id).all()

            # 查向量库 (RAG)
            results = collection.query(query_texts=[req.message], n_results=3)
            context = "\n".join(results["documents"][0]) if results["documents"] and results["documents"][0] else ""

            system_prompt = "你是一个专属知识库助手。请优先根据参考资料回答。如果不知道，请说不知道。"
            if context:
                system_prompt += f"\n\n【参考资料】\n{context}"

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": req.message})

            db.add(Message(session_id=req.session_id, role="user", content=req.message))
            db.commit()

            # 试探工具调用（非流式）
            check_response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = check_response.choices[0].message

            if response_message.tool_calls:
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "get_weather":
                        # 注意这里加了 await，因为真实的查天气是异步请求
                        function_response = await get_weather(city=function_args.get("city"))
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        })

                stream_response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True
                )
            else:
                stream_response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True
                )

            full_answer = ""
            async for chunk in stream_response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer += delta
                    yield delta

            db.add(Message(session_id=req.session_id, role="assistant", content=full_answer))
            db.commit()

        finally:
            db.close()

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/webui", response_class=HTMLResponse)
def web_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>颜月卿的项目</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f3f4f6; color: #333; height: 100vh; display: flex; overflow: hidden; }

            /* ===== 左侧侧边栏 ===== */
            #sidebar { width: 260px; background-color: #1e293b; color: white; display: flex; flex-direction: column; }
            .sidebar-header { padding: 15px; }
            .new-chat-btn { width: 100%; background: #2563eb; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 15px; transition: 0.2s;}
            .new-chat-btn:hover { background: #1d4ed8; }

            #topic-list { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 5px; }
            .topic-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 10px; border-radius: 8px; cursor: pointer; transition: 0.2s; color: #cbd5e1; font-size: 14px; }
            .topic-item:hover { background: #334155; color: white; }
            .topic-item.active { background: #2563eb; color: white; }

            .delete-btn { background: transparent; border: none; color: #ef4444; font-size: 16px; cursor: pointer; display: none; }
            .topic-item:hover .delete-btn { display: block; }
            .delete-btn:hover { color: #f87171; }

            /* ===== 右侧聊天主区 ===== */
            #main { flex: 1; display: flex; flex-direction: column; }
            header { background: white; padding: 15px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; font-size: 20px; font-weight: bold; color: #2563eb; z-index: 10; }

            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; align-items: center; }
            #chat-box { width: 100%; max-width: 800px; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 80%; padding: 12px 18px; border-radius: 12px; line-height: 1.6; font-size: 15px; word-wrap: break-word;}
            .user-msg { align-self: flex-end; background: #2563eb; color: white; border-bottom-right-radius: 2px; }
            .ai-msg { align-self: flex-start; background: white; color: #333; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e5e7eb; }
            .ai-msg::before { content: "🤖"; margin-right: 8px; }

            footer { background: white; padding: 20px; border-top: 1px solid #e5e7eb; display: flex; justify-content: center; }
            .input-wrapper { width: 100%; max-width: 800px; display: flex; gap: 10px; background: #f9fafb; padding: 10px; border-radius: 12px; border: 1px solid #d1d5db; align-items: center; }
            .input-wrapper:focus-within { border-color: #2563eb; }
            input[type="text"] { flex: 1; border: none; background: transparent; outline: none; font-size: 16px; padding: 8px; color: #333; }
            .send-btn { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; }
        </style>
    </head>
    <body>
        <!-- 侧边栏 -->
        <div id="sidebar">
            <div class="sidebar-header">
                <button class="new-chat-btn" onclick="startNewChat()">+ 新建对话</button>
            </div>
            <div id="topic-list"></div>
        </div>

        <!-- 聊天主区 -->
        <div id="main">
            <header>颜月卿的项目</header>
            <div id="chat-container">
                <div id="chat-box"></div>
            </div>
            <footer>
                <div class="input-wrapper">
                    <input type="text" id="user-input" placeholder="输入你的问题..." onkeypress="if(event.key === 'Enter') sendMessage()">
                    <button class="send-btn" id="send-btn" onclick="sendMessage()">发送</button>
                </div>
            </footer>
        </div>

        <script>
            let currentSessionId = null; // 当前正在聊天的 session_id
            let isChatting = false;
            const chatBox = document.getElementById('chat-box');
            const chatContainer = document.getElementById('chat-container');
            const topicList = document.getElementById('topic-list');

            // 页面加载完，先去拉取话题列表
            window.onload = () => {
                loadTopics();
                startNewChat(); // 默认开一个新的空白对话
            };

            // 加载左侧历史话题列表
            async function loadTopics() {
                const res = await fetch('/api/sessions');
                const topics = await res.json();
                topicList.innerHTML = '';
                topics.forEach(t => {
                    const div = document.createElement('div');
                    div.className = `topic-item ${currentSessionId === t.session_id ? 'active' : ''}`;
                    // 点击加载记录，点击垃圾桶删除
                    div.innerHTML = `
                        <span onclick="loadHistory('${t.session_id}')" style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${t.title}</span>
                        <button class="delete-btn" onclick="deleteTopic(event, '${t.session_id}')">🗑️</button>
                    `;
                    topicList.appendChild(div);
                });
            }

            // 新建对话（生成随机 session_id）
            function startNewChat() {
                // 用随机数生成一个独特的话题 ID
                currentSessionId = "topic_" + Date.now();
                chatBox.innerHTML = '<div class="message ai-msg">这是一个全新的对话，开始提问吧！</div>';
                document.querySelectorAll('.topic-item').forEach(el => el.classList.remove('active'));
            }

            // 加载历史记录
            async function loadHistory(sessionId) {
                if(isChatting) return; // 正在生成回复时不允许切走
                currentSessionId = sessionId;
                loadTopics(); // 刷新高亮状态
                chatBox.innerHTML = ''; 

                const res = await fetch(`/api/history/${sessionId}`);
                const history = await res.json();

                history.forEach(msg => {
                    const div = document.createElement('div');
                    div.className = `message ${msg.role === 'user' ? 'user-msg' : 'ai-msg'}`;
                    div.innerText = msg.content;
                    chatBox.appendChild(div);
                });
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            // 删除特定话题
            async function deleteTopic(event, sessionId) {
                event.stopPropagation();
                if (!confirm('确定要删除这个历史话题吗？')) return;

                await fetch(`/api/session/${sessionId}`, { method: 'DELETE' });

                if (currentSessionId === sessionId) startNewChat(); // 如果删的是当前正在看的，就新建一个
                loadTopics(); // 刷新列表
            }

            // 发送消息
            async function sendMessage() {
                if (isChatting) return;
                const inputElement = document.getElementById('user-input');
                const message = inputElement.value.trim();
                if (!message) return;

                const sendBtn = document.getElementById('send-btn');

                // 如果还没有 session_id，生成一个
                if (!currentSessionId) startNewChat();

                // 画出用户气泡
                const userBubble = document.createElement('div');
                userBubble.className = 'message user-msg';
                userBubble.innerText = message;
                chatBox.appendChild(userBubble);

                // 画出 AI 气泡（空的）
                const aiBubble = document.createElement('div');
                aiBubble.className = 'message ai-msg';
                aiBubble.innerText = '';
                chatBox.appendChild(aiBubble);

                chatContainer.scrollTop = chatContainer.scrollHeight;
                inputElement.value = '';
                isChatting = true;
                sendBtn.disabled = true;

                try {
                    const response = await fetch('/chat/stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: currentSessionId, message: message })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder('utf-8');

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        aiBubble.innerText += decoder.decode(value, { stream: true });
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }

                    // 回答完毕后，刷新左侧话题列表（这样如果这是一个新对话，左边就会出现它的标题）
                    loadTopics();

                } catch (error) {
                    aiBubble.innerText += `\\n[请求出错: ${error}]`;
                }

                isChatting = false;
                sendBtn.disabled = false;
                inputElement.focus();
            }
        </script>
    </body>
    </html>
    """
    return html_content