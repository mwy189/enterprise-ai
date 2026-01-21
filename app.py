from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ========= DeepSeek 配置 =========
client = OpenAI(
    api_key="sk-ab0bce7a83084d0896814ac560eafa73",
    base_url="https://api.deepseek.com"
)

# ========= FastAPI =========
app = FastAPI()

# ========= 向量模型 =========
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ========= 请求结构 =========
class AskRequest(BaseModel):
    question: str
    role: str   # HR / TECH

# ========= Prompt 构造 =========
def build_prompt(context, question):
    return f"""
你是公司内部 AI 助手。

规则：
1. 只能根据【内部资料摘要】回答
2. 不允许编造
3. 无资料就回答：资料中未找到相关信息

【内部资料摘要】
{context}

【问题】
{question}
"""

# ========= API =========
@app.post("/ask")
def ask(req: AskRequest):
    db = FAISS.load_local(
        f"vector_db/{req.role}",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(req.question, k=3)
    context = "\n".join([d.page_content for d in docs])

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": build_prompt(context, req.question)}
        ]
    )

    return {"answer": response.choices[0].message.content}
