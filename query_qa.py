import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# DeepSeek 客户端（保持全局，不用每次都创建）
client = OpenAI(
    api_key="sk-ab0bce7a83084d0896814ac560eafa73",
    base_url="https://api.deepseek.com"
)

# query 函数
def query(role, question):
    # 加载向量库
    db = Chroma(
        persist_directory=f"vector_db/{role}",
        embedding_function=embeddings
    )

    # 相似度检索
    docs = db.similarity_search(question, k=3)
    print(f"🔹 检索到 {len(docs)} 条文档")
    for i, d in enumerate(docs):
        print(f"文档{i}内容: {d.page_content}")
    context = "\n".join([d.page_content for d in docs])

    # Prompt
    messages = [
        {"role": "system", "content": """
你是公司内部制度问答助手。
你【只能】根据提供的资料回答问题。
如果资料中没有明确说明，请直接回答：资料中未提及，无法确认。
严禁补充、推测、扩展任何未在资料中出现的内容。
回答要简洁、准确，不使用通用人力资源常识。
"""},

        {"role": "user", "content":f"""
【公司制度资料】
{context}

【问题】
{question}

【回答要求】
- 仅使用资料中的信息
- 不允许引入外部常识
- 未提及内容请明确说明未提及
"""}
    ]

    # 调用 DeepSeek
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    role = "HR"
    print("企业制度问答助手已启动，输入 exit 退出")
    while True:
        question = input("\n请输入问题：")
        if question.lower() in ["exit", "quit"]:
            print("退出问答助手")
            break
        answer = query(role, question)
        print("\n🔹 回答：")
        print(answer)
