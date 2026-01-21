import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 文档加载
from langchain_community.document_loaders import TextLoader
# Embedding
from langchain_community.embeddings import HuggingFaceEmbeddings
# 向量库
from langchain_community.vectorstores import FAISS
# 文本切分器
from langchain_text_splitters import CharacterTextSplitter

from langchain_chroma import Chroma

# Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 短文本切分器
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=0
)

def build(role):
    # 加载文本
    loader = TextLoader(f"data/{role}.txt", encoding="utf-8")
    docs = loader.load()

    # 切分文本
    documents = splitter.split_documents(docs)

    # 打印调试
    print(f"🔹 {role} 文档条数: {len(documents)}")
    for i, d in enumerate(documents):
        print(f"文档{i}内容: {d.page_content}")

    # 构建 Chroma 向量库
    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=f"vector_db/{role}"
    )
    # db.persist()

    print(f"✅ {role} 知识库构建完成")

if __name__ == "__main__":
    build("HR")
    build("TECH")