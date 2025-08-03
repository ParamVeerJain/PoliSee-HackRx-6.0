#utils.vector_store_loader.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

DB_PATH = "vector_store/faiss_index"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
