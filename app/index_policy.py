from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
PDF_PATH = "data/Bajaj Allianz.pdf"
DB_PATH = "vector_store/faiss_index"

def create_index():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print("Embedding chunks...")
    embeddings =  HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Saving FAISS index...")
    vectorstore.save_local(DB_PATH)
    print(f"Index created at: {DB_PATH}")

if __name__ == "__main__":
    create_index()
