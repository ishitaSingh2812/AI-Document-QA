from fastapi import FastAPI

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Load PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Free embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store embeddings in ChromaDB
db = Chroma.from_documents(chunks, embeddings)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

@app.get("/")
def home():
    return {"message": "AI Document QA Running"}

@app.get("/ask")
def ask(question: str):

    docs = retriever.get_relevant_documents(question)

    answer = ""

    for doc in docs:
        answer += doc.page_content + "\n"

    return {"answer": answer}