import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# .env dosyasını yükle
load_dotenv()

# PDF yolu
pdf_path = r"C:\Users\bahaa\OneDrive\Masaüstü\gemini_app_2\backend\rag_docs\akreditasyon-calistayi-sonuc-bildirgesi.pdf"

# PDF yükle
pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
pages_split = text_splitter.split_documents(pages)

# Embedding modeli
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# MongoDB bağlantısı
MONGO_URI = os.getenv("MONGO_URI")  # Atlas ya da local connection string
DB_NAME = os.getenv("DB_NAME", "rag")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "stock_market")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Eğer collection boşsa dolduralım
vectorstore = MongoDBAtlasVectorSearch.from_documents(
    documents=pages_split,
    embedding=embedding_model,
    collection=collection,
    index_name="vector_index"  # MongoDB Atlas’ta oluşturduğun index adıyla aynı olmalı
)

print("✅ MongoDB VectorStore oluşturuldu ve PDF içeriği eklendi.")
