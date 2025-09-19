import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import glob
from langchain_huggingface import HuggingFaceEmbeddings
# .env dosyasını yükleu
load_dotenv()

##enpoint olarak ytü,boun,itü olabilir
######
docs_path = r"C:\Users\bahaa\OneDrive\Masaüstü\gemini_app_2\backend\rag_docs\iüc"
files = glob.glob(os.path.join(docs_path, "*"))


# Tüm dokümanları yükle
all_docs = []
for f in files:
    file_path = os.path.join(docs_path, f)
    if f.endswith(".pdf"):
        print(f"📄 PDF dosyası yükleniyor: {f}")
        loader = PyPDFLoader(file_path)
    elif f.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        print(f"📄 DOCX dosyası yükleniyor: {f}")
    elif f.endswith(".doc"):
        #loader = UnstructuredWordDocumentLoader(file_path)
        print(f"📄 DOC dosyası Yüklenemedi")
    elif f.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        print(f"📄 TXT dosyası yükleniyor: {f}")
    else:
        print(f"⚠️ Desteklenmeyen dosya formatı: {f}")
        continue
    
    docs = loader.load()
    all_docs.extend(docs)

print(f"📄 Toplam {len(all_docs)} doküman yüklendi.")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
docs_split = text_splitter.split_documents(all_docs)

# Embedding modeli
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# MongoDB bağlantısı
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "rag")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "iuc")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# VectorStore oluşturma (yeni eklemeler için append olacak)
vectorstore = MongoDBAtlasVectorSearch.from_documents(
    documents=docs_split,
    embedding=embedding_model,
    collection=collection,
    index_name="iuc_search"
)

print("✅ MongoDB VectorStore güncellendi, tüm dosyalar eklendi.")
