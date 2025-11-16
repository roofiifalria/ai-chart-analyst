import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.core.config import settings

# --- IMPOR BARU YANG HILANG ---
from langchain_community.vectorstores import Chroma
# --- ---------------------- ---

# --- INISIALISASI SAAT SERVER STARTUP ---
# Kita memuat objek-objek berat ini satu kali saja saat server dimulai,
# bukan setiap kali ada request. Ini menghemat banyak waktu.

print("RAG Service: Memuat model embedding...")
# 1. Muat model embedding (sama seperti di notebook)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("RAG Service: Model embedding dimuat.")

print("RAG Service: Menghubungkan ke ChromaDB Cloud...")
# 2. Inisialisasi Klien ChromaDB Cloud
client = chromadb.CloudClient(
    api_key=settings.CHROMA_API_KEY,
    tenant=settings.CHROMA_TENANT,
    database=settings.CHROMA_DATABASE
)
print("RAG Service: Terhubung ke ChromaDB Cloud.")

# 3. Inisialisasi LangChain Vector Store
# Ini adalah objek yang akan kita gunakan untuk melakukan pencarian
# Baris ini sekarang akan berhasil karena 'Chroma' sudah diimpor
vector_store = Chroma(
    client=client,
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embeddings
)

print("RAG Service: Siap menerima query.")
# --- SELESAI INISIALISASI ---


def query_knowledge_base(query_text: str, k: int = 2) -> list[Document]:
    """
    Melakukan similarity search ke ChromaDB Cloud.
    """
    print(f"RAG Service: Menerima query: '{query_text}'")
    try:
        results = vector_store.similarity_search(query_text, k=k)
        print(f"RAG Service: Menemukan {len(results)} dokumen.")
        return results
    except Exception as e:
        print(f"RAG Service: Error saat query: {e}")
        return []