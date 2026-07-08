import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.core.config import settings
from flashrank import Ranker, RerankRequest # <-- 1. IMPOR FLASHRANK

# --- INISIALISASI (Dijalankan sekali saat startup) ---

print("RAG Service: Memuat model embedding...")
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("RAG Service: Model embedding dimuat.")

print("RAG Service: Menghubungkan ke ChromaDB Cloud...")
client = chromadb.CloudClient(
    api_key=settings.CHROMA_API_KEY,
    tenant=settings.CHROMA_TENANT,
    database=settings.CHROMA_DATABASE
)
print("RAG Service: Terhubung ke ChromaDB Cloud.")

vector_store = Chroma(
    client=client,
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embeddings
)

# --- OPTIONAL: Rerank embedding model (used to compute embedding-based score for reranked passages)
print("RAG Service: Memuat rerank embedding (opsional)...")
try:
    rerank_embeddings = HuggingFaceEmbeddings(
        model_name=getattr(settings, 'RERANK_EMBEDDING_MODEL', settings.EMBEDDING_MODEL),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("RAG Service: Rerank embedding dimuat.")
except Exception as e:
    rerank_embeddings = None
    print(f"RAG Service: Gagal memuat rerank embedding: {e}")

# --- 2. INISIALISASI RERANKER ---
# Kita gunakan model 'ms-marco-MiniLM-L-12-v2' (default) yang ringan dan akurat
print("RAG Service: Memuat model Reranker (FlashRank)...")
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./model_cache")
print("RAG Service: Model Reranker siap.")

# --- FUNGSI QUERY (DIPERBARUI) ---

def query_knowledge_base(query_text: str, k: int = 3) -> list[Document]:
    """
    Melakukan 2-Stage Retrieval:
    1. Similarity Search (ambil kandidat banyak, misal 15)
    2. Reranking (ambil k terbaik, misal 3)
    """
    print(f"RAG Service: Menerima query: '{query_text}'")
    
    try:
        # TAHAP 1: Similarity Search (Ambil lebih banyak kandidat)
        initial_k = 30 
        candidates = vector_store.similarity_search(query_text, k=initial_k)
        
        if not candidates:
            return []
            
        print(f"RAG Service: Menemukan {len(candidates)} kandidat awal. Melakukan Reranking...")

        # TAHAP 2: Reranking dengan FlashRank
        # FlashRank butuh format list of dict
        passages = [
            {
                "id": str(i), 
                "text": doc.page_content, 
                "meta": doc.metadata
            } 
            for i, doc in enumerate(candidates)
        ]

        rerank_request = RerankRequest(query=query_text, passages=passages)
        results = ranker.rerank(rerank_request)

        # Ambil Top-K hasil reranking
        top_results = results[:k]
        
        # --- [TAMBAHAN BARU] PRINT SKOR KE TERMINAL & embed similarity ---
        print(f"\n--- HASIL RANKING (Top {k}) ---")
        # If rerank_embeddings available, build embeddings for query and top passages
        query_embed = None
        if rerank_embeddings is not None:
            try:
                query_embed = rerank_embeddings.embed_query(query_text)
            except Exception as ee:
                print(f"RAG Service: Gagal membuat embedding query untuk rerank: {ee}")

        for idx, res in enumerate(top_results):
            src = res['meta'].get('source', 'Unknown')
            score = res.get('score', 0.0)
            txt = res['text']
            # Compute embedding similarity if possible
            embed_sim = None
            if rerank_embeddings is not None and query_embed is not None:
                try:
                    pass_embed = rerank_embeddings.embed_documents([txt])[0]
                    # cosine similarity
                    import math
                    def cos_sim(a, b):
                        if not a or not b:
                            return 0.0
                        dot = sum(x*y for x,y in zip(a,b))
                        na = math.sqrt(sum(x*x for x in a))
                        nb = math.sqrt(sum(x*x for x in b))
                        if na == 0 or nb == 0:
                            return 0.0
                        return dot/(na*nb)
                    embed_sim = cos_sim(query_embed, pass_embed)
                except Exception:
                    embed_sim = None

            print(f"Rank {idx+1}: Skor {score:.4f} | Source: {src} | EmbedSim: {embed_sim if embed_sim is not None else 'N/A'}")
            print(f"       Teks: {txt[:160]}...")
        print("----------------------------------\n")
        # ---------------------------------------------

        # Kembalikan ke format Document LangChain
        # SESUDAH
        final_docs = []
        for res in top_results:
            meta = res['meta'].copy() if isinstance(res['meta'], dict) else {}
            raw_score = res.get('score')
            # FlashRank mengembalikan numpy.float32 — konversi ke Python float
            # native di sini supaya semua konsumen di hilir (chat.py, evaluation) aman
            meta['rerank_score'] = float(raw_score) if raw_score is not None else None
            final_docs.append(Document(
                page_content=res['text'], 
                metadata=meta
            ))

        return final_docs

    except Exception as e:
        print(f"RAG Service: Error saat query/rerank: {e}")
        # Fallback: jika rerank gagal, coba return hasil pencarian biasa (opsional)
        return []