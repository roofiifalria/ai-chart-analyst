from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
import uvicorn

# --- IMPOR ROUTER BARU ---
from app.api import chat as chat_router

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="AI Chart Analyst Backend",
    version="1.0"
)

# --- PENGATURAN CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint sederhana untuk mengecek apakah server berjalan
@app.get("/", tags=["Status"])
def read_root():
    """
    Endpoint root untuk cek status server.
    """
    return {"status": "ok", "message": "AI Chart Analyst Backend is running!"}


# --- ENDPOINT TES RAG (HAPUS ATAU KOMENTARI) ---
# @app.get("/test_rag", tags=["Test"])
# def test_rag_endpoint(query: str = "pola bendera"):
#     ... (Kode lama dihapus) ...
#     return ...


# --- TAMBAHKAN API ROUTER UTAMA KITA ---
app.include_router(chat_router.router, prefix="/api", tags=["Analysis"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)