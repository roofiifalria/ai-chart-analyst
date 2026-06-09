from langchain_community.chat_models import ChatOllama
from app.core.config import settings

# File ini menginisialisasi dan mengekspor model LLM kita.
# Kita memuatnya sekali di sini agar server tidak perlu memuat ulang
# model untuk setiap request.

print("LLM Service: Menghubungkan ke Ollama...")

# 1. LLM Generatif (Untuk berpikir dan menjawab)
# Ini adalah model standar untuk sintesis akhir.
# HAPUS parameter mirostat/tfs_z jika ada untuk menghilangkan warning
llm_generative = ChatOllama(
    model=settings.GENERATIVE_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.2,  
)

# 2. LLM Visi (Untuk ekstraksi JSON)
# Ini adalah model YANG SAMA, tapi kita memaksanya 
# untuk HANYA merespons dalam format JSON.
llm_vision_json = ChatOllama(
    model=settings.VISION_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    format="json",
    temperature=0.0   
)

print("LLM Service: Terhubung.")
print(f"  - Model Generatif: {settings.GENERATIVE_MODEL}")
print(f"  - Model Visi JSON: {settings.VISION_MODEL}")