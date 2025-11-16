from langchain_community.chat_models import ChatOllama
from app.core.config import settings

# File ini menginisialisasi dan mengekspor model LLM kita.
# Kita memuatnya sekali di sini agar server tidak perlu memuat ulang
# model untuk setiap request.

print("LLM Service: Menghubungkan ke Ollama...")

# 1. LLM Generatif (Untuk berpikir dan menjawab)
# Ini adalah model standar untuk sintesis akhir.
llm_generative = ChatOllama(
    model=settings.GENERATIVE_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.2,  # Sedikit kreatif, tapi tetap faktual
)

# 2. LLM Visi (Untuk ekstraksi JSON)
# Ini adalah model YANG SAMA, tapi kita memaksanya 
# untuk HANYA merespons dalam format JSON.
llm_vision_json = ChatOllama(
    model=settings.VISION_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    format="json",  # <-- Kuncinya ada di sini
    temperature=0.0   # Harus sangat kaku dan faktual
)

print("LLM Service: Terhubung.")
print(f"  - Model Generatif: {settings.GENERATIVE_MODEL}")
print(f"  - Model Visi JSON: {settings.VISION_MODEL}")