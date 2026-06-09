from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# --- Path ke file .env ---
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')

class Settings(BaseSettings):
    """
    Membaca variabel lingkungan dari file .env
    """
    
    model_config = SettingsConfigDict(env_file=env_path, extra='ignore')

    # Variabel yang dibaca dari .env
    FRONTEND_URL: str
    OLLAMA_BASE_URL: str
    VISION_MODEL: str
    GENERATIVE_MODEL: str
    EMBEDDING_MODEL: str
    # Optional comma-separated list of fallback vision models (used when primary fails)
    VISION_FALLBACK_MODELS: str = "deepseek-v3.1:671b-cloud"
    # Number of times to retry the primary vision model if it returns 500 or transient errors
    VISION_MAX_RETRIES: int = 2
    # Seconds delay between retries of the primary vision model
    VISION_RETRY_DELAY: float = 1.0
    # Optional embedding model used specifically for reranking (calculate cosine similarity)
    RERANK_EMBEDDING_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    
    CHROMA_API_KEY: str
    CHROMA_TENANT: str
    CHROMA_DATABASE: str

    # --- BARU ---
    CHROMA_COLLECTION_NAME: str

# Membuat satu instance 'settings'
settings = Settings()