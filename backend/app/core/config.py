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
    
    CHROMA_API_KEY: str
    CHROMA_TENANT: str
    CHROMA_DATABASE: str

    # --- BARU ---
    CHROMA_COLLECTION_NAME: str

# Membuat satu instance 'settings'
settings = Settings()