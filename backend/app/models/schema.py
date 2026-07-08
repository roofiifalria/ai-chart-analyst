from pydantic import BaseModel
from typing import List, Optional

# Ini adalah struktur JSON yang kita PAKSA 
# model visi untuk mengeluarkannya.
class VisionExtraction(BaseModel):
    key_patterns: List[str]
    key_levels: List[str]
    indicators: List[str]
    summary: str
    
    class Config:
        # Contoh data untuk dokumentasi API
        json_schema_extra = {
            "example": {
                "key_patterns": ["Head and Shoulders", "Bearish Flag"],
                "key_levels": ["Support at 45000", "Resistance at 48000"],
                "indicators": ["RSI Oversold", "MACD Bearish Cross"],
                "summary": "Harga menunjukkan pola reversal bearish setelah gagal menembus resisten."
            }
        }

# --- SKEMA UNTUK MODUL EVALUASI (Precision / Recall / F1-Score) ---

class ModelCombination(BaseModel):
    """Satu kombinasi model Ollama (vision + generative) yang akan dievaluasi."""
    vision_model: Optional[str] = None
    generative_model: Optional[str] = None


class EvaluateRequest(BaseModel):
    """
    Body untuk POST /api/evaluate.

    Contoh:
    {
        "model_combinations": [
            {"vision_model": "qwen2.5vl:7b", "generative_model": "llama3.1:8b"},
            {"vision_model": "llava:13b", "generative_model": "mistral:7b"}
        ]
    }

    Jika model_combinations dikosongkan, evaluasi akan otomatis memakai
    VISION_MODEL dan GENERATIVE_MODEL yang ada di .env (1 kombinasi saja).
    """
    model_combinations: List[ModelCombination] = []
    dataset_path: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model_combinations": [
                    {"vision_model": "qwen2.5vl:7b", "generative_model": "llama3.1:8b"},
                    {"vision_model": "llava:13b", "generative_model": "mistral:7b"}
                ]
            }
        }