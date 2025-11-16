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

# (Kita bisa tambahkan skema lain di sini nanti)