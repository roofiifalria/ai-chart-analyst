from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import base64
import json
import asyncio

# Impor semua service kita
from app.services.rag_service import query_knowledge_base
from app.services.llm_service import llm_generative, llm_vision_json
from app.models.schema import VisionExtraction

# LangChain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

router = APIRouter()

# Helper untuk mengubah gambar menjadi base64 (dibutuhkan oleh model Visi)
async def image_to_base64(file: UploadFile) -> str:
    contents = await file.read()
    return base64.b64encode(contents).decode('utf-8')

# Helper untuk menggabungkan potongan teks dari RAG
def format_rag_context(documents: list) -> str:
    context = ""
    for i, doc in enumerate(documents):
        context += f"--- Konteks Referensi {i+1} ---\n"
        context += f"Sumber: {doc.metadata.get('source', 'Tidak diketahui')}\n"
        context += f"Teks: {doc.page_content}\n"
        context += "---------------------------------\n\n"
    return context

# --- PROMPT TEMPLATES ---

# Prompt untuk Langkah A (Ekstraksi Visi ke JSON)
VISION_PROMPT_TEMPLATE = """
Anda adalah AI analis chart teknikal yang presisi. Tugas Anda adalah mengekstrak
informasi dari gambar chart ini HANYA dalam format JSON.
JANGAN berikan penjelasan apa pun di luar JSON.

Format JSON yang harus Anda keluarkan adalah:
{
    "key_patterns": ["Daftar pola chart yang terlihat"],
    "key_levels": ["Daftar level support/resistance utama"],
    "indicators": ["Daftar sinyal indikator yang terlihat (misal: RSI, MACD)"],
    "summary": "Ringkasan singkat 1 kalimat dari apa yang Anda lihat"
}

Gambar Chart:
[GAMBAR]
"""

# Prompt untuk Langkah C (Sintesis Akhir)
SYNTHESIS_PROMPT_TEMPLATE = """
Anda adalah "AI Chart Analyst", seorang analis teknikal senior dan seorang edukator.
Tugas Anda adalah memberikan analisis lengkap berdasarkan informasi yang diberikan.

PERINGATAN PENTING: JANGAN PERNAH memberikan nasihat keuangan. JANGAN PERNAH
mengatakan "Beli", "Jual", "Stop Loss", atau "Take Profit".
Fokus HANYA pada edukasi dan analisis objektif.

Anda memiliki 4 sumber informasi:
1.  **Gambar Chart Asli**: [GAMBAR]
2.  **Pertanyaan Pengguna**: {user_query}
3.  **Temuan Awal AI Visi (JSON)**: {vision_json}
4.  **Konteks dari Buku Teks (RAG)**: {rag_context}

Gunakan semua informasi ini untuk menyusun jawaban yang mendalam, jelas, dan edukatif.
Jawab pertanyaan pengguna, jelaskan apa arti temuan visi (JSON), dan
gunakan konteks RAG untuk menjelaskan teori di baliknya.

Analisis Anda:
"""

# --- API ENDPOINT UTAMA ---

@router.post("/analyze_chart")
async def analyze_chart_endpoint(
    query: str = Form(...),
    image_file: UploadFile = File(...)
):
    """
    Endpoint utama untuk menganalisis chart.
    Ini menjalankan alur Orkestrasi 3 langkah (A-B-C).
    """
    try:
        # --- LANGKAH AWAL: Siapkan data ---
        # 1. Ubah gambar ke Base64
        base64_image = await image_to_base64(image_file)
        
        # 2. Buat pesan gambar untuk LangChain
        image_message = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        }

        # --- LANGKAH A: EKSTRAKSI VISI (Panggilan AI #1) ---
        print("Langkah A: Memulai Ekstraksi Visi...")
        vision_prompt = VISION_PROMPT_TEMPLATE.replace("[GAMBAR]", "")
        
        vision_message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                image_message
            ]
        )
        
        # Panggil model llm_vision_json
        vision_response = await llm_vision_json.ainvoke([vision_message])
        
        # Ambil dan parse output JSON
        vision_json_output = json.loads(vision_response.content)
        
        # Validasi dengan Pydantic (opsional tapi bagus)
        try:
            extracted_data = VisionExtraction(**vision_json_output)
            print("Langkah A: Ekstraksi Visi Sukses.")
        except Exception as pydantic_error:
            print(f"Langkah A: Gagal validasi Pydantic: {pydantic_error}")
            raise HTTPException(status_code=500, detail=f"Gagal memvalidasi output visi: {pydantic_error}")

        
        # --- LANGKAH B: RETRIEVAL (Panggilan RAG) ---
        print("Langkah B: Memulai Retrieval RAG...")
        # Buat query RAG dari kata kunci yang diekstrak
        rag_query_keywords = " ".join(
            extracted_data.key_patterns + extracted_data.key_levels + extracted_data.indicators
        )
        
        # Panggil service RAG kita
        rag_documents = query_knowledge_base(rag_query_keywords, k=2)
        rag_context = format_rag_context(rag_documents)
        print("Langkah B: Retrieval RAG Sukses.")


        # --- LANGKAH C: SINTESIS AKHIR (Panggilan AI #2 - STREAMING) ---
        print("Langkah C: Memulai Sintesis Akhir...")
        
        # Siapkan Final Prompt
        final_prompt_text = SYNTHESIS_PROMPT_TEMPLATE.format(
            user_query=query,
            vision_json=json.dumps(extracted_data.dict(), indent=2),
            rag_context=rag_context
        ).replace("[GAMBAR]", "")
        
        final_message = HumanMessage(
            content=[
                {"type": "text", "text": final_prompt_text},
                image_message # Kita kirim gambar lagi untuk referensi akhir
            ]
        )

        # Fungsi generator untuk streaming
        async def response_streamer():
            # Gunakan .astream() untuk respons streaming
            async for chunk in llm_generative.astream([final_message]):
                # chunk.content berisi potongan teks
                yield chunk.content
            
            print("Langkah C: Streaming Selesai.")

        # Kembalikan respons streaming
        return StreamingResponse(response_streamer(), media_type="text/plain")

    except Exception as e:
        print(f"Error di /analyze_chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))