# backend/app/api/chat.py (VERSI DIPERBARUI)

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import base64
import json
import asyncio
from typing import Optional # <-- 1. IMPOR 'Optional'

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
    if not documents:
        return "Tidak ada konteks referensi yang ditemukan di database."
    
    context = "\n--- Konteks Referensi ---\n"
    for i, doc in enumerate(documents):
        context += f"Sumber {i+1}: {doc.metadata.get('source', 'Tidak diketahui')}\n"
        context += f"Teks: {doc.page_content}\n"
        context += "---------------------------\n"
    return context

# --- PROMPT TEMPLATES (DIPERBARUI) ---

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
""" # Catatan: [GAMBAR] dihapus, akan ditambahkan oleh LangChain

# Prompt untuk Langkah C (Sintesis Akhir - GAMBAR + TEKS)
# 2. PROMPT DIPERBARUI UNTUK MENGHAPUS MARKDOWN
SYNTHESIS_PROMPT_TEMPLATE = """
Anda adalah "AI Chart Analyst", seorang analis teknikal senior dan edukator.
Tugas Anda adalah memberikan analisis lengkap berdasarkan informasi yang diberikan.

PERINGATAN PENTING: 
1. JANGAN PERNAH memberikan nasihat keuangan. JANGAN PERNAH mengatakan "Beli", "Jual", atau "Stop Loss". Fokus HANYA pada edukasi.
2. JAWAB HANYA DALAM TEKS BIASA (PLAIN TEXT).
3. JANGAN GUNAKAN MARKDOWN (seperti '###', '**', '---', atau '*').
4. Jawab dalam format paragraf yang mengalir dan mudah dibaca.

Anda memiliki 3 sumber informasi:
1.  **Pertanyaan Pengguna**: {user_query}
2.  **Temuan Awal AI Visi (JSON)**: {vision_json}
3.  **Konteks dari Buku Teks (RAG)**: {rag_context}

Tugas Anda:
1. Jawab langsung pertanyaan pengguna.
2. Jelaskan apa arti temuan visi (JSON) dalam konteks chart.
3. Gunakan konteks RAG untuk menjelaskan teori di balik temuan tersebut.

Analisis Anda (dalam Bahasa Indonesia, plain text, tanpa markdown):
"""

# 3. PROMPT BARU UNTUK KONDISI TEKS-SAJA
TEXT_ONLY_SYNTHESIS_PROMPT_TEMPLATE = """
Anda adalah "AI Chart Analyst", seorang analis teknikal dan edukator.
Tugas Anda adalah menjawab pertanyaan pengguna tentang konsep analisis teknikal.

PERINGATAN PENTING: 
1. JANGAN PERNAH memberikan nasihat keuangan.
2. JAWAB HANYA DALAM TEKS BIASA (PLAIN TEXT).
3. JANGAN GUNAKAN MARKDOWN (seperti '###', '**', '---', atau '*').
4. Jawab dalam format paragraf yang mengalir dan mudah dibaca.

Anda memiliki 2 sumber informasi:
1.  **Pertanyaan Pengguna**: {user_query}
2.  **Konteks dari Buku Teks (RAG)**: {rag_context}

Tugas Anda:
Gunakan konteks RAG untuk menjawab pertanyaan pengguna secara jelas dan edukatif.
Jika konteks RAG tidak relevan, jawab pertanyaan berdasarkan pengetahuan umum Anda tentang trading.

Jawaban Anda (dalam Bahasa Indonesia, plain text, tanpa markdown):
"""

# --- API ENDPOINT UTAMA (DIPERBARUI) ---

@router.post("/analyze_chart")
async def analyze_chart_endpoint(
    # 1. 'image_file' sekarang Opsional
    query: str = Form(...),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Endpoint utama untuk menganalisis chart (multimodal) ATAU
    menjawab pertanyaan teks-saja (text-only).
    """
    
    # Fungsi generator untuk streaming
    async def response_streamer(final_message: HumanMessage):
        async for chunk in llm_generative.astream([final_message]):
            yield chunk.content
        print("Langkah C: Streaming Selesai.")

    try:
        # 3. LOGIKA PERCABANGAN (JIKA ADA GAMBAR)
        if image_file:
            print(f"Menerima request MULTIMODAL (Teks + Gambar: {image_file.filename})")
            
            # --- LANGKAH AWAL: Siapkan data ---
            base64_image = await image_to_base64(image_file)
            image_message = {
                "type": "image_url",
                "image_url": f"data:{image_file.content_type};base64,{base64_image}"
            }

            # --- LANGKAH A: EKSTRAKSI VISI (Panggilan AI #1) ---
            print("Langkah A: Memulai Ekstraksi Visi...")
            vision_message = HumanMessage(
                content=[
                    {"type": "text", "text": VISION_PROMPT_TEMPLATE},
                    image_message
                ]
            )
            
            vision_response = await llm_vision_json.ainvoke([vision_message])
            vision_json_output = json.loads(vision_response.content)
            
            try:
                extracted_data = VisionExtraction(**vision_json_output)
                print("Langkah A: Ekstraksi Visi Sukses.")
            except Exception as pydantic_error:
                print(f"Langkah A: Gagal validasi Pydantic: {pydantic_error}")
                # Kita tidak menghentikan proses, kita coba lanjutkan tanpa JSON yang rapi
                extracted_data_json = vision_response.content
                rag_query_keywords = query # Fallback ke query pengguna
            else:
                extracted_data_json = json.dumps(extracted_data.dict(), indent=2)
                rag_query_keywords = " ".join(
                    extracted_data.key_patterns + extracted_data.key_levels + extracted_data.indicators
                )

            
            # --- LANGKAH B: RETRIEVAL (Panggilan RAG) ---
            print(f"Langkah B: Memulai Retrieval RAG dengan keywords: {rag_query_keywords}")
            rag_documents = query_knowledge_base(rag_query_keywords, k=2)
            rag_context = format_rag_context(rag_documents)
            print("Langkah B: Retrieval RAG Sukses.")


            # --- LANGKAH C: SINTESIS AKHIR (Panggilan AI #2 - STREAMING) ---
            print("Langkah C: Memulai Sintesis Akhir (Multimodal)...")
            final_prompt_text = SYNTHESIS_PROMPT_TEMPLATE.format(
                user_query=query,
                vision_json=extracted_data_json,
                rag_context=rag_context
            )
            
            final_message = HumanMessage(
                content=[
                    {"type": "text", "text": final_prompt_text},
                    image_message # Kita kirim gambar lagi untuk referensi akhir
                ]
            )

            return StreamingResponse(response_streamer(final_message), media_type="text/plain")

        # 3. LOGIKA PERCABANGAN (JIKA HANYA TEKS)
        else:
            print(f"Menerima request TEKS-SAJA (Query: {query})")
            
            # --- LANGKAH A (TEKS): (Dilewati) ---

            # --- LANGKAH B (TEKS): RETRIEVAL (Panggilan RAG) ---
            print(f"Langkah B (Teks): Memulai Retrieval RAG dengan query: {query}")
            rag_documents = query_knowledge_base(query, k=2)
            rag_context = format_rag_context(rag_documents)
            print("Langkah B (Teks): Retrieval RAG Sukses.")
            
            # --- LANGKAH C (TEKS): SINTESIS AKHIR (Panggilan AI #2 - STREAMING) ---
            print("Langkah C (Teks): Memulai Sintesis Akhir...")
            final_prompt_text = TEXT_ONLY_SYNTHESIS_PROMPT_TEMPLATE.format(
                user_query=query,
                rag_context=rag_context
            )
            
            final_message = HumanMessage(content=final_prompt_text) # Tidak ada gambar
            
            return StreamingResponse(response_streamer(final_message), media_type="text/plain")

    except Exception as e:
        print(f"Error di /analyze_chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))