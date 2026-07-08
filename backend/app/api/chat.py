from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import base64
import json
import asyncio
import os
from datetime import datetime
from typing import Optional, List

# Impor service kita
from app.services.rag_service import query_knowledge_base
from app.services.llm_service import llm_generative, llm_vision_json
from langchain_ollama import ChatOllama
from app.core.config import settings
from app.models.schema import VisionExtraction
from langchain_core.messages import HumanMessage
import logging

# rag_service.py — 2 baris ini
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# ganti jadi:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import numpy as np

class NumpySafeEncoder(json.JSONEncoder):
    """Jaring pengaman: encoder JSON yang aman untuk tipe numpy (float32, int64, ndarray)
    kalau-kalau ada nilai numpy lain yang lolos dari sumbernya."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
# --- KONFIGURASI LOGGING ---
CHAT_LOG_DIR = "logs/chat"
os.makedirs(CHAT_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(CHAT_LOG_DIR, "server.log")

logger = logging.getLogger("ai_chart.chat")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    # file
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

router = APIRouter()

# Helper untuk mengubah gambar menjadi base64
async def image_to_base64(file: UploadFile) -> str:
    """Convert uploaded image to base64 string"""
    contents = await file.read()
    # Reset file pointer untuk bisa dibaca lagi jika diperlukan
    await file.seek(0)
    return base64.b64encode(contents).decode('utf-8')

# Helper untuk memformat History menjadi teks yang bisa dibaca AI
def format_chat_history(history: List[dict]) -> str:
    """Format chat history into readable text for AI context"""
    if not history:
        return "Belum ada riwayat percakapan."
    
    formatted = ""
    for msg in history:
        role = "User" if msg.get("role") == "user" else "AI"
        content = msg.get("content", "")
        
        # Batasi panjang per pesan, tapi tetap simpan konteks penting
        if len(content) > 1000:
            # Ambil awal dan akhir untuk mempertahankan konteks
            truncated = content[:400] + "\n...[dipotong]...\n" + content[-400:]
            formatted += f"{role}: {truncated}\n\n"
        else:
            formatted += f"{role}: {content}\n\n"
    
    return formatted.strip()

def format_rag_context(documents: list) -> str:
    """Format RAG documents into readable context"""
    if not documents:
        return "Tidak ada konteks referensi yang ditemukan di database."
    
    context = "\n--- Konteks Referensi dari Knowledge Base ---\n"
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'Tidak diketahui')
        content = doc.page_content
        
        # Batasi panjang setiap dokumen RAG
        if len(content) > 500:
            content = content[:500] + "..."
        
        context += f"\nSumber {i}: {source}\n"
        context += f"{content}\n"
        context += "-" * 50 + "\n"
    
    return context

# --- PROMPT TEMPLATES ---

VISION_PROMPT_TEMPLATE = """
Anda adalah AI analis chart teknikal yang presisi. Tugas Anda adalah mengekstrak
informasi dari gambar chart ini HANYA dalam format JSON.
JANGAN berikan penjelasan apa pun di luar JSON.

Format JSON yang harus Anda keluarkan adalah:
{
    "key_patterns": ["Daftar pola chart yang terlihat, misal: Head and Shoulders, Double Top, dll"],
    "key_levels": ["Daftar level support/resistance utama dengan nilai angka jika terlihat"],
    "indicators": ["Daftar sinyal indikator yang terlihat (misal: RSI Overbought, MACD Bullish Cross)"],
    "summary": "Ringkasan singkat 1-2 kalimat dari apa yang Anda lihat di chart ini"
}

PENTING: Respons Anda HARUS berupa JSON valid, tidak boleh ada teks lain.
"""

SYNTHESIS_PROMPT_TEMPLATE = """
Anda adalah "AI Chart Analyst" - asisten edukasi trading yang ramah dan profesional.

PERINGATAN PENTING:
1. JANGAN PERNAH memberikan nasihat keuangan (seperti "Beli sekarang", "Jual di harga X").
2. Fokus pada EDUKASI: jelaskan pola, indikator, dan konsep trading.
3. Gunakan format Markdown agar rapi:
   - ## untuk judul utama
   - ### untuk sub-judul
   - **bold** untuk penekanan
   - * atau - untuk bullet points
4. Berikan penjelasan yang mudah dipahami pemula, tapi tetap akurat.

INFORMASI YANG TERSEDIA:

### 1. Riwayat Percakapan Sebelumnya:
{chat_history}

### 2. Analisis Gambar Chart BARU (Hasil Ekstraksi AI):
{vision_json}

### 3. Pertanyaan User SAAT INI:
"{user_query}"

### 4. Konteks dari Buku/Referensi Trading (RAG):
{rag_context}

TUGAS ANDA:
1. Jawab pertanyaan user berdasarkan chart BARU yang baru saja diupload
2. Gunakan informasi dari analisis vision AI di atas
3. Jika user merujuk ke percakapan lama, gunakan 'Riwayat Percakapan' untuk konteks
4. Referensi ke knowledge base jika relevan
5. Format jawaban dengan Markdown yang rapi
6. Tetap fokus pada EDUKASI, bukan rekomendasi trading

Berikan penjelasan yang komprehensif, terstruktur, dan mudah dipahami!
"""

TEXT_ONLY_SYNTHESIS_PROMPT_TEMPLATE = """
Anda adalah "AI Chart Analyst" - asisten edukasi trading yang ramah dan profesional.

PERINGATAN PENTING:
1. JANGAN PERNAH memberikan nasihat keuangan (seperti "Beli", "Jual", "Hold").
2. Fokus pada EDUKASI: jelaskan konsep, strategi, dan teori trading.
3. Gunakan format Markdown agar rapi:
   - ## untuk judul utama
   - ### untuk sub-judul
   - **bold** untuk penekanan
   - * atau - untuk bullet points
4. Berikan penjelasan yang mudah dipahami.

INFORMASI YANG TERSEDIA:

### 1. Riwayat Percakapan Sebelumnya:
{chat_history}

### 2. Pertanyaan User SAAT INI:
"{user_query}"

### 3. Konteks dari Buku/Referensi Trading (RAG):
{rag_context}

TUGAS ANDA:
1. Jawab pertanyaan user dengan jelas dan terstruktur
2. Jika user bertanya tentang gambar yang DIUPLOAD SEBELUMNYA, lihat informasi di 'Riwayat Percakapan'
3. Gunakan knowledge base untuk memperkaya jawaban
4. Format dengan Markdown yang rapi
5. Tetap fokus pada EDUKASI

Berikan penjelasan yang komprehensif dan mudah dipahami!
"""

# --- API ENDPOINT UTAMA ---

@router.post("/analyze_chart")
async def analyze_chart_endpoint(
    query: str = Form(...),
    history: str = Form("[]"),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Main endpoint untuk analisis chart dengan/tanpa gambar.
    
    Args:
        query: Pertanyaan user
        history: JSON string berisi riwayat chat
        image_file: File gambar chart (opsional)
    
    Returns:
        StreamingResponse dengan jawaban AI
    """
    
    # === DEBUGGING LOG ===
    logger.info("%s", "=" * 60)
    logger.info("📥 [REQUEST] Query (first 180 chars): %s", (query or "")[0:180])
    logger.info("📥 [REQUEST] Raw history size (chars): %s", len(history or ""))
    logger.info("📥 [REQUEST] Image present: %s", (image_file.filename if image_file else 'None'))
    logger.info("%s", "=" * 60)

    # Save a lightweight 'request received' JSON so we can inspect incoming fields fast
    try:
        request_short_log = {
            "timestamp": datetime.now().isoformat(),
            "query": (query or "")[:400],
            "history_raw_len": len(history or ""),
            "image_filename": image_file.filename if image_file else None
        }
        req_logfile = os.path.join(CHAT_LOG_DIR, f"request_received_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(req_logfile, "w", encoding="utf-8") as rf:
            json.dump(request_short_log, rf, ensure_ascii=False, indent=2)
        logger.debug("Saved short request receipt log to %s", req_logfile)
    except Exception:
        logger.exception("Failed to write request_receipt log")
    
    # === PARSING HISTORY ===
    chat_history_list = []
    try:
        parsed_history = json.loads(history)
        if isinstance(parsed_history, list):
            chat_history_list = parsed_history
            logger.info("✅ [HISTORY] Successfully parsed %s messages", len(chat_history_list))
        else:
            logger.warning("⚠️ [HISTORY] Invalid format, expected list, got %s", type(parsed_history))
    except json.JSONDecodeError as e:
        logger.error("❌ [HISTORY] JSON parse error: %s", e)
    except Exception as e:
        logger.error("❌ [HISTORY] Unexpected error: %s", e)
    
    chat_history_text = format_chat_history(chat_history_list)
    
    # === INISIALISASI LOG DATA ===
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": "",
        "query": query,
        "history_length": len(chat_history_list),
        "image_filename": image_file.filename if image_file else None,
        "ai_response": "",
        "stream_chunks": [],
        "vision_attempts": [],
        "error": None
    }

    # === STREAMING GENERATOR ===
    async def response_streamer(final_message: HumanMessage, log_data: dict):
        """Generator untuk streaming response dari LLM"""
        full_response = ""
        try:
            # If a vision model was used, inform the client up-front
            if log_data.get("vision_model_used"):
                header = f"🔎 Vision model used: {log_data['vision_model_used']}\n\n"
                yield header

            logger.info("🤖 [LLM] Starting stream...")
            chunk_index = 0
            async for chunk in llm_generative.astream([final_message]):
                content = chunk.content
                # Log every chunk both to logger and to the per-chat log_data
                logger.debug("🔸 [LLM CHUNK %s] %s", chunk_index, content[:300])
                log_data.setdefault('stream_chunks', []).append({
                    'index': chunk_index,
                    'text_preview': content[:500]
                })
                chunk_index += 1
                full_response += content
                yield content
            logger.info("✅ [LLM] Stream completed. Total length: %s chars", len(full_response))
            
        except Exception as e:
            error_msg = f"\n\n⚠️ **Maaf, terjadi kesalahan saat memproses respons.**\nDetail: {str(e)}"
            logger.exception("❌ [LLM] Streaming error: %s", e)
            full_response += error_msg
            yield error_msg
            log_data["error"] = str(e)
        
        # === SIMPAN LOG ===
        try:
            log_data["ai_response"] = full_response
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(CHAT_LOG_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, cls=NumpySafeEncoder)
            logger.info("💾 [LOG] Saved chat log to %s", filepath)
        except Exception as e:
            logger.error("⚠️ [LOG] Failed to save log: %s", e)

    # === MAIN LOGIC ===
    try:
        # --- MODE 1: MULTIMODAL (DENGAN GAMBAR BARU) ---
        if image_file:
            log_data["mode"] = "multimodal"
            logger.info("🖼️ [MODE] Multimodal (image: %s)", image_file.filename)
            
            # Convert image to base64
            try:
                base64_image = await image_to_base64(image_file)
                logger.info("✅ [IMAGE] Converted to base64 (length: %s chars)", len(base64_image))
            except Exception as e:
                logger.exception("❌ [IMAGE] Conversion failed: %s", e)
                raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
            
            image_message = {
                "type": "image_url",
                "image_url": f"data:{image_file.content_type};base64,{base64_image}"
            }

            # STEP 1: Vision Extraction
            logger.info("👁️ [VISION] Starting image analysis...")
            vision_message = HumanMessage(content=[
                {"type": "text", "text": VISION_PROMPT_TEMPLATE},
                image_message
            ])
            
            extracted_data_json = "{}"
            rag_keywords = query
            
            # Try primary vision model then fallback to alternate cloud-based models if parsing/validation fail
            def _make_rag_keywords(ed: VisionExtraction, q: str):
                try:
                    return " ".join(ed.key_patterns + ed.indicators + [q])
                except Exception:
                    return q

            def _extract_json_from_text(text: str):
                """Attempt to find a JSON object inside a chunk of text by finding the first '{' and last '}' and parsing that substring."""
                try:
                    if not text or '{' not in text:
                        return None
                    start = text.find('{')
                    end = text.rfind('}')
                    if start == -1 or end == -1 or end <= start:
                        return None
                    candidate = text[start:end+1]
                    return json.loads(candidate)
                except Exception:
                    return None

            primary_ok = False
            extracted_data = None

            primary_attempts = settings.VISION_MAX_RETRIES if getattr(settings, 'VISION_MAX_RETRIES', None) is not None else 1
            retry_delay = settings.VISION_RETRY_DELAY if getattr(settings, 'VISION_RETRY_DELAY', None) is not None else 1.0

            vision_response = None
            primary_model_name = settings.VISION_MODEL

            # Retry loop for primary model (handles transient 500s)
            for attempt in range(1, primary_attempts + 1):
                attempt_record = {
                    'model': primary_model_name,
                    'attempt': attempt,
                    'invocation_success': False,
                    'parse_success': False,
                    'content_preview': None,
                    'error': None
                }
                try:
                    logger.info("🔁 [VISION] Primary model (%s) attempt %s/%s", primary_model_name, attempt, primary_attempts)
                    vision_response = await llm_vision_json.ainvoke([vision_message])
                    attempt_record['invocation_success'] = True
                    attempt_record['content_preview'] = getattr(vision_response, 'content', '')[:1000]
                    log_data.setdefault('vision_attempts', []).append(attempt_record)
                    logger.info("✅ [VISION] Primary raw response (first 300 chars): %s", getattr(vision_response, 'content', '')[:300])
                    
                    # Try parsing and validation
                    try:
                        vision_json_obj = json.loads(vision_response.content)
                        extracted_data = VisionExtraction(**vision_json_obj)
                        log_data["vision_model_used"] = primary_model_name
                        extracted_data_json = json.dumps(extracted_data.dict(), indent=2, ensure_ascii=False)
                        rag_keywords = _make_rag_keywords(extracted_data, query)
                        logger.info("✅ [VISION] Extracted patterns: %s", extracted_data.key_patterns)
                        # mark last attempt parse success
                        if log_data.get('vision_attempts'):
                            log_data['vision_attempts'][-1]['parse_success'] = True
                        primary_ok = True
                        break
                    except Exception as inner_e:
                        logger.warning("⚠️ [VISION] Primary model produced unparsable/invalid JSON: %s", inner_e)
                        raw = getattr(vision_response, 'content', '')
                        logger.debug("⚠️ [VISION] Primary raw content: %s", raw)
                        # Try to extract a JSON object substring and parse that
                        try_obj = _extract_json_from_text(raw)
                        if try_obj:
                            try:
                                extracted_data = VisionExtraction(**try_obj)
                                extracted_data_json = json.dumps(extracted_data.dict(), indent=2, ensure_ascii=False)
                                rag_keywords = _make_rag_keywords(extracted_data, query)
                                logger.info("✅ [VISION] Successfully recovered JSON from text using substring parse")
                                log_data["vision_model_used"] = primary_model_name
                                if log_data.get('vision_attempts'):
                                    log_data['vision_attempts'][-1]['parse_success'] = True
                                primary_ok = True
                                break
                            except Exception as e2:
                                logger.warning("⚠️ [VISION] Recovered JSON failed validation: %s", e2)
                        else:
                            logger.debug("⚠️ [VISION] No JSON object found in primary response text")
                        
                        # If this was the last attempt, break
                        if attempt >= primary_attempts:
                            logger.error("❌ [VISION] Primary model failed after %s attempts", primary_attempts)
                            break
                        else:
                            logger.info("⏳ [VISION] Waiting %ss before retrying primary model", retry_delay)
                            await asyncio.sleep(retry_delay)
                            
                except Exception as e:
                    attempt_record['error'] = str(e)
                    log_data.setdefault('vision_attempts', []).append(attempt_record)
                    logger.warning("⚠️ [VISION] Primary model attempt %s failed: %s", attempt, e)
                    if attempt < primary_attempts:
                        logger.info("⏳ [VISION] Waiting %ss before retrying primary model", retry_delay)
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error("❌ [VISION] Primary model failed after %s attempts", primary_attempts)

            # If primary failed, try a set of fallback cloud models sequentially from settings
            if not primary_ok:
                fallback_models = []
                raw = getattr(settings, 'VISION_FALLBACK_MODELS', '')
                if raw:
                    fallback_models = [m.strip() for m in raw.split(',') if m.strip()]

                for idx, alt_model in enumerate(fallback_models, start=1):
                    attempt_record = {
                        'model': alt_model,
                        'attempt': idx,
                        'invocation_success': False,
                        'parse_success': False,
                        'content_preview': None,
                        'error': None
                    }
                    try:
                        logger.info("🔁 [VISION] Attempting fallback model: %s", alt_model)
                        alt_instance = ChatOllama(model=alt_model, base_url=settings.OLLAMA_BASE_URL, format='json', temperature=0.0)
                        alt_resp = await alt_instance.ainvoke([vision_message])
                        attempt_record['invocation_success'] = True
                        attempt_record['content_preview'] = getattr(alt_resp, 'content', '')[:1000]
                        log_data.setdefault('vision_attempts', []).append(attempt_record)
                        logger.info("🔁 [VISION] Fallback %s response preview: %s", alt_model, getattr(alt_resp, 'content', '')[:300])
                        try:
                            alt_obj = json.loads(alt_resp.content)
                            alt_data = VisionExtraction(**alt_obj)
                            extracted_data = alt_data
                            log_data["vision_model_used"] = alt_model
                            extracted_data_json = json.dumps(extracted_data.dict(), indent=2, ensure_ascii=False)
                            rag_keywords = _make_rag_keywords(extracted_data, query)
                            logger.info("✅ [VISION] Successful extraction with fallback model %s", alt_model)
                            # mark parse success on the last attempt record
                            if log_data.get('vision_attempts'):
                                log_data['vision_attempts'][-1]['parse_success'] = True
                            primary_ok = True
                            break
                        except Exception as exc_parse:
                            logger.warning("⚠️ [VISION] Fallback model %s produced invalid JSON or failed validation: %s", alt_model, exc_parse)
                            # mark parse error
                            if log_data.get('vision_attempts'):
                                log_data['vision_attempts'][-1]['error'] = str(exc_parse)
                            continue
                    except Exception as e:
                        attempt_record['error'] = str(e)
                        log_data.setdefault('vision_attempts', []).append(attempt_record)
                        logger.warning("⚠️ [VISION] Invocation with fallback %s failed: %s", alt_model, e)

            # If none worked, use the default failure message and keep rag_keywords as original query
            if not primary_ok:
                extracted_data_json = json.dumps({
                    'key_patterns': [],
                    'key_levels': [],
                    'indicators': [],
                    'summary': 'Gagal mengekstrak data dari gambar'
                }, indent=2)
                rag_keywords = query

            # STEP 2: RAG Query
            logger.info("📚 [RAG] Querying knowledge base with: %s", rag_keywords[:200])
            try:
                rag_docs = query_knowledge_base(rag_keywords, k=5)
                # Log rag results (make small preview for the log)
                rag_results_log = []
                for i, d in enumerate(rag_docs, start=1):
                    src = d.metadata.get('source', 'Unknown') if hasattr(d, 'metadata') else 'Unknown'
                    rerank_score = d.metadata.get('rerank_score') if hasattr(d, 'metadata') else None
                    snippet = d.page_content[:200] if hasattr(d, 'page_content') else ''
                    rag_results_log.append({'rank': i, 'source': src, 'rerank_score': rerank_score, 'snippet': snippet})
                    logger.info("📄 [RAG RESULT] Rank %s source=%s score=%s snippet=%s", i, src, rerank_score, snippet[:80])
                log_data['rag_results'] = rag_results_log
                rag_context = format_rag_context(rag_docs)
                logger.info("✅ [RAG] Found %s documents", len(rag_docs))
            except Exception as e:
                logger.exception("⚠️ [RAG] Query failed: %s", e)
                rag_context = "Tidak dapat mengakses knowledge base saat ini."

            # STEP 3: Synthesis
            logger.info("🔄 [SYNTHESIS] Building final prompt...")
            final_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                chat_history=chat_history_text,
                vision_json=extracted_data_json,
                user_query=query,
                rag_context=rag_context
            )
            
            final_message = HumanMessage(content=final_prompt)

            return StreamingResponse(
                response_streamer(final_message, log_data), 
                media_type="text/plain"
            )

        # --- MODE 2: TEXT ONLY (LANJUTAN CHAT) ---
        else:
            log_data["mode"] = "text_only"
            logger.info("💬 [MODE] Text only (continuation chat)")

            # STEP 1: RAG Query
            logger.info("📚 [RAG] Querying knowledge base with query: %s", (query or "")[0:150])
            try:
                rag_docs = query_knowledge_base(query, k=5)
                rag_context = format_rag_context(rag_docs)
                logger.info("✅ [RAG] Found %s documents", len(rag_docs))
            except Exception as e:
                logger.error("⚠️ [RAG] Query failed: %s", e)
                rag_context = "Tidak dapat mengakses knowledge base saat ini."
            
            # STEP 2: Synthesis
            logger.info("🔄 [SYNTHESIS] Building final prompt...")
            final_prompt = TEXT_ONLY_SYNTHESIS_PROMPT_TEMPLATE.format(
                chat_history=chat_history_text,
                user_query=query,
                rag_context=rag_context
            )
            
            final_message = HumanMessage(content=final_prompt)
            
            return StreamingResponse(
                response_streamer(final_message, log_data), 
                media_type="text/plain"
            )

    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.exception("❌ [ERROR] Unexpected error in endpoint: %s", e)
        log_data["error"] = str(e)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )