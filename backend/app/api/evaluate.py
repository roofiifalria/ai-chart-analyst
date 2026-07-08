"""
app/api/evaluate.py

Endpoint API BARU (tidak mengubah endpoint /api/analyze_chart yang sudah ada)
untuk menjalankan evaluasi kuantitatif terhadap model-model Ollama yang
dipakai di AI Chart Analyst, menggunakan metrik Precision, Recall, F1-Score,
dan Accuracy.

Endpoint yang tersedia:
- POST /api/evaluate            -> jalankan evaluasi untuk 1 atau lebih kombinasi model
- GET  /api/evaluate/results     -> daftar hasil evaluasi yang tersimpan
- GET  /api/evaluate/results/{filename} -> ambil detail 1 hasil evaluasi tersimpan
- GET  /api/evaluate/dataset     -> lihat isi dataset ground truth yang sedang dipakai

Frontend TIDAK PERLU diubah untuk memakai endpoint ini; ini murni tambahan
untuk keperluan analisis/laporan (bisa dites lewat Swagger UI di /docs).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
import logging

from app.models.schema import EvaluateRequest
from evaluation.evaluation_service import (
    compare_models,
    list_saved_results,
    load_saved_result,
    DEFAULT_DATASET_PATH,
    _load_dataset,
)

logger = logging.getLogger("ai_chart.evaluate_api")

router = APIRouter()


@router.post("/evaluate")
async def evaluate_endpoint(request: EvaluateRequest):
    """
    Jalankan evaluasi Precision/Recall/F1-Score untuk satu atau beberapa
    kombinasi model Ollama (vision_model + generative_model), lalu kembalikan
    perbandingan performa dan model terbaik berdasarkan F1-Score keseluruhan.

    Jika 'model_combinations' kosong, otomatis menggunakan model default
    dari .env (VISION_MODEL & GENERATIVE_MODEL) sebagai satu-satunya kombinasi.

    CATATAN: proses ini menjalankan inferensi Ollama secara SYNCHRONOUS untuk
    setiap kasus x setiap kombinasi model, sehingga bisa memakan waktu cukup
    lama jika dataset atau jumlah kombinasi model besar.
    """
    dataset_path = request.dataset_path or DEFAULT_DATASET_PATH

    combos = request.model_combinations or []
    combos_as_dict = [
        {"vision_model": c.vision_model, "generative_model": c.generative_model}
        for c in combos
    ] or [{"vision_model": None, "generative_model": None}]  # fallback ke default .env

    try:
        # Validasi dataset ada isinya sebelum menjalankan proses yang berat
        _load_dataset(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        logger.info("🔬 [EVAL API] Memulai evaluasi untuk %s kombinasi model...", len(combos_as_dict))
        result = compare_models(combos_as_dict, dataset_path=dataset_path, save_result=True)
        logger.info("✅ [EVAL API] Evaluasi selesai. Model terbaik: %s", result.get("best_model"))
        return result
    except Exception as e:
        logger.exception("❌ [EVAL API] Evaluasi gagal: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluasi gagal: {str(e)}")


@router.get("/evaluate/results")
async def list_results_endpoint():
    """Daftar semua file hasil evaluasi yang pernah disimpan (terbaru dulu)."""
    try:
        files = list_saved_results()
        return {"count": len(files), "results": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/results/{filename}")
async def get_result_endpoint(filename: str):
    """
    Ambil isi lengkap 1 file hasil evaluasi tersimpan (untuk dilihat/diexport ke laporan).
    File .json dikembalikan sebagai JSON, file .txt dikembalikan sebagai teks polos
    (supaya bisa langsung dibaca rapi lewat browser/Swagger, bukan ter-escape sebagai string).
    """
    try:
        content = load_saved_result(filename)
        if filename.endswith(".txt"):
            return PlainTextResponse(content)
        return content
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluate/dataset")
async def get_dataset_endpoint():
    """Lihat isi dataset ground truth yang sedang dipakai untuk evaluasi."""
    try:
        cases = _load_dataset(DEFAULT_DATASET_PATH)
        return {"dataset_path": DEFAULT_DATASET_PATH, "num_cases": len(cases), "cases": cases}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))