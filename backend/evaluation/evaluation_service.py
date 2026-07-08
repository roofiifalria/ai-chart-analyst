"""
evaluation/evaluation_service.py

Service utama untuk menjalankan evaluasi model Ollama (vision & generative)
terhadap dataset ground truth, lalu menghitung Precision / Recall / F1-Score
serta Accuracy, dan membandingkan beberapa model sekaligus.

ALUR EVALUASI PER KASUS (per baris di ground_truth.json):
1. Jika ada image_path -> jalankan Vision Model untuk ekstraksi JSON
   (key_patterns, key_levels, indicators), lalu bandingkan dengan
   expected_key_patterns / expected_key_levels / expected_indicators.
2. Jalankan RAG (opsional, dipakai supaya mendekati kondisi produksi) +
   Generative Model untuk menghasilkan jawaban akhir, lalu bandingkan
   dengan expected_answer_keywords untuk menilai relevansi jawaban.
3. Gabungkan hasil vision + jawaban akhir menjadi metrik per-kasus,
   lalu diagregasi menjadi metrik per-model (micro-average).

Semua ini menggunakan pipeline SEBENARNYA yang dipakai endpoint produksi
(vision prompt, generative prompt) supaya hasil evaluasi merefleksikan
kondisi nyata aplikasi, bukan simulasi terpisah.
"""

from __future__ import annotations
import json
import time
import os
import base64
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.models.schema import VisionExtraction
from app.api.chat import (
    VISION_PROMPT_TEMPLATE,
    TEXT_ONLY_SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    format_rag_context,
)
from app.services.rag_service import query_knowledge_base
from evaluation.metrics import (
    score_label_set,
    score_answer_relevance,
    aggregate_scores,
    compute_accuracy,
    LabelSetScore,
)

logger = logging.getLogger("ai_chart.evaluation")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

EVAL_DIR = os.path.dirname(__file__)
DEFAULT_DATASET_PATH = os.path.join(EVAL_DIR, "dataset", "ground_truth.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    if not cases:
        raise ValueError(
            f"Dataset '{dataset_path}' tidak memiliki 'cases'. "
            "Isi dulu ground_truth.json dengan minimal 1 kasus uji."
        )
    return cases


def _image_to_base64(image_path: str) -> Optional[str]:
    if not image_path:
        return None
    full_path = image_path
    if not os.path.isabs(full_path):
        # relative terhadap root backend/
        backend_root = os.path.dirname(EVAL_DIR)
        full_path = os.path.join(backend_root, image_path)
    if not os.path.exists(full_path):
        logger.warning("⚠️ [EVAL] Gambar tidak ditemukan: %s", full_path)
        return None
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Sama seperti helper di chat.py: coba ambil substring JSON dari teks."""
    try:
        if not text or "{" not in text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _run_vision_model(model_name: str, image_b64: str, query: str) -> Dict[str, Any]:
    """
    Jalankan satu model Ollama sebagai vision model untuk satu gambar,
    kembalikan dict hasil ekstraksi (key_patterns, key_levels, indicators, summary)
    beserta metadata (latency, error jika ada).
    """
    result = {
        "model": model_name,
        "success": False,
        "latency_sec": None,
        "raw_content": None,
        "extracted": None,
        "error": None,
    }
    try:
        vision_instance = ChatOllama(
            model=model_name,
            base_url=settings.OLLAMA_BASE_URL,
            format="json",
            temperature=0.0,
        )
        image_message = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{image_b64}",
        }
        vision_message = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT_TEMPLATE},
            image_message,
        ])

        start = time.time()
        response = vision_instance.invoke([vision_message])
        elapsed = time.time() - start
        result["latency_sec"] = round(elapsed, 3)
        result["raw_content"] = getattr(response, "content", "")

        try:
            parsed = json.loads(result["raw_content"])
        except Exception:
            parsed = _extract_json_from_text(result["raw_content"])

        if parsed:
            validated = VisionExtraction(**parsed)
            result["extracted"] = validated.dict()
            result["success"] = True
        else:
            result["error"] = "Gagal parsing JSON dari respons vision model"

    except Exception as e:
        result["error"] = str(e)
        logger.warning("⚠️ [EVAL] Vision model %s gagal: %s", model_name, e)

    return result


def _run_generative_model(
    model_name: str,
    query: str,
    vision_json: Optional[str],
    rag_context: str,
    use_multimodal_prompt: bool,
) -> Dict[str, Any]:
    """
    Jalankan satu model Ollama sebagai generative/synthesis model,
    kembalikan teks jawaban akhir beserta metadata.
    """
    result = {
        "model": model_name,
        "success": False,
        "latency_sec": None,
        "answer_text": "",
        "error": None,
    }
    try:
        gen_instance = ChatOllama(
            model=model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2,
        )

        if use_multimodal_prompt:
            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                chat_history="Belum ada riwayat percakapan.",
                vision_json=vision_json or "{}",
                user_query=query,
                rag_context=rag_context,
            )
        else:
            prompt = TEXT_ONLY_SYNTHESIS_PROMPT_TEMPLATE.format(
                chat_history="Belum ada riwayat percakapan.",
                user_query=query,
                rag_context=rag_context,
            )

        message = HumanMessage(content=prompt)

        start = time.time()
        response = gen_instance.invoke([message])
        elapsed = time.time() - start

        result["latency_sec"] = round(elapsed, 3)
        result["answer_text"] = getattr(response, "content", "")
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        logger.warning("⚠️ [EVAL] Generative model %s gagal: %s", model_name, e)

    return result


def evaluate_single_case(
    case: Dict[str, Any],
    vision_model: str,
    generative_model: str,
) -> Dict[str, Any]:
    """
    Evaluasi 1 kasus uji untuk 1 kombinasi (vision_model, generative_model).
    Mengembalikan detail hasil + skor precision/recall/f1 untuk kasus ini.
    """
    case_id = case.get("case_id", "unknown")
    query = case.get("query", "")
    image_path = case.get("image_path")

    case_result = {
        "case_id": case_id,
        "query": query,
        "vision_model": vision_model,
        "generative_model": generative_model,
        "vision": None,
        "generative": None,
        "scores": {},
    }

    vision_json_str = None
    vision_extracted = {"key_patterns": [], "key_levels": [], "indicators": [], "summary": ""}
    has_image = bool(image_path)

    # --- STEP 1: Vision extraction (jika ada gambar) ---
    if has_image:
        image_b64 = _image_to_base64(image_path)
        if image_b64:
            vision_result = _run_vision_model(vision_model, image_b64, query)
            case_result["vision"] = vision_result
            if vision_result["success"]:
                vision_extracted = vision_result["extracted"]
                vision_json_str = json.dumps(vision_extracted, ensure_ascii=False)
        else:
            case_result["vision"] = {
                "model": vision_model,
                "success": False,
                "error": f"Gambar tidak ditemukan/tidak dapat dibaca: {image_path}",
            }

    # --- STEP 2: Hitung skor vision (precision/recall/f1) ---
    patterns_score = score_label_set(
        case.get("expected_key_patterns", []), vision_extracted.get("key_patterns", [])
    )
    levels_score = score_label_set(
        case.get("expected_key_levels", []), vision_extracted.get("key_levels", [])
    )
    indicators_score = score_label_set(
        case.get("expected_indicators", []), vision_extracted.get("indicators", [])
    )

    case_result["scores"]["key_patterns"] = patterns_score.to_dict()
    case_result["scores"]["key_levels"] = levels_score.to_dict()
    case_result["scores"]["indicators"] = indicators_score.to_dict()

    # --- STEP 3: RAG + Generative (jawaban akhir) ---
    try:
        rag_docs = query_knowledge_base(query, k=5)
        rag_context = format_rag_context(rag_docs)
    except Exception as e:
        logger.warning("⚠️ [EVAL] RAG query gagal untuk case %s: %s", case_id, e)
        rag_context = "Tidak dapat mengakses knowledge base saat ini."

    gen_result = _run_generative_model(
        generative_model,
        query=query,
        vision_json=vision_json_str,
        rag_context=rag_context,
        use_multimodal_prompt=has_image,
    )
    case_result["generative"] = gen_result

    answer_score = score_answer_relevance(
        case.get("expected_answer_keywords", []), gen_result.get("answer_text", "")
    )
    case_result["scores"]["answer_relevance"] = answer_score.to_dict()

    # --- STEP 4: Skor gabungan untuk kasus ini (semua label set) ---
    all_scores_for_case: List[LabelSetScore] = [
        patterns_score, levels_score, indicators_score, answer_score
    ]
    case_result["scores"]["case_overall"] = aggregate_scores(all_scores_for_case)

    return case_result


def evaluate_model_pair(
    vision_model: str,
    generative_model: str,
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> Dict[str, Any]:
    """
    Jalankan evaluasi penuh (semua kasus di dataset) untuk 1 kombinasi model,
    lalu agregasi menjadi metrik precision/recall/f1/accuracy keseluruhan.
    """
    cases = _load_dataset(dataset_path)
    case_results = []

    pattern_scores, level_scores, indicator_scores, answer_scores, overall_scores = [], [], [], [], []
    perfect_case_count = 0
    total_latency = {"vision": [], "generative": []}

    for case in cases:
        result = evaluate_single_case(case, vision_model, generative_model)
        case_results.append(result)

        pattern_scores.append(_dict_to_labelscore(result["scores"]["key_patterns"]))
        level_scores.append(_dict_to_labelscore(result["scores"]["key_levels"]))
        indicator_scores.append(_dict_to_labelscore(result["scores"]["indicators"]))
        answer_scores.append(_dict_to_labelscore(result["scores"]["answer_relevance"]))
        overall_scores.append(_dict_to_labelscore(result["scores"]["case_overall"]))

        if result["scores"]["case_overall"]["f1_score"] == 1.0:
            perfect_case_count += 1

        if result.get("vision") and result["vision"].get("latency_sec") is not None:
            total_latency["vision"].append(result["vision"]["latency_sec"])
        if result.get("generative") and result["generative"].get("latency_sec") is not None:
            total_latency["generative"].append(result["generative"]["latency_sec"])

        # Cetak progres per-kasus supaya evaluasi yang lama tidak terasa 'diam'
        case_f1 = result["scores"]["case_overall"]["f1_score"]
        print(
            f"  [{vision_model} / {generative_model}] "
            f"{result['case_id']}: F1={case_f1} "
            f"(P={result['scores']['case_overall']['precision']}, "
            f"R={result['scores']['case_overall']['recall']})"
        )

    metrics = {
        "key_patterns": aggregate_scores(pattern_scores),
        "key_levels": aggregate_scores(level_scores),
        "indicators": aggregate_scores(indicator_scores),
        "answer_relevance": aggregate_scores(answer_scores),
        "overall": aggregate_scores(overall_scores),
    }
    accuracy = compute_accuracy(perfect_case_count, len(cases))

    summary = {
        "vision_model": vision_model,
        "generative_model": generative_model,
        "num_cases": len(cases),
        "metrics": metrics,
        "accuracy_perfect_match": accuracy,
        "avg_latency_sec": {
            "vision": round(sum(total_latency["vision"]) / len(total_latency["vision"]), 3)
                if total_latency["vision"] else None,
            "generative": round(sum(total_latency["generative"]) / len(total_latency["generative"]), 3)
                if total_latency["generative"] else None,
        },
        "case_results": case_results,
    }

    # --- CETAK RINGKASAN DETAIL (breakdown per kategori metrik) UNTUK MODEL INI ---
    print(f"\n--- Ringkasan Metrik: vision={vision_model} | generative={generative_model} ---")
    for category, m in metrics.items():
        print(
            f"  {category:<18} Precision={m['precision']:<8} Recall={m['recall']:<8} "
            f"F1-Score={m['f1_score']:<8} (TP={m['true_positive']}, FP={m['false_positive']}, FN={m['false_negative']})"
        )
    print(f"  {'Accuracy (perfect-match)':<18} {accuracy}")
    print("-" * 70 + "\n")

    return summary


def _dict_to_labelscore(d: Dict[str, Any]) -> LabelSetScore:
    """Helper untuk merekonstruksi LabelSetScore dari dict (untuk re-aggregasi)."""
    ls = LabelSetScore()
    ls.true_positive = d.get("true_positive", 0)
    ls.false_positive = d.get("false_positive", 0)
    ls.false_negative = d.get("false_negative", 0)
    return ls


def _print_comparison_table(ranking: List[Dict[str, Any]]) -> None:
    """
    Cetak tabel perbandingan Precision / Recall / F1-Score / Accuracy
    ke terminal/console supaya langsung terlihat saat proses evaluasi selesai
    (berguna untuk screenshot/lampiran laporan KP).
    """
    header = (
        f"{'Rank':<5}{'Vision Model':<22}{'Generative Model':<22}"
        f"{'Precision':<11}{'Recall':<10}{'F1-Score':<11}{'Accuracy':<10}"
    )
    line = "=" * len(header)

    print("\n" + line)
    print("HASIL PERBANDINGAN MODEL (Precision / Recall / F1-Score / Accuracy)")
    print(line)
    print(header)
    print("-" * len(header))

    for row in ranking:
        print(
            f"{row['rank']:<5}"
            f"{str(row['vision_model'])[:20]:<22}"
            f"{str(row['generative_model'])[:20]:<22}"
            f"{row['overall_precision']:<11}"
            f"{row['overall_recall']:<10}"
            f"{row['overall_f1']:<11}"
            f"{row['accuracy_perfect_match']:<10}"
        )

    print(line)
    if ranking:
        best = ranking[0]
        print(
            f"🏆 MODEL TERBAIK: vision={best['vision_model']} | "
            f"generative={best['generative_model']} | F1-Score={best['overall_f1']}"
        )
    print(line + "\n")


def _build_txt_report(comparison_result: Dict[str, Any]) -> str:
    """
    Bangun ringkasan hasil evaluasi dalam bentuk teks polos (.txt) yang mudah
    dibaca dan langsung bisa dilampirkan ke laporan, tanpa perlu membuka JSON.
    """
    lines: List[str] = []
    add = lines.append

    add("=" * 91)
    add("LAPORAN EVALUASI AI CHART ANALYST — PRECISION / RECALL / F1-SCORE / ACCURACY")
    add("=" * 91)
    add(f"Timestamp        : {comparison_result['timestamp']}")
    add(f"Dataset dipakai  : {comparison_result['dataset_used']}")
    add(f"Jumlah kombinasi : {comparison_result['num_combinations']}")
    add("")

    add("-" * 91)
    add("TABEL PERBANDINGAN MODEL (diurutkan dari F1-Score tertinggi)")
    add("-" * 91)
    header = (
        f"{'Rank':<5}{'Vision Model':<22}{'Generative Model':<22}"
        f"{'Precision':<11}{'Recall':<10}{'F1-Score':<11}{'Accuracy':<10}"
    )
    add(header)
    add("-" * len(header))
    for row in comparison_result["ranking"]:
        add(
            f"{row['rank']:<5}"
            f"{str(row['vision_model'])[:20]:<22}"
            f"{str(row['generative_model'])[:20]:<22}"
            f"{row['overall_precision']:<11}"
            f"{row['overall_recall']:<10}"
            f"{row['overall_f1']:<11}"
            f"{row['accuracy_perfect_match']:<10}"
        )
    add("")

    best = comparison_result.get("best_model")
    if best:
        add(
            f"MODEL TERBAIK: vision={best['vision_model']} | "
            f"generative={best['generative_model']} | F1-Score={best['overall_f1']}"
        )
    add("")

    add("=" * 91)
    add("RINCIAN METRIK PER MODEL (breakdown per kategori)")
    add("=" * 91)
    for summary in comparison_result["full_results"]:
        add("")
        add(f"Model: vision={summary['vision_model']} | generative={summary['generative_model']}")
        add(f"Jumlah kasus diuji: {summary['num_cases']}")
        add("-" * 70)
        for category, m in summary["metrics"].items():
            add(
                f"  {category:<18} Precision={m['precision']:<8} Recall={m['recall']:<8} "
                f"F1-Score={m['f1_score']:<8} (TP={m['true_positive']}, FP={m['false_positive']}, FN={m['false_negative']})"
            )
        add(f"  {'Accuracy (perfect-match)':<18} {summary['accuracy_perfect_match']}")
        avg_lat = summary.get("avg_latency_sec", {})
        add(f"  Rata-rata latency  : vision={avg_lat.get('vision')}s | generative={avg_lat.get('generative')}s")

    add("")
    add("=" * 91)
    add("RINCIAN PER KASUS UJI")
    add("=" * 91)
    for summary in comparison_result["full_results"]:
        add("")
        add(f">> Model: vision={summary['vision_model']} | generative={summary['generative_model']}")
        for case in summary["case_results"]:
            overall = case["scores"]["case_overall"]
            add(
                f"   [{case['case_id']}] query=\"{case['query'][:60]}\" "
                f"-> F1={overall['f1_score']} P={overall['precision']} R={overall['recall']}"
            )

    add("")
    add("=" * 91)
    add("Catatan metodologi: metrik dihitung dengan pencocokan label case-insensitive")
    add("& toleran variasi kata (fuzzy ringan), micro-averaged di seluruh kasus uji.")
    add("Lihat evaluation/README.md untuk penjelasan lengkap metodologi.")
    add("=" * 91)

    return "\n".join(lines)


def compare_models(
    model_combinations: List[Dict[str, str]],
    dataset_path: str = DEFAULT_DATASET_PATH,
    save_result: bool = True,
) -> Dict[str, Any]:
    """
    Jalankan evaluasi untuk BEBERAPA kombinasi model sekaligus, lalu
    bandingkan F1-score keseluruhan untuk menentukan model terbaik.

    model_combinations contoh:
    [
        {"vision_model": "qwen2.5vl:7b", "generative_model": "llama3.1:8b"},
        {"vision_model": "llava:13b", "generative_model": "mistral:7b"},
    ]
    """
    all_summaries = []
    for combo in model_combinations:
        vision_model = combo.get("vision_model") or settings.VISION_MODEL
        generative_model = combo.get("generative_model") or settings.GENERATIVE_MODEL
        logger.info("🔬 [EVAL] Mengevaluasi kombinasi: vision=%s | generative=%s", vision_model, generative_model)
        summary = evaluate_model_pair(vision_model, generative_model, dataset_path)
        all_summaries.append(summary)

    # Ranking berdasarkan F1-score overall (descending)
    ranked = sorted(
        all_summaries,
        key=lambda s: s["metrics"]["overall"]["f1_score"],
        reverse=True,
    )

    best = ranked[0] if ranked else None

    comparison_result = {
        "timestamp": datetime.now().isoformat(),
        "dataset_used": dataset_path,
        "num_combinations": len(model_combinations),
        "ranking": [
            {
                "rank": idx + 1,
                "vision_model": s["vision_model"],
                "generative_model": s["generative_model"],
                "overall_f1": s["metrics"]["overall"]["f1_score"],
                "overall_precision": s["metrics"]["overall"]["precision"],
                "overall_recall": s["metrics"]["overall"]["recall"],
                "accuracy_perfect_match": s["accuracy_perfect_match"],
                "avg_latency_sec": s["avg_latency_sec"],
            }
            for idx, s in enumerate(ranked)
        ],
        "best_model": {
            "vision_model": best["vision_model"],
            "generative_model": best["generative_model"],
            "overall_f1": best["metrics"]["overall"]["f1_score"],
        } if best else None,
        "full_results": all_summaries,
    }

    # --- CETAK RINGKASAN METRIK KE TERMINAL ---
    _print_comparison_table(comparison_result["ranking"])

    if save_result:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- Simpan JSON (data lengkap, untuk diproses ulang / dibuka programatik) ---
        json_filename = f"eval_comparison_{timestamp_str}.json"
        json_filepath = os.path.join(RESULTS_DIR, json_filename)
        try:
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            comparison_result["saved_to_json"] = json_filepath
            logger.info("💾 [EVAL] Hasil evaluasi (JSON) disimpan ke %s", json_filepath)
        except Exception as e:
            logger.error("⚠️ [EVAL] Gagal menyimpan hasil evaluasi JSON: %s", e)

        # --- Simpan TXT (ringkasan yang mudah dibaca, untuk lampiran laporan) ---
        txt_filename = f"eval_comparison_{timestamp_str}.txt"
        txt_filepath = os.path.join(RESULTS_DIR, txt_filename)
        try:
            txt_report = _build_txt_report(comparison_result)
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write(txt_report)
            comparison_result["saved_to_txt"] = txt_filepath
            logger.info("💾 [EVAL] Hasil evaluasi (TXT) disimpan ke %s", txt_filepath)
        except Exception as e:
            logger.error("⚠️ [EVAL] Gagal menyimpan hasil evaluasi TXT: %s", e)

    return comparison_result


def list_saved_results() -> List[str]:
    """Daftar file hasil evaluasi (.json dan .txt) yang tersimpan di evaluation/results/."""
    if not os.path.exists(RESULTS_DIR):
        return []
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json") or f.endswith(".txt")]
    files.sort(reverse=True)
    return files


def load_saved_result(filename: str) -> Any:
    """
    Muat kembali 1 file hasil evaluasi berdasarkan nama file.
    File .json dikembalikan sebagai dict, file .txt dikembalikan sebagai teks polos.
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File hasil evaluasi '{filename}' tidak ditemukan.")
    with open(filepath, "r", encoding="utf-8") as f:
        if filename.endswith(".json"):
            return json.load(f)
        return f.read()