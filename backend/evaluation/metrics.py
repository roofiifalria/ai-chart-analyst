"""
evaluation/metrics.py

Modul metrik untuk evaluation_service.py: menghitung Precision / Recall /
F1-Score untuk perbandingan label (key_patterns, key_levels, indicators)
dan relevansi jawaban (answer_relevance), lalu mengagregasi skor per-kasus
menjadi skor keseluruhan per-model (micro-average), serta menghitung
Accuracy berbasis "perfect match" (F1 = 1.0 pada seluruh kategori kasus).

METODOLOGI PENCOCOKAN LABEL (case-insensitive + fuzzy ringan):
- Setiap label dinormalisasi: lowercase, strip whitespace, spasi ganda
  dirapikan menjadi satu spasi, tanda baca ringan (- _ . ,) diseragamkan
  menjadi spasi tunggal. Ini membuat "Golden Cross", "golden-cross", dan
  "golden_cross " dianggap label yang sama.
- Setelah normalisasi, pencocokan dilakukan dengan exact match antar
  string yang sudah dinormalisasi (bukan exact match string mentah).
  Ini disebut "fuzzy ringan": cukup untuk mentoleransi variasi format
  penulisan model, tapi tidak melakukan similarity/embedding matching
  yang bisa menimbulkan false positive pada label yang mirip tapi beda arti.

CARA HITUNG TP / FP / FN (untuk satu kasus, satu kategori label):
- True Positive  (TP): label yang ADA di expected DAN ADA di actual.
- False Positive (FP): label yang ADA di actual TAPI TIDAK ADA di expected
                        (model "berhalusinasi" / menyebut hal yang tidak diminta).
- False Negative (FN): label yang ADA di expected TAPI TIDAK ADA di actual
                        (model "melewatkan" / gagal mendeteksi).
- True Negative tidak dihitung karena label set bersifat open-vocabulary
  (tidak ada "populasi lengkap label yang mungkin" untuk dijadikan pembagi).

Precision = TP / (TP + FP)  -> dari yang disebut model, berapa % yang benar.
Recall    = TP / (TP + FN)  -> dari yang seharusnya disebut, berapa % yang tertangkap.
F1-Score  = 2 * P * R / (P + R)

Kasus tepi (edge case) yang ditangani secara eksplisit:
- expected kosong & actual kosong -> P=R=F1=1.0 (tidak ada yang diharapkan,
  tidak ada yang salah disebut: kasus ini dianggap "berhasil sempurna").
- expected kosong & actual tidak kosong -> P=0.0, R=1.0 (semua FP, recall
  otomatis sempurna karena tidak ada yang perlu ditemukan).
- expected tidak kosong & actual kosong -> P=1.0, R=0.0 (tidak ada FP karena
  tidak ada yang disebut, tapi recall gagal total).
- TP+FP=0 dengan expected tidak kosong -> Precision didefinisikan 0.0
  (bukan undefined/NaN) supaya aman dipakai di rata-rata dan perbandingan.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Normalisasi & pencocokan label
# ---------------------------------------------------------------------------

_PUNCT_TO_SPACE_RE = re.compile(r"[-_.,;:/\\]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _normalize_label(label: Any) -> str:
    """
    Normalisasi satu label untuk pencocokan case-insensitive + fuzzy ringan.
    Contoh: " Golden_Cross " dan "golden-cross" -> "golden cross"
    """
    if label is None:
        return ""
    text = str(label).strip().lower()
    text = _PUNCT_TO_SPACE_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _normalize_label_list(labels: List[Any]) -> List[str]:
    """Normalisasi list label, buang string kosong hasil normalisasi, unik-kan."""
    if not labels:
        return []
    normalized = (_normalize_label(l) for l in labels)
    # dedupe sambil pertahankan urutan kemunculan pertama
    seen: Dict[str, None] = {}
    for n in normalized:
        if n and n not in seen:
            seen[n] = None
    return list(seen.keys())


# ---------------------------------------------------------------------------
# LabelSetScore: struktur skor per-kasus / per-kategori
# ---------------------------------------------------------------------------

@dataclass
class LabelSetScore:
    """
    Menyimpan TP/FP/FN mentah untuk satu perbandingan label set (atau satu
    penilaian relevansi jawaban), lalu menyediakan precision/recall/f1
    turunan via properti dan .to_dict().

    Dirancang agar bisa "dijumlahkan" antar banyak kasus (lihat
    aggregate_scores) untuk menghasilkan metrik micro-averaged.
    """
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    # Detail label mana saja yang match/miss, berguna untuk debugging &
    # tidak dipakai dalam perhitungan agregat (tidak ikut ke .to_dict()
    # utama supaya struktur output tetap ringkas & konsisten).
    matched_labels: List[str] = field(default_factory=list)
    missing_labels: List[str] = field(default_factory=list)
    extra_labels: List[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        if denom == 0:
            # Tidak ada yang disebut model sama sekali.
            # Jika memang tidak ada yang diharapkan (FN juga 0), anggap
            # precision sempurna (tidak ada kesalahan). Jika ada yang
            # diharapkan tapi tidak disebut, precision tetap didefinisikan
            # 1.0 karena tidak ada FP (tidak salah menyebut apa pun) —
            # kegagalannya tercermin di recall, bukan precision.
            return 1.0
        return round(self.true_positive / denom, 4)

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        if denom == 0:
            # Tidak ada yang diharapkan sama sekali -> recall sempurna
            # secara definisi (tidak ada yang mungkin terlewat).
            return 1.0
        return round(self.true_positive / denom, 4)

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "matched_labels": self.matched_labels,
            "missing_labels": self.missing_labels,
            "extra_labels": self.extra_labels,
        }


# ---------------------------------------------------------------------------
# Fungsi penilaian per-kasus
# ---------------------------------------------------------------------------

def score_label_set(expected: List[Any], actual: List[Any]) -> LabelSetScore:
    """
    Bandingkan satu set label yang diharapkan (ground truth) dengan satu set
    label hasil ekstraksi model (misalnya key_patterns, key_levels, atau
    indicators untuk satu kasus uji), lalu kembalikan LabelSetScore berisi
    TP/FP/FN beserta rincian label yang match/miss/extra.

    Pencocokan bersifat case-insensitive + fuzzy ringan (lihat _normalize_label).
    """
    expected_norm = _normalize_label_list(expected or [])
    actual_norm = _normalize_label_list(actual or [])

    expected_set = set(expected_norm)
    actual_set = set(actual_norm)

    matched = expected_set & actual_set
    missing = expected_set - actual_set   # diharapkan tapi tidak muncul -> FN
    extra = actual_set - expected_set     # muncul tapi tidak diharapkan -> FP

    return LabelSetScore(
        true_positive=len(matched),
        false_positive=len(extra),
        false_negative=len(missing),
        matched_labels=sorted(matched),
        missing_labels=sorted(missing),
        extra_labels=sorted(extra),
    )


def score_answer_relevance(expected_keywords: List[Any], answer_text: str) -> LabelSetScore:
    """
    Nilai relevansi jawaban akhir generative model dengan mengecek keyword
    mana dari expected_answer_keywords yang benar-benar muncul (sebagai
    substring, case-insensitive + fuzzy ringan) di dalam answer_text.

    Ini BUKAN pencocokan token-per-token pada jawaban (jawaban model bersifat
    bebas/naratif), melainkan pengecekan "apakah topik/konsep kunci yang
    diharapkan disebutkan di jawaban". Karena itu tidak ada konsep "extra
    keyword" (false_positive selalu 0): jawaban boleh berisi kalimat apa pun
    di luar keyword tanpa dianggap kesalahan, karena tugas model adalah
    menjelaskan, bukan membatasi diri hanya pada daftar keyword.
    """
    keywords_norm = _normalize_label_list(expected_keywords or [])
    answer_norm = _normalize_label(answer_text or "")

    matched: List[str] = []
    missing: List[str] = []

    for kw in keywords_norm:
        if kw and kw in answer_norm:
            matched.append(kw)
        else:
            missing.append(kw)

    return LabelSetScore(
        true_positive=len(matched),
        false_positive=0,
        false_negative=len(missing),
        matched_labels=sorted(matched),
        missing_labels=sorted(missing),
        extra_labels=[],
    )


# ---------------------------------------------------------------------------
# Agregasi lintas-kasus (micro-average) & accuracy
# ---------------------------------------------------------------------------

def aggregate_scores(scores: List[LabelSetScore]) -> Dict[str, Any]:
    """
    Gabungkan banyak LabelSetScore (misalnya seluruh kasus uji dalam satu
    kategori, atau seluruh kategori dalam satu kasus) menjadi satu metrik
    micro-averaged: TP/FP/FN dijumlahkan dahulu, baru precision/recall/f1
    dihitung dari total tersebut. Ini berbeda dari macro-average (rata-rata
    dari precision/recall per-kasus) dan sengaja dipilih micro-average agar
    kasus dengan jumlah label lebih banyak punya bobot lebih besar,
    merefleksikan performa model secara keseluruhan pada seluruh dataset.

    Mengembalikan dict (bukan LabelSetScore) sesuai pemakaian di
    evaluation_service.py: metrics["overall"]["f1_score"], dsb.
    """
    total = LabelSetScore()
    for s in scores or []:
        total.true_positive += s.true_positive
        total.false_positive += s.false_positive
        total.false_negative += s.false_negative
        total.matched_labels.extend(s.matched_labels)
        total.missing_labels.extend(s.missing_labels)
        total.extra_labels.extend(s.extra_labels)

    return total.to_dict()


def compute_accuracy(perfect_case_count: int, total_cases: int) -> float:
    """
    Accuracy berbasis "perfect match": proporsi kasus uji yang mencapai
    F1-Score = 1.0 pada case_overall (seluruh kategori: key_patterns,
    key_levels, indicators, answer_relevance sekaligus benar sempurna)
    dari total kasus yang dievaluasi.

    Dikembalikan sebagai float (dibulatkan 4 desimal), 0.0 jika tidak ada
    kasus sama sekali (menghindari ZeroDivisionError).
    """
    if not total_cases:
        return 0.0
    return round(perfect_case_count / total_cases, 4)