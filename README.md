# Laporan Perencanaan Proyek: AI Chart Analyst (Versi Powerhouse Lokal)

**Versi:** 1.3 (Lokal - Qwen3-VL + Chroma Cloud)
**Tanggal:** 16 November 2025
**Disusun Oleh:** (Gemini)
**Status:** Draf Perencanaan (Revisi ChromaDB Cloud)

## Ringkasan Eksekutif

Proyek "AI Chart Analyst" bertujuan untuk mengembangkan sistem kecerdasan buatan canggih yang mampu memberikan analisis teknikal mendalam dari gambar chart keuangan (saham, kripto, forex) yang diunggah oleh pengguna.

Inti dari proyek ini adalah arsitektur **RAG Hybrid yang ditenagai oleh satu model "Powerhouse" multimodal (`qwen3-vl:235b-cloud`)** yang berjalan 100% secara lokal (self-hosted). Sistem ini memanfaatkan kemampuan "Superior Text-Centric Performance" dan "Stronger Multimodal Reasoning" dari model tersebut untuk melakukan ekstraksi visual dan sintesis analitis dalam satu alur kerja yang terpadu.

Database pengetahuan (RAG) akan di-host pada layanan **ChromaDB Cloud** terkelola untuk skalabilitas dan keandalan.

## 1. Pendahuluan

...(Tidak ada perubahan di bagian ini)...

## 2. Konsep & Arsitektur Sistem

### 2.1. Filosofi Desain: "Powerhouse Terpadu" (Lokal)

...(Tidak ada perubahan di bagian ini)...

### 2.2. Komponen Utama Sistem

1.  **Frontend (Aplikasi Klien):**
    * (Tidak berubah) Teknologi: React/Vite, Vue, dll.

2.  **Backend (API Orchestrator Lokal):**
    * (Tidak berubah) Teknologi: Python (FastAPI).

3.  **Modul AI Terpadu (Visi + Generatif):**
    * **Deskripsi:** Layanan AI Self-Hosted (via **Ollama**) yang mengekpos model `qwen3-vl`.
    * **Teknologi (Model):** **`qwen3-vl:235b-cloud`**.

4.  **Knowledge Base (Vector Database Cloud):**
    * **Deskripsi:** Database khusus yang berjalan di layanan **ChromaDB Cloud** terkelola.
    * **Teknologi:** **ChromaDB Cloud**.

### 2.3. Diagram Arsitektur

`[Diagram yang menunjukkan alur: Pengguna -> Frontend (Web) -> Backend (Lokal). Backend kemudian bercabang ke 3 jalur: (1) -> Server Model Lokal (Ollama) (untuk Visi & Generasi). (2) -> **Koneksi Internet ke ChromaDB Cloud** (untuk RAG). Hasil dari (1) dan (2) kembali ke Backend untuk sintesis akhir -> Frontend -> Pengguna.]`

### 2.4. Alur Data Detail (Langkah demi Langkah)

1.  **Input:** Pengguna mengunggah `chart.png` dan bertanya, "Analisis pola ini dan apa artinya?"
2.  **Orkestrasi (Backend Lokal):** Backend FastAPI menerima file dan teks.
3.  **Ekstraksi Visi (Langkah A - Panggilan AI #1):**
    * Backend mengirim `chart.png` ke **Modul AI Terpadu (`qwen3-vl`)** untuk ekstraksi JSON.
4.  **Retrieval (Langkah B - Panggilan DB Cloud):**
    * Backend mengambil keyword dari JSON, membuat *embedding* (lokal), dan meng-query **ChromaDB Cloud** melalui internet.
5.  **Augmentasi & Generasi (Langkah C - Panggilan AI #2):**
    * Backend menyusun *Final Prompt* untuk **Modul AI Terpadu (`qwen3-vl`)** (Gambar + Pertanyaan + Konteks RAG + JSON).
6.  **Sintesis:** `qwen3-vl` memproses semua konteks dan menulis analisis.
7.  **Respon:** Jawaban dikirim kembali ke pengguna melalui Frontend.

## 3. Kebutuhan Proyek & Persiapan

### 3.1. Persiapan Data (Knowledge Base)

...(Tidak ada perubahan di bagian ini)...

### 3.2. Tumpukan Teknologi (Tools & Software) - VERSI POWERHOUSE LOKAL

| Kategori | Teknologi (Rekomendasi) | Alternatif | Alasan |
| :--- | :--- | :--- | :--- |
| **Frontend** | **React (Vite) + TailwindCSS** | Vue.js, Svelte | Ekosistem besar, cepat, modern. |
| **Backend** | **Python (FastAPI)** | Flask | Asinkron, ekosistem AI terbaik. |
| **Orkestrasi AI** | **LangChain (Python)** | LlamaIndex | Standar industri untuk merangkai *chain* AI. |
| **Server AI Lokal** | **Ollama** | vLLM, TensorRT-LLM | Sangat mudah untuk setup & menjalankan model besar via API. |
| **Model AI Terpadu** | **`qwen3-vl:235b-cloud` (GGUF/Quantized)** | Model `qwen-vl` terbesar yang tersedia | Model tunggal untuk Visi dan Generasi dengan kemampuan SOTA. |
| **Vector Database** | **ChromaDB Cloud** | Pinecone, Weaviate (Cloud) | Layanan terkelola, tidak perlu setup server DB. |
| **Model Embedding** | **HuggingFace (misal: `all-MiniLM-L6-v2`)** | `BAAI/bge-m3` | Gratis dan berjalan 100% lokal. |

### 3.3. Kebutuhan Perangkat Keras & Infrastruktur (Lokal)

**Ini adalah PERUBAHAN PALING KRITIS.** Menjalankan model 235B+ parameter secara lokal **BUKAN** untuk perangkat keras konsumen.

* **GPU (Wajib - Kelas Server):**
    * Model 235B yang terkuantisasi (misal, Q4_K_M GGUF) membutuhkan sekitar **120-130 GB VRAM**.
    * **Minimum (Mutlak):** **2x NVIDIA H100 (80GB)** atau **2x NVIDIA A100 (80GB)** atau **4x RTX 4090 (24GB)**.
    * **Rekomendasi:** Server Multi-GPU khusus (misal, 4x H100 80GB).
* **RAM Sistem:** Minimal **128GB RAM**, direkomendasikan **256GB+**.
* **Penyimpanan:** SSD NVMe M.2 (Kapasitas 1TB+).
* **Sistem Operasi:** **Linux (Ubuntu Server)**.
* **Jaringan:** Koneksi internet yang **stabil dan cepat** (diperlukan untuk koneksi konstan ke ChromaDB Cloud).

### 3.4. Kebutuhan Tim (Perkiraan)

...(Tidak ada perubahan di bagian ini)...

## 4. Rencana Pengembangan (Milestones)

* **Fase 0: Persiapan & Ingesti Data (Minggu 1-2)**
    * [ ] Kumpulkan semua sumber data (PDF, artikel).
    * [ ] Tulis skrip untuk membersihkan dan melakukan *chunking* data.
    * [ ] **Setup Akun ChromaDB Cloud.**
    * [ ] **Jalankan Notebook Ingesti (`01_Data_Ingestion.ipynb`)** untuk memuat data ke Cloud.
    * [ ] Uji *retrieval* (pencarian) pada Cloud DB.

* **Fase 1: Pembangunan Backend & Otak AI (Minggu 3-6)**
    * [ ] Setup Lingkungan AI Server (Ollama, multi-GPU).
    * [ ] Setup proyek FastAPI.
    * [ ] Buat *chain* LangChain untuk **Ekstraksi Visi Lokal**.
    * [ ] Buat *chain* LangChain untuk **Sintesis Akhir Lokal** (termasuk *query* ke ChromaDB Cloud).
    * [ ] Buat endpoint API `/analyze`.
    * [ ] Uji endpoint secara menyeluruh.

* **Fase 2: Pembangunan Frontend & Integrasi (Minggu 7-9)**
    * ...(Tidak ada perubahan di bagian ini)...

* **Fase 3: Pengujian, Umpan Balik & Peluncuran (Minggu 10-12)**
    * ...(Tidak ada perubahan di bagian ini)...

## 5. Analisis Risiko & Mitigasi (Versi Powerhouse Lokal)

| Risiko | Deskripsi | Probabilitas | Dampak | Mitigasi |
| :--- | :--- | :--- | :--- | :--- |
| **Kebutuhan Hardware** | **Risiko #1.** Perangkat keras tidak mampu menjalankan model 235B. | **Sangat Tinggi** | **Kritis (Blocker)** | 1. Gunakan versi model yang **jauh lebih kecil**. 2. Investasi besar pada perangkat keras server. |
| **Ketergantungan Jaringan**| Koneksi internet ke ChromaDB Cloud lambat atau tidak stabil, menyebabkan kegagalan RAG. | **Sedang** | **Tinggi** | Pastikan server backend memiliki koneksi internet premium/stabil. Terapkan *caching* dan *retry logic*. |
| **Kecepatan Inferensi**| Model 235B membutuhkan waktu 1-3 menit per respons. | **Tinggi** | **Tinggi** | Gunakan kuantisasi (GGUF). Gunakan server inferensi (vLLM). Beri *loading indicator* yang jelas di Frontend. |
| **Kompleksitas Setup**| Kesulitan mengkonfigurasi driver CUDA dan server model untuk **multi-GPU**. | **Tinggi** | **Tinggi** | Gunakan Ollama untuk menyederhanakan. Ikuti panduan MLOps dengan ketat. |
| **Halusinasi RAG**| (Risiko ini tetap ada). | **Rendah-Sedang**| **Tinggi** | Kualitas *Knowledge Base* harus sangat tinggi. |
| **Masalah Etika/Legal**| (Risiko ini tetap ada). | **Tinggi** | **Kritis** | **DISCLAIMER** yang sangat jelas. *System Prompt* LLM lokal harus diinstruksikan untuk *tidak pernah* memberikan perintah (imperatif). |

## 6. Kesimpulan

Menggunakan model `qwen3-vl:235b-cloud` (lokal) yang dikombinasikan dengan **ChromaDB Cloud** (terkelola) adalah arsitektur hybrid yang sangat kuat. Ini memberikan kemampuan AI SOTA secara lokal sambil memindahkan beban manajemen database ke cloud.

Keberhasilan bergantung pada: (1) Ketersediaan **perangkat keras server multi-GPU**, dan (2) **Koneksi internet yang stabil** dari server tersebut ke ChromaDB Cloud.#   a i - c h a r t - a n a l y s t  
 