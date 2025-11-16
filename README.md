# AI Chart Analyst

AI Chart Analyst adalah aplikasi web lengkap yang menggunakan arsitektur RAG (Retrieval-Augmented Generation) multimodal. Aplikasi ini memungkinkan pengguna untuk mengunggah gambar chart keuangan dan mengajukan pertanyaan, atau melakukan chat berbasis teks tentang konsep analisis teknikal.

##  Fitur Utama

- **Analisis Visual**: Menggunakan model `llava:7b` (atau model Visi lainnya) untuk mengekstrak pola, level support/resistance, dan indikator dari gambar chart.
- **Basis Pengetahuan (RAG)**: Terhubung ke database vektor (ChromaDB Cloud) yang berisi pengetahuan analisis teknikal untuk memberikan konteks pada temuan visual.
- **Chat Multimodal**: Dapat menjawab pertanyaan *tentang gambar* yang diunggah.
- **Chat Teks-Saja**: Dapat menjawab pertanyaan umum tentang trading (misalnya, "Apa itu RSI?") menggunakan RAG.
- **Backend Asinkron**: Dibangun dengan FastAPI (Python) untuk menangani permintaan AI secara efisien.
- **Frontend Reaktif**: Antarmuka chatbot yang responsif dibangun dengan React (Vite).
- **Streaming Respons**: Jawaban AI di-stream kata per kata untuk pengalaman pengguna yang lebih baik.

## Tumpukan Teknologi

- **Backend**: Python, FastAPI, LangChain
- **Frontend**: React (Vite), JavaScript, CSS
- **Model AI**: Ollama (menjalankan `llava:7b` atau model Visi lainnya)
- **Database Vektor**: ChromaDB Cloud
- **Model Embedding**: `all-MiniLM-L6-v2` (dijalankan secara lokal oleh LangChain)

## Struktur Proyek
/ai-chart-analyst/ ├── /backend/ # Server FastAPI (Python) │ ├── /app/ │ │ ├── /api/ # Endpoint API (chat.py) │ │ ├── /core/ # Konfigurasi (config.py) │ │ ├── /models/ # Skema Pydantic (schema.py) │ │ └── /services/ # Logika bisnis (llm_service.py, rag_service.py) │ ├── .env # Kredensial (HARUS DIBUAT) │ └── requirements.txt │ ├── /frontend/ # Aplikasi Chatbot (React) │ ├── /src/ │ │ ├── App.jsx # Komponen UI utama │ │ ├── App.css # Styling untuk App.jsx │ │ └── index.css # Styling global │ └── package.json │ ├── /notebooks/ │ └── 01_Data_Ingestion.ipynb # Skrip untuk mengisi ChromaDB │ └── README.md # Dokumentasi ini

## Instalasi & Menjalankan

### Persyaratan Awal

1.  **Python 3.10+**
2.  **Node.js 20+**
3.  **Ollama**: Pastikan Ollama sudah terinstal dan berjalan.
4.  **Tarik Model Ollama**:
    ```bash
    ollama pull llava:7b
    ```
5.  **Akun ChromaDB Cloud**: Dapatkan API Key, Tenant, dan nama Database dari ChromaDB.

### 1. Backend (`http://localhost:8000`)

Buka terminal pertama:

```bash
# 1. Navigasi ke folder backend
cd backend

# 2. Buat dan aktifkan virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Buat file .env
# Salin kredensial Anda ke dalam file 'backend/.env' (lihat .env.example)
# Pastikan semua variabel (OLLAMA_BASE_URL, CHROMA_API_KEY, dll) sudah diisi.

# 5. JALANKAN SERVER BACKEND
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
