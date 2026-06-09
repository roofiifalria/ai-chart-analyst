
# AI Chart Analyst
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/558270dd-1b1e-420c-9a43-a3aac4aec9a2" />

AI Chart Analyst adalah aplikasi web multimodal berbasis arsitektur **RAG (Retrieval-Augmented Generation)**. Aplikasi ini memungkinkan pengguna mengunggah gambar chart keuangan untuk dianalisis, sekaligus menyediakan fitur chat berbasis teks mengenai berbagai konsep analisis teknikal.

---

## ✨ Fitur Utama

* **Analisis Visual Chart**
  Menggunakan model `qwen3-vl:235b-cloud` (atau model visi lainnya) untuk mengekstrak pola, support–resistance, dan indikator teknikal dari gambar.

* **RAG dengan Basis Pengetahuan**
  Terhubung dengan ChromaDB Cloud berisi materi analisis teknikal sehingga jawaban lebih akurat dan kontekstual.

* **Chat Multimodal**
  Menjawab pertanyaan yang berkaitan dengan gambar chart yang diunggah.

* **Chat Teks Saja**
  Mendukung percakapan seputar konsep trading seperti “Apa itu RSI?” atau “Bagaimana membaca candlestick?”.

* **Backend Asinkron**
  Dibangun dengan **FastAPI** untuk menangani permintaan secara cepat dan efisien.

* **Frontend Interaktif**
  Dibangun dengan **React (Vite)** dengan UI chatbot yang responsif.

* **Streaming Respons**
  Jawaban AI dikirim secara streaming untuk pengalaman percakapan yang lebih natural.

---

## 🧩 Tumpukan Teknologi

* **Backend**: Python, FastAPI, LangChain
* **Frontend**: React (Vite), JavaScript, CSS
* **Model AI**: Ollama (`qwen3-vl:235b-cloud` atau model visi lain)
* **Database Vektor**: ChromaDB Cloud
* **Embedding Model**: `all-MiniLM-L6-v2` (dijalankan lokal via LangChain)

---

## 📁 Struktur Proyek

```
/ai-chart-analyst/
│
├── /backend/                   # Server FastAPI
│   ├── /app/
│   │   ├── /api/               # Endpoint API (chat.py)
│   │   ├── /core/              # Konfigurasi (config.py)
│   │   ├── /models/            # Skema Pydantic (schema.py)
│   │   └── /services/          # Logika bisnis (llm_service.py, rag_service.py)
│   ├── .env                    # Kredensial (WAJIB dibuat)
│   └── requirements.txt
│
├── /frontend/                  # Aplikasi React
│   ├── /src/
│   │   ├── App.jsx             # Komponen UI utama
│   │   ├── App.css             # Styling App.jsx
│   │   └── index.css           # Styling global
│   └── package.json
│
├── /notebooks/
│   └── 01_Data_Ingestion.ipynb # Script untuk mengisi ChromaDB
│
└── README.md
```

---

## 🚀 Instalasi & Menjalankan Proyek

### Persyaratan

1. **Python 3.10+**
2. **Node.js 20+**
3. **Ollama** (sudah terinstal dan berjalan)
4. Tarik model Ollama:

   ```bash
   ollama pull llava:7b
   ```
5. **Akun ChromaDB Cloud**
   Siapkan API Key, Tenant, dan nama Database.

---

## 1. Menjalankan Backend ([http://localhost:8000](http://localhost:8000))

Buka terminal pertama:

```bash
# 1. Masuk ke folder backend
cd backend

# 2. Buat & aktifkan virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Buat file .env
# Isi semua variabel (OLLAMA_BASE_URL, CHROMA_API_KEY, dll)
# berdasarkan template dari .env.example

# 5. Jalankan server backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

