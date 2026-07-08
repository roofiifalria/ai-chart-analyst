
# AI Chart Analyst
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/558270dd-1b1e-420c-9a43-a3aac4aec9a2" />

# AI Chart Analyst

AI Chart Analyst is a multimodal Retrieval-Augmented Generation (RAG) application designed to analyze cryptocurrency charts using Vision Language Models, vector retrieval, and Large Language Models.

The system combines image understanding, technical analysis knowledge retrieval, and natural language generation to provide contextual chart analysis from uploaded cryptocurrency chart images.

## Features

* Upload cryptocurrency chart images
* Vision-based chart analysis
* Technical pattern extraction
* Support and resistance detection
* Retrieval-Augmented Generation (RAG)
* ChromaDB vector search
* Local LLM inference using Ollama
* Streaming AI responses
* Modern React-based user interface

---

## System Architecture

```text
User
  ↓
React Frontend
  ↓
FastAPI Backend
  ↓
Vision Model (Qwen VL)
  ↓
Chart Information Extraction
  ↓
ChromaDB Retrieval
  ↓
LLM Synthesis
  ↓
Final Analysis
```

---

## Technology Stack

### Frontend

* React
* JavaScript
* Vite

### Backend

* FastAPI
* Python

### AI Components

* Ollama
* LangChain
* ChromaDB
* Sentence Transformers
* Vision Language Model
* Large Language Model

---

## Project Structure

```text
RAG
├── backend
│   ├── app
│   ├── scripts
│   ├── model_cache
│   └── requirements.txt
│
├── frontend
│   ├── src
│   ├── public
│   └── package.json
│
└── README.md
```

---

## Prerequisites

Before running the project, make sure the following software is installed:

* Python 3.11+
* Node.js 18+
* Ollama
* Git

---

## Installation

### Clone Repository

```bash
git clone https://github.com/roofiifalria/ai-chart-analyst.git

cd ai-chart-analyst
```

---

### Backend Setup

```bash
cd backend

python -m venv .venv

.\.venv\Scripts\activate

pip install -r requirements.txt
```

Create `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434

GENERATIVE_MODEL=<your_model>

VISION_MODEL=<your_vision_model>

CHROMA_API_KEY=<your_key>
CHROMA_TENANT=<your_tenant>
CHROMA_DATABASE=<your_database>
CHROMA_COLLECTION_NAME=<your_collection>
```

---

### Frontend Setup

```bash
cd frontend

npm install
```

---

## Running the Application

### Terminal 1 — Start Ollama

```bash
ollama serve
```

---

### Terminal 2 — Start Backend

```bash
cd backend

.\.venv\Scripts\activate

uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

### Terminal 3 — Start Frontend

```bash
cd frontend

npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## Usage

1. Open the frontend application.
2. Upload a cryptocurrency chart image.
3. Enter a question or analysis request.
4. Submit the request.
5. Receive AI-generated chart analysis.

Example:

```text
Analyze this BTC chart and identify the trend.
```

```text
What are the major support and resistance levels?
```

```text
Explain the chart pattern shown in this image.
```

---

## API Endpoint

### Analyze Chart

```http
POST /api/analyze_chart
```

Parameters:

| Parameter  | Type   |
| ---------- | ------ |
| query      | string |
| image_file | file   |

---

## Future Improvements

* Real-time market data integration
* Additional technical indicators
* Multi-asset support
* Analysis history
* User authentication
* Performance optimization

---

## Author

Roofiif Alria

Institut Teknologi Sepuluh Nopember (ITS)

PT. Adma Digital Solusi Internship Project

```
```

