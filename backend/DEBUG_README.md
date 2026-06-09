Debugging notes and quick tests for `analyze_chart` endpoint

1) Restart the backend (so code changes take effect):

PowerShell (recommended):

cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1  # if you use venv
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

If `uvicorn` is not available on your path, install it (`pip install uvicorn[standard]`), or use the correct venv.

2) Quick curl tests (PowerShell caveat: `curl` maps to Invoke-WebRequest). Use `curl.exe` if available:

# curl example (Windows PowerShell, use curl.exe)
curl.exe -v -X POST "http://127.0.0.1:8000/api/analyze_chart" -F "query=testing-from-curl" -F "history=[{\"role\":\"user\",\"content\":\"hello\"}]"

# Or use the provided Python test script
cd backend
python scripts/test_post.py

3) Logs and where to look

- A small per-request receipt file is written to: backend/logs/chat/request_received_YYYYMMDD_HHMMSS.json
- The server logger writes to: backend/logs/chat/server.log (if uvicorn runs and the server picks up the updated code)
- Each completed chat session is saved to: backend/logs/chat/chat_YYYYMMDD_HHMMSS.json

4) What we changed

- Frontend: `frontend/src/App.jsx` — fixed the history payload to include the new user message (use currentMessages) and added debug console logs for outgoing FormData.
- Backend: `backend/app/api/chat.py` — added structured logging to console & `logs/chat/server.log`, and writes a small `request_received` JSON when request arrives.

If the server does not show the new logger behavior, restart the backend (step #1) then re-run the tests above. The `request_received_*.json` file should appear immediately after each request is sent.