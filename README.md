# WhatsApp Chat Q&A Agent (LangGraph)

A FastAPI web app that lets a user paste WhatsApp-exported text and ask questions about it. The LangGraph workflow has two explicit nodes:

1. `retrieve_messages`: ranks message lines/excerpts by keyword overlap with the question.
2. `answer_question`: asks Gemini to answer only from the retrieved excerpts and admit when evidence is missing.

## Local run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Set the Gemini API key:
- Windows PowerShell: `$env:GOOGLE_API_KEY="your_key_here"`
- macOS/Linux: `export GOOGLE_API_KEY="your_key_here"`

Run:

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## Render deploy

1. Push these files to a GitHub repository.
2. On Render, create a **Blueprint** and select that repository (it detects `render.yaml`), or create a Web Service manually.
3. Add `GOOGLE_API_KEY` in Render → Environment. Never commit the key.
4. Deploy. Check `/health` for `{"status":"ok"}`.

Build command: `pip install -r requirements.txt`
Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## How to get chat text

In WhatsApp, choose a chat → menu → More → Export chat → Without media. Open the exported `.txt` file and paste the text into the page. For very large exports, use a smaller relevant section.

## Notes

- This is not a direct WhatsApp integration; it analyzes text the user pastes.
- Chat text is sent to Google Gemini for processing. Only submit conversations you have permission to share and avoid sensitive data.
- Retrieval is intentionally simple lexical matching for this quick prototype; questions phrased very differently from the messages may miss relevant excerpts. A stronger version could use embeddings/vector search and persistent conversation sessions.
- If `gemini-2.5-flash` is unavailable for your API key, change `GEMINI_MODEL` in the environment to a model enabled for your account.
