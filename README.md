# BookChat — PDF RAG Assistant

Upload a PDF book and ask questions through a Streamlit chat interface.

## Stack
- Streamlit UI
- PyPDFLoader and RecursiveCharacterTextSplitter
- Local Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- ChromaDB vector search
- Groq answer generation

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Run:
```powershell
streamlit run app.py
```

On first run, the embedding model is downloaded. Upload a text-based PDF, click **Process book**, then ask questions. The current book is indexed in an in-memory Chroma collection. Processing another book replaces the active book in this browser session.

Scanned PDFs may need OCR. Never commit `.env` or your API key to GitHub. `create_database.py` is optional and supports the original fixed-PDF workflow.
