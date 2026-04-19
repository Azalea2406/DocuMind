# Document Intelligence System

An end-to-end AI pipeline that transforms unstructured PDFs into structured, explainable insights — with section-wise summaries, named entity extraction, clause-level risk classification, and a conversational Q&A interface.

Built as a portfolio project demonstrating PDF parsing, NLP, vector search, LLM prompting, and full-stack Python.

---

## What it does

Upload any PDF — contract, report, policy, or agreement — and the system produces:

- **Executive summary** with key points and obligations
- **Section-wise breakdown** with individual summaries
- **Named entity extraction** — parties, dates, money, legal references
- **Risk analysis** — clause classification with Safe / Review / Critical labels and 0–100 score
- **RAG-powered Q&A** — ask questions, get answers cited to specific sections
- **JSON export** — full structured report download

---

## Tech stack

| Layer | Tool |
|---|---|
| PDF extraction | pdfplumber + PyMuPDF (fallback) |
| Structure parsing | Regex + spaCy |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) — local, free |
| Vector search | FAISS |
| LLM | Groq (llama-3.1-8b-instant) — free API |
| UI | Streamlit |
| Backend API | FastAPI |

**Total API cost: $0** — Groq is free, embeddings run locally.

---

## Project structure

```
doc-intelligence/
├── app.py                # Streamlit UI
├── main.py               # FastAPI backend (optional REST API)
├── parser.py             # PDF text extraction
├── structure_parser.py   # Heading and section detection
├── embedder.py           # Chunking + FAISS vector index
├── entity_extractor.py   # spaCy + custom regex entity extraction
├── analyzer.py           # Q&A, summaries, risk analysis (LLM layer)
├── quick_test.py         # Test script without a real PDF
├── requirements.txt
├── .env                  # API keys (not committed)
└── .gitignore
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/doc-intelligence.git
cd doc-intelligence
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first run will also download the `all-MiniLM-L6-v2` embedding model (~90MB) automatically.

### 4. Set up environment variables

Create a `.env` file in the project root:

```
USE_GROQ=true
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Optional: Run as a REST API

```bash
uvicorn main:app --reload
```

Endpoints:
- `POST /upload` — upload a PDF, returns structured JSON
- `GET /search/{file_id}?q=your+query` — semantic search over a processed document

---

## How it works

```
PDF Upload
    │
    ▼
parser.py          → extracts raw text page by page (pdfplumber → PyMuPDF fallback)
    │
    ▼
structure_parser.py → detects headings, splits into sections, runs spaCy entity extraction
    │
    ├──────────────────────────┐
    ▼                          ▼
embedder.py              entity_extractor.py
chunks + FAISS index     parties, dates, money, clause refs
    │                          │
    └──────────┬───────────────┘
               ▼
           analyzer.py
           ├── Q&A engine     (RAG: retrieve top-k chunks → LLM answer)
           ├── Summarizer     (section summaries → executive overview)
           └── Risk analyser  (clause classification → score + label)
               │
               ▼
           app.py  →  Streamlit UI  +  JSON export
```

---

## Usage

1. Open the app at `http://localhost:8501`
2. Upload a PDF using the sidebar
3. Click **Analyze**
4. Explore the five tabs:
   - **Overview** — executive summary, risk score, key points
   - **Sections** — section-by-section breakdown with risk flags
   - **Risk** — critical flags and risk breakdown bar chart
   - **Entities** — extracted parties, dates, money, legal references
   - **Ask a Question** — chat interface for document Q&A
5. Download the full report as JSON

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `USE_GROQ` | Yes | Set to `true` to use Groq |
| `GROQ_API_KEY` | Yes (if Groq) | Free key from console.groq.com |
| `OPENAI_API_KEY` | Only if not using Groq | OpenAI key |

---

## Limitations

- Scanned PDFs (image-only) are not supported — text must be digitally embedded
- Very large documents (100+ pages) may be slow on the free Groq tier
- Risk analysis is LLM-based and should not be used as a substitute for legal review

---

## Future improvements

- [ ] OCR support for scanned PDFs via pytesseract
- [ ] Multi-document comparison
- [ ] Clause-level redlining suggestions
- [ ] Export to Word / PDF report
- [ ] User authentication for multi-user deployment

---

## License

MIT License — free to use, modify, and distribute.

