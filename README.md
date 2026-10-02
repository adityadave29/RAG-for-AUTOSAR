# AUTOSAR RAG Pipeline

A specialized, modular Retrieval-Augmented Generation (RAG) pipeline for **AUTOSAR** engineering documents and specifications.

---

## 📁 Project Structure

```
AUTOSAR-RAG/
├── docs/                     # Place your AUTOSAR PDFs / Markdown / Text specs here
├── data/
│   └── chroma_db/            # Local ChromaDB vector storage (auto-created)
├── src/
│   ├── config.py             # Central paths & model configurations
│   ├── parser.py             # PDF (PyMuPDF/pypdf) & text parser
│   ├── chunker.py            # Recursive section-aware chunker with metadata injection
│   ├── vector_store.py       # ChromaDB vector store manager
│   ├── ingest.py             # End-to-end ingestion pipeline
│   ├── retriever.py          # Vector retrieval & Cross-Encoder re-ranker (Next step)
│   └── generator.py          # Gemini LLM generation engine (Next step)
├── main.py                   # Main CLI tool
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variables template
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your AUTOSAR Documents
Drop any official AUTOSAR PDFs (SWS, SRS, EXP, TPS) or `.md`/`.txt` files into the `docs/` folder:
```bash
cp /path/to/AUTOSAR_SWS_CAN.pdf ./docs/
```

### 3. Run Document Ingestion
```bash
# Ingest all files in docs/ into local ChromaDB
python main.py ingest

# Or clear previous index and re-ingest fresh
python main.py ingest --clear

# Check current collection stats
python main.py status

# Run application
streamlit run app.py
```
