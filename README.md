# WM (Waste Management Inc.) — OTPP Tech Assessment

## Objective

Two complementary analyses on Waste Management Inc. (NYSE: WM):

1. **AI Application** — End-to-end pipeline that scrapes WM investor-relations press releases, extracts structured annual revenue and free-cash-flow guidance using a fine-tuned local LLM (LLaMA 3.2), compares guidance against SEC-filed actuals, and surfaces revisions and misses through an interactive RAG Q&A agent over earnings call transcripts.

2. **Predictive Model** — Regime-aware XGBoost classifier trained on a cross-sectional panel of WM sector peers to predict 3-day directional price moves (±2% threshold), with SHAP-based feature interpretation and signal threshold selection via Precision-Recall curves.

---

## Data Sources

| Source | Content | Tool |
|---|---|---|
| WM Investor Relations | Quarterly earnings press release PDFs | `playwright` (web scraper) |
| SEC EDGAR | 10-K and 10-Q filings (revenue, FCF) | `edgartools` |
| Earnings call transcripts | Speaker-level Q&A and prepared remarks | `defeatbeta-api` |
| Yahoo Finance | Daily OHLCV prices for WM + sector peers + macro ETFs | `yfinance` |

---

## Directory Structure

```
├── data/
│   ├── raw/
│   │   ├── news_releases/      # downloaded earnings press release PDFs
│   │   ├── filings/            # 10-K and 10-Q plain text (10-K/, 10-Q/)
│   │   └── transcripts/        # earnings call transcripts as plain text
│   ├── interim/                # intermediate outputs (local LLM guidance, financials)
│   ├── processed/              # final cleaned datasets
│   └── training/               # fine-tuning JSONL splits (train/val/test)
├── notebooks/
│   ├── Analysis_AI_application.ipynb   # AI pipeline: scrape → extract → analyse → RAG
│   ├── Predictive_Model.ipynb          # ML pipeline: features → XGBoost → SHAP
│   └── Finetune_Guidance_Extraction.ipynb  # LLaMA 3.2 fine-tuning on Colab (GPU)
├── src/
│   ├── config.py                   # project-wide path and ticker constants
│   ├── utils.py                    # shared logging and filesystem utilities
│   ├── web_scraper.py              # scraper for WM IR press release PDFs
│   ├── pdf_extractor.py            # PDF section extractor (outlook/guidance sections)
│   ├── local_guidance_extractor.py # structured guidance extraction via local Ollama LLM
│   ├── generate_training_data.py   # synthetic + real fine-tuning data generator
│   ├── financials_extractor.py     # revenue and FCF from SEC filings via edgartools
│   ├── filling_collector.py        # 10-K and 10-Q plain-text downloader
│   ├── transcripts_collector.py    # earnings call transcript downloader
│   ├── earning_release_vector.py   # ChromaDB vector store for transcript RAG
│   ├── local_agent_rag.py          # RAG Q&A agent over transcripts (CLI)
│   ├── analysis.py                 # guidance normalization, target bands, revision table
│   ├── plotting.py                 # revenue guidance vs actual chart utilities
├── model/                          # fine-tuned GGUF model weights (Ollama)
├── outputs/                        # figures, tables, walk-forward predictions
├── chrome_langchain_db/            # persisted ChromaDB vector store
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama (required for local LLM and embeddings)
```bash
ollama serve
ollama pull llama3.2
ollama pull mxbai-embed-large
# Load the fine-tuned guidance model (after fine-tuning step)
ollama create wm-guidance -f model/Modelfile
```

### 3. AI Application notebook
Open `notebooks/Analysis_AI_application.ipynb` and run cells in order:
- Scrapes press release PDFs → extracts outlook sections → runs local LLM extraction
- Downloads SEC financials → normalises guidance → builds target bands and revision table
- Launches interactive RAG Q&A over earnings call transcripts

### 4. Predictive Model notebook
Open `notebooks/Predictive_Model.ipynb` and run cells in order:
- Downloads market data for WM + sector peers via Yahoo Finance
- Engineers 50+ technical, macro, and volume features
- Trains regime-aware XGBoost models on cross-sectional panel
- Evaluates with AP-lift metrics, PR curves, and SHAP plots

### 5. Fine-tune the guidance LLM (optional, requires GPU)
Upload `data/training/*.jsonl` to Google Colab and run `notebooks/Finetune_Guidance_Extraction.ipynb`.
Export the GGUF model and register it with Ollama as `wm-guidance`.


