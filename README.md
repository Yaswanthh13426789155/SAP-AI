# SAP Ticket Resolver

Streamlit app for SAP ticket triage across `DEV`, `QA`, `TEST`, and `PROD`.

## Features

- Resolves common SAP support tickets using a structured runbook catalog
- Adds environment-aware guidance and guardrails for each landscape
- Suggests relevant SAP T-codes, checks, fixes, and escalation conditions
- Supports local note matching and optional FAISS-based retrieval
- Supports OpenAI-backed answer generation when `OPENAI_API_KEY` is configured
- Can ingest SAP web content from official SAP domains into a local text corpus

## Files

- `ui.py`: Streamlit frontend
- `app.py`: Core SAP ticket resolver
- `sap_ticket_catalog.py`: Structured SAP runbook catalog
- `tools.py`: LangChain tool wrappers
- `agent.py`: Optional LangChain agent entry point
- `embed.py`: Build a local `sap_index` from text data
- `sap_data.py`: Export the public SAP BASIS dataset to `sap_dataset.txt`
- `sap_web_ingest.py`: Pull SAP web content from allowed SAP URLs into `sap_web_data.txt`
- `sap_sources.txt`: Curated SAP internet sources used by the web ingestion script
- `rag_sap.py`: Lightweight Ollama + FAISS CLI example

## Run Locally

```powershell
python -m streamlit run ui.py --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`.

## OpenAI Setup

Set the API key as an environment variable before starting the app:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Or create a local `.env` file from `.env.example` and place the key there.

Optional:

```powershell
$env:OPENAI_MODEL="gpt-4.1-mini"
```

The app uses local SAP runbooks by default and automatically upgrades to OpenAI-backed answers when the key is present and the UI engine is set to `auto` or `openai`.

## Optional Knowledge Base Build

1. Export extra SAP data:

```powershell
python sap_data.py
```

2. Build the FAISS index:

```powershell
python embed.py
```

3. Pull SAP web content from internet sources:

```powershell
python sap_web_ingest.py
```

## Notes

- The app works without a FAISS index by using the local runbook catalog and `sap_tickets.txt`.
- The app can also use `sap_dataset.txt` and `sap_web_data.txt` when those files are present.
- This project uses retrieval and grounded generation, not OpenAI fine-tuning. For SAP ticket support, that is safer and easier to update than retraining a model every time the source data changes.
- For best results, paste full ticket text including the environment, error, T-code, and business impact.
