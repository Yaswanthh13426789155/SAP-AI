# SAP AI

Streamlit app for SAP ticket triage across `DEV`, `QA`, `TEST`, and `PROD`.

## Features

- Resolves common SAP support tickets using a structured runbook catalog
- Adds environment-aware guidance and guardrails for each landscape
- Suggests relevant SAP T-codes, checks, fixes, and escalation conditions
- Supports local note matching and optional FAISS-based retrieval
- Supports OpenAI-backed answer generation when `OPENAI_API_KEY` is configured
- Supports multiple open-source AI backends including Ollama, OpenAI-compatible local servers, and Hugging Face local models
- Can ingest SAP web content from official SAP domains into a local text corpus

## Files

- `ui.py`: Streamlit frontend
- `app.py`: Core SAP AI resolver
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

For Streamlit Community Cloud, add the same values in the app Secrets settings instead of committing them to Git:

```toml
OPENAI_API_KEY="your_openai_api_key_here"
OPENAI_MODEL="gpt-4.1-mini"
```

If OpenAI is configured but the project has no quota or is rate limited, the app falls back to the local SAP runbooks and explains that in the response instead of failing with a traceback.

## Ollama Setup

If you have Ollama running locally, the app can use it through the built-in HTTP API.

Optional:

```powershell
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="llama3"
$env:OPEN_SOURCE_BACKEND="ollama"
```

In the UI, choose `ollama` to force the local model.
In the current app behavior, `auto` uses OpenAI when configured, otherwise any available open-source backend, and otherwise the built-in SAP runbooks. Choose `ollama` when you explicitly want the local Ollama model.

Optional:

```powershell
$env:OLLAMA_TIMEOUT_SECONDS="70"
```

If you host the app remotely, `OLLAMA_BASE_URL` must point to an Ollama endpoint reachable from that server. `localhost` only works when Ollama runs on the same machine as the app.

## Open-Source AI Setup

The new `open_source` engine can route across the most common open-source deployment styles:

- `ollama`: local Ollama server
- `openai_compatible`: LM Studio, vLLM, LocalAI, llama.cpp server, or similar endpoints that expose an OpenAI-style `/v1` API
- `hf_local`: local Hugging Face Transformers model running on the same machine

Optional:

```powershell
$env:OPEN_SOURCE_BACKEND="auto"
$env:OPEN_SOURCE_API_BASE_URL="http://127.0.0.1:1234/v1"
$env:OPEN_SOURCE_API_MODEL="meta-llama-3.1-8b-instruct"
$env:OPEN_SOURCE_API_KEY="not-needed"
$env:OPEN_SOURCE_API_TIMEOUT_SECONDS="25"
$env:HF_LOCAL_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
$env:HF_LOCAL_MAX_NEW_TOKENS="220"
```

Notes:

- Use `open_source` in the UI to prefer open-source AI over OpenAI.
- Use `openai_compatible` directly when you want to force a local API endpoint such as LM Studio.
- Use `hf_local` only when the machine has a supported runtime such as PyTorch installed.
- This integration covers the major open-source backend styles, not literally every framework-specific serving stack.

Vector retrieval is optional. To enable FAISS context enrichment, set:

```powershell
$env:ENABLE_VECTOR_CONTEXT="1"
```

## Host On The Internet

This app is best hosted with Streamlit Community Cloud because it deploys directly from GitHub and runs the Python backend for you.

1. Push the repo to GitHub.
2. In Streamlit Community Cloud, create a new app from `Yaswanthh13426789155/SAP-AI`.
3. Select branch `main` and main file path `ui.py`.
4. In the app Secrets settings, add `OPENAI_API_KEY` and optional `OPENAI_MODEL`.
5. Deploy to get a public `*.streamlit.app` URL.

GitHub Pages is not a fit for this project because it is designed for static sites, while this app needs a live Python/Streamlit backend.

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
