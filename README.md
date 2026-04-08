# SAP AI

Streamlit app for SAP ticket triage across `DEV`, `QA`, `TEST`, and `PROD`.

## Features

- Resolves common SAP support tickets using a structured runbook catalog
- Adds environment-aware guidance and guardrails for each landscape
- Adds system-aware and subsystem-aware guidance across major SAP platforms
- Supports a configurable SAP landscape registry so new systems and subsystems can be added without changing the core app
- Detects mixed tickets that combine multiple SAP issues and splits them into parallel workstreams
- Supports screenshot intake with image preprocessing, OCR, NLP signal extraction, and neural similarity matching
- Suggests relevant SAP T-codes, checks, fixes, and escalation conditions
- Supports local note matching and optional FAISS-based retrieval
- Supports local SAP router tuning so ticket-to-runbook matching can improve over time on your own machine
- Supports OpenAI-backed answer generation when `OPENAI_API_KEY` is configured
- Supports multiple open-source AI backends including Ollama, Open LLM API-compatible servers, and Hugging Face local models
- Can ingest SAP web content from official SAP domains into a local text corpus

## Files

- `ui.py`: Streamlit frontend
- `app.py`: Core SAP AI resolver
- `sap_ticket_catalog.py`: Structured SAP runbook catalog
- `sap_landscape.py`: Built-in SAP system and subsystem registry with auto-detection helpers
- `sap_landscape.example.json`: Example override file for customer-specific SAP systems and subsystems
- `sap_intelligence.py`: NLP, neural similarity, OCR, and image-preprocessing helpers for richer issue analysis
- `tools.py`: LangChain tool wrappers
- `agent.py`: Optional LangChain agent entry point
- `embed.py`: Build a local `sap_index` from text data
- `sap_data.py`: Export the public SAP BASIS dataset to `sap_dataset.txt`
- `sap_web_ingest.py`: Pull SAP web content from allowed SAP URLs into `sap_web_data.txt`
- `sap_sources.txt`: Curated SAP internet sources used by the web ingestion script
- `rag_sap.py`: Lightweight Ollama + FAISS CLI example
- `sap_training.py`: Time-budgeted trainer for a lightweight SAP ticket router used to improve runbook matching
- `packages.txt`: Linux system packages for cloud deployment, including Tesseract OCR

## Run Locally

```powershell
python -m streamlit run ui.py --browser.gatherUsageStats false
```

Or on Windows PowerShell:

```powershell
.\run_app.ps1
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
- `open_llm`: LM Studio, vLLM, LocalAI, llama.cpp server, or similar endpoints that expose an OpenAI-style `/v1` API
- `hf_local`: local Hugging Face Transformers model running on the same machine

Optional:

```powershell
$env:OPEN_SOURCE_BACKEND="auto"
$env:OPEN_LLM_API_BASE_URL="http://127.0.0.1:1234/v1"
$env:OPEN_LLM_MODEL="meta-llama-3.1-8b-instruct"
$env:OPEN_LLM_API_KEY="your_open_llm_api_key_here"
$env:OPEN_LLM_TIMEOUT_SECONDS="25"
$env:HF_LOCAL_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
$env:HF_LOCAL_MAX_NEW_TOKENS="220"
```

Notes:

- Use `open_source` in the UI to prefer open-source AI over OpenAI.
- Use `open_llm` directly when you want to force an API-key-based OpenAI-compatible endpoint.
- Use `hf_local` only when the machine has a supported runtime such as PyTorch installed.
- Set `HF_TOKEN` if your Hugging Face model or download path needs authentication.
- The older `OPEN_SOURCE_API_*` variable names still work for backward compatibility.
- This integration covers the major open-source backend styles, not literally every framework-specific serving stack.

Vector retrieval is optional. To enable FAISS context enrichment, set:

```powershell
$env:ENABLE_VECTOR_CONTEXT="1"
```

## SAP Landscape Integration

The app now supports environment plus system plus subsystem routing.

Built-in system coverage includes:

- SAP ECC
- SAP S/4HANA
- SAP BW/4HANA
- SAP Fiori and Gateway
- SAP PI/PO
- SAP Integration Suite
- SAP GRC
- SAP Solution Manager
- SAP HANA Database
- SAP MDG

To add customer-specific systems or subsystems:

1. Copy `sap_landscape.example.json` to `sap_landscape.json`
2. Add your system IDs, labels, aliases, integration points, and subsystem entries
3. Restart the app

`sap_landscape.json` is loaded automatically when present, so you can add custom hubs, satellite systems, or subsystem-specific ownership rules without editing `app.py`.

## NLP And Image Analysis

The app can now enrich a ticket with:

- NLP signal extraction from ticket text
- neural similarity matching against SAP runbooks, systems, and subsystems
- screenshot preprocessing for OCR
- OCR-based extraction of T-codes, status codes, HTTP errors, transport references, and users

Use it by uploading a screenshot in the chat form alongside the ticket text.

Notes:

- OCR uses `pytesseract`, so the Python package alone is not enough on Windows. Install the Tesseract OCR binary and make sure it is available on the machine path.
- If OCR is not available, the app still works and falls back to text-only analysis without crashing.

## Host On The Internet

This app is best hosted with Streamlit Community Cloud because it deploys directly from GitHub and runs the Python backend for you.

Deployment target:

- Repo: `Yaswanthh13426789155/SAP-AI`
- Branch: `main`
- Main file: `ui.py`
- Python version: `3.12`

Cloud setup steps:

1. Open Streamlit Community Cloud and create a new app from `Yaswanthh13426789155/SAP-AI`.
2. Select branch `main` and main file path `ui.py`.
3. In Advanced settings, choose Python `3.12`.
4. Keep `packages.txt` in the repo so the cloud build installs Tesseract OCR for screenshots.
5. In the app Secrets settings, add the keys you want to use.
6. Deploy to get a public `*.streamlit.app` URL.

Recommended cloud secrets:

```toml
OPENAI_API_KEY="your_openai_api_key_here"
OPENAI_MODEL="gpt-4.1-mini"
OPEN_SOURCE_BACKEND="auto"
OPEN_LLM_API_BASE_URL=""
OPEN_LLM_MODEL=""
OPEN_LLM_API_KEY=""
HF_TOKEN=""
HF_LOCAL_MODEL=""
ENABLE_VECTOR_CONTEXT="1"
```

Important cloud notes:

- `OLLAMA_BASE_URL="http://127.0.0.1:11434"` will not work on Streamlit Community Cloud because `localhost` there is the cloud container, not your PC.
- Use OpenAI, a publicly reachable OpenAI-compatible endpoint, or keep the built-in SAP rules mode for cloud deployment.
- `packages.txt` installs Tesseract so OCR can work remotely on Linux without manual server setup.

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

## Optional 8-Hour Tuning Run

This project can now tune a lightweight SAP ticket router on the local machine.
The tuned router is used to improve runbook selection for ambiguous tickets before answer generation.

Run it with an 8-hour wall-clock budget:

```powershell
python sap_training.py --time-budget-hours 8
```

Notes:

- On this machine class of hardware, this tunes the local SAP router, not OpenAI or Ollama themselves.
- The training job writes progress to `.cache/sap_training/status.json`.
- The best checkpoint is saved under `.cache/sap_training/sap_router/`.
- If you want to leave it running in the background on Windows:

```powershell
Start-Process python -WorkingDirectory . -ArgumentList 'sap_training.py','--time-budget-hours','8'
```

## Notes

- The app works without a FAISS index by using the local runbook catalog and `sap_tickets.txt`.
- The app can also use `sap_dataset.txt` and `sap_web_data.txt` when those files are present.
- The app can load a local `sap_landscape.json` file to extend the built-in SAP system and subsystem catalog for your landscape.
- This project uses retrieval and grounded generation, not OpenAI fine-tuning. For SAP ticket support, that is safer and easier to update than retraining a model every time the source data changes.
- The local tuning workflow improves SAP runbook routing with your local corpus and is designed for CPU-only machines.
- For best results, paste full ticket text including the environment, error, T-code, and business impact.
