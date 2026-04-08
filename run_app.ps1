$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

python -m streamlit run ui.py --server.headless true --browser.gatherUsageStats false
