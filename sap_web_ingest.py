from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


SOURCE_FILE = Path("sap_sources.txt")
OUTPUT_FILE = Path("sap_web_data.txt")
DEFAULT_URLS = [
    "https://help.sap.com/docs/",
    "https://api.sap.com/",
]
ALLOWED_DOMAINS = {
    "help.sap.com",
    "community.sap.com",
    "api.sap.com",
}


def load_urls(source_file=SOURCE_FILE):
    if not source_file.exists():
        return DEFAULT_URLS

    urls = []
    for line in source_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        urls.append(value)

    return urls or DEFAULT_URLS


def is_allowed(url):
    hostname = urlparse(url).netloc.lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in ALLOWED_DOMAINS)


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "header", "footer"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    text = "\n".join(lines[:250])

    if title and title not in text:
        return f"{title}\n{text}".strip()

    return text


def fetch_url(url):
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "SAP AI/1.0"},
    )
    response.raise_for_status()
    return extract_text_from_html(response.text)


def build_web_dataset():
    documents = []
    errors = []

    for url in load_urls():
        if not is_allowed(url):
            errors.append(f"Skipped non-SAP domain: {url}")
            continue

        try:
            text = fetch_url(url)
        except Exception as exc:
            errors.append(f"Failed: {url} -> {exc}")
            continue

        if text:
            documents.append(f"Source: {url}\n{text}")

    OUTPUT_FILE.write_text("\n\n".join(documents), encoding="utf-8")
    return len(documents), errors


if __name__ == "__main__":
    count, errors = build_web_dataset()
    print(f"Wrote {count} web document(s) to {OUTPUT_FILE}")
    for error in errors:
        print(error)
