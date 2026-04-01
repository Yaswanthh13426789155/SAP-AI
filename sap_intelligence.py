import io
import os
import re
from functools import lru_cache
from pathlib import Path
from shutil import which

from dotenv import load_dotenv
from sap_landscape import get_sap_landscape
from sap_ticket_catalog import TICKET_CATALOG


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TCODE_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{1,4}[A-Z]?\b")
TRANSPORT_PATTERN = re.compile(r"\b[A-Z0-9]{3}K\d{6,}\b", re.IGNORECASE)
HTTP_PATTERN = re.compile(r"\bHTTP\s*[45]\d{2}\b", re.IGNORECASE)
STATUS_PATTERN = re.compile(r"\bstatus\s*\d{2,3}\b", re.IGNORECASE)
RC_PATTERN = re.compile(r"\bRC\s*\d+\b", re.IGNORECASE)
QUEUE_PATTERN = re.compile(r"\b(?:qRFC|tRFC|queue)\s+[A-Z0-9_/-]+\b", re.IGNORECASE)
IDOC_PATTERN = re.compile(r"\bIDOC\b[\s:#-]*([0-9]{6,20})?", re.IGNORECASE)
USER_PATTERN = re.compile(r"\buser\s+([A-Z0-9_.-]{3,30})\b", re.IGNORECASE)
DOCUMENT_PATTERN = re.compile(r"\b(?:document|invoice|delivery|workflow item|queue|job)\s*(?:number|id|name)?\s*[:#-]?\s*([A-Z0-9_/.-]{4,30})\b", re.IGNORECASE)
USER_STOPWORDS = {
    "gets",
    "cannot",
    "fails",
    "failed",
    "locked",
    "lock",
    "with",
    "after",
    "before",
    "when",
    "from",
}

DOMAIN_SIGNAL_MAP = {
    "Security": ["authorization", "access denied", "role", "su53", "locked user", "user locked"],
    "Integration": ["idoc", "queue", "qrfc", "trfc", "sm58", "smq1", "smq2", "middleware", "interface"],
    "Basis": ["transport", "rc 8", "dump", "st22", "job cancelled", "sm37", "instance", "spool"],
    "FI": ["invoice", "fb60", "fb08", "ob52", "posting", "f110", "company code"],
    "MM/SD": ["va01", "pricing", "condition", "migo", "material", "delivery", "stock"],
    "Fiori/Gateway": ["fiori", "launchpad", "odata", "http 500", "gateway", "/iwfnd/error_log"],
}


def pillow_is_available():
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def get_tesseract_executable():
    configured = str(os.getenv("TESSERACT_CMD", "")).strip()
    if configured and Path(configured).exists():
        return configured

    discovered = which("tesseract")
    if discovered:
        return discovered

    common_paths = [
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ]
    for candidate in common_paths:
        if candidate.exists():
            return str(candidate)
    return ""


def ocr_is_available():
    if not pillow_is_available():
        return False
    try:
        import pytesseract

        executable = get_tesseract_executable()
        if not executable:
            return False
        pytesseract.pytesseract.tesseract_cmd = executable
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def neural_nlp_is_available():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        return False
    return True


def normalize_whitespace(text):
    return " ".join(str(text or "").split())


def configure_huggingface_auth():
    token = str(
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def clean_ocr_text(text):
    cleaned = normalize_whitespace(text)
    replacements = {
        "HTTPS00": "HTTP 500",
        "AIWFND": "/IWFND",
        "AIWBEP": "/IWBEP",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"(?i)status[\$s]?1\b", "Status 51", cleaned)
    return cleaned


def safe_unique(items, limit=None):
    values = []
    seen = set()
    for item in items or []:
        normalized = normalize_whitespace(item).lower()
        if not normalized or normalized in seen:
            continue
        values.append(normalize_whitespace(item))
        seen.add(normalized)
        if limit and len(values) >= limit:
            break
    return values


@lru_cache(maxsize=1)
def load_sentence_model():
    from sentence_transformers import SentenceTransformer

    configure_huggingface_auth()
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def load_reference_items():
    items = []
    for ticket in TICKET_CATALOG:
        items.append(
            {
                "label": ticket["title"],
                "type": f"runbook/{ticket['area']}",
                "text": " ".join(
                    safe_unique(
                        [ticket["title"], ticket["root_cause"]]
                        + ticket.get("keywords", [])[:5]
                        + ticket.get("error_signals", [])[:4]
                        + ticket.get("symptoms", [])[:4]
                    )
                ),
            }
        )

    for system_id, profile in get_sap_landscape().items():
        items.append(
            {
                "label": profile.get("label", system_id),
                "type": "system",
                "text": " ".join(
                    safe_unique(
                        [profile.get("label", system_id), profile.get("type", "SAP system")]
                        + profile.get("aliases", [])[:5]
                        + profile.get("integration_points", [])[:4]
                    )
                ),
            }
        )
        for subsystem_id, subsystem in profile.get("subsystems", {}).items():
            items.append(
                {
                    "label": subsystem.get("label", subsystem_id),
                    "type": f"subsystem/{profile.get('label', system_id)}",
                    "text": " ".join(
                        safe_unique(
                            [subsystem.get("label", subsystem_id), subsystem.get("focus", "SAP subsystem")]
                            + subsystem.get("aliases", [])[:5]
                            + subsystem.get("integration_points", [])[:4]
                        )
                    ),
                }
            )
    return items


@lru_cache(maxsize=1)
def load_reference_vectors():
    model = load_sentence_model()
    items = load_reference_items()
    vectors = model.encode([item["text"] for item in items], normalize_embeddings=True)
    return items, vectors


def cosine_similarity(vector_a, vector_b):
    return sum(float(a) * float(b) for a, b in zip(vector_a, vector_b))


def semantic_matches(text, top_k=5):
    query = normalize_whitespace(text)
    if not query or not neural_nlp_is_available():
        return []

    try:
        model = load_sentence_model()
        items, vectors = load_reference_vectors()
        query_vector = model.encode([query], normalize_embeddings=True)[0]
    except Exception:
        return []

    scored = []
    for item, vector in zip(items, vectors):
        score = cosine_similarity(query_vector, vector)
        if score >= 0.3:
            type_bonus = 0.0
            if item["type"] == "system":
                type_bonus = 0.03
            elif item["type"].startswith("subsystem/"):
                type_bonus = 0.02
            scored.append(
                {
                    "label": item["label"],
                    "type": item["type"],
                    "score": round(float(score), 3),
                    "rank_score": round(float(score + type_bonus), 3),
                }
            )
    scored.sort(key=lambda entry: entry["rank_score"], reverse=True)

    selected = []
    counts = {"system": 0, "subsystem": 0, "runbook": 0}
    for entry in scored:
        entry_type = entry["type"]
        bucket = "runbook"
        if entry_type == "system":
            bucket = "system"
        elif entry_type.startswith("subsystem/"):
            bucket = "subsystem"

        if bucket == "system" and counts[bucket] >= 1:
            continue
        if bucket == "subsystem" and counts[bucket] >= 2:
            continue
        if bucket == "runbook" and counts[bucket] >= 2:
            continue

        entry = {key: value for key, value in entry.items() if key != "rank_score"}
        selected.append(entry)
        counts[bucket] += 1
        if len(selected) >= top_k:
            break

    return selected


def preprocess_issue_image(image_bytes):
    if not pillow_is_available():
        return None, ["Pillow is not installed, so image preprocessing is unavailable."]

    try:
        from PIL import Image, ImageFilter, ImageOps

        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        original_width, original_height = image.size
        image = ImageOps.grayscale(image)
        if max(image.size) < 1800:
            image = image.resize((image.width * 2, image.height * 2))
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=180, threshold=3))
        image = image.point(lambda value: 255 if value > 165 else 0)
        metadata = [
            f"Input image size: {original_width}x{original_height}",
            f"Preprocessed image size: {image.width}x{image.height}",
            "Applied grayscale, autocontrast, denoise, sharpen, and threshold steps for OCR.",
        ]
        return image, metadata
    except Exception as exc:
        return None, [f"Image preprocessing failed: {exc}"]


def extract_text_from_image(image_bytes):
    warnings = []
    image, preprocessing_notes = preprocess_issue_image(image_bytes)
    warnings.extend(preprocessing_notes)
    if image is None:
        return "", warnings

    if not ocr_is_available():
        warnings.append("OCR engine is not available. Install pytesseract and the Tesseract binary to read screenshots.")
        return "", warnings

    try:
        import pytesseract

        executable = get_tesseract_executable()
        if executable:
            pytesseract.pytesseract.tesseract_cmd = executable
        text = pytesseract.image_to_string(image, config="--psm 6")
        return clean_ocr_text(text), warnings
    except Exception as exc:
        warnings.append(f"OCR extraction failed: {exc}")
        return "", warnings


def extract_entities(text):
    source = str(text or "")
    upper_source = source.upper()
    tcode_matches = [
        match
        for match in TCODE_PATTERN.findall(upper_source)
        if not match.startswith(("HTTP", "HTTPS", "STATUS"))
    ]
    user_matches = [
        match
        for match in USER_PATTERN.findall(source)
        if match.lower() not in USER_STOPWORDS
    ]
    entities = {
        "tcodes": safe_unique(tcode_matches, limit=6),
        "transports": safe_unique(TRANSPORT_PATTERN.findall(source), limit=4),
        "http_codes": safe_unique([match.upper() for match in HTTP_PATTERN.findall(source)], limit=4),
        "status_codes": safe_unique([match.upper() for match in STATUS_PATTERN.findall(source)], limit=4),
        "return_codes": safe_unique([match.upper() for match in RC_PATTERN.findall(source)], limit=4),
        "queues": safe_unique(QUEUE_PATTERN.findall(source), limit=4),
        "idocs": safe_unique(
            [match.group(1) for match in IDOC_PATTERN.finditer(source) if match.group(1)],
            limit=4,
        ),
        "users": safe_unique(user_matches, limit=4),
        "objects": safe_unique(DOCUMENT_PATTERN.findall(source), limit=6),
    }
    return {key: value for key, value in entities.items() if value}


def detect_domain_signals(text):
    lowered = str(text or "").lower()
    signals = []
    for domain, keywords in DOMAIN_SIGNAL_MAP.items():
        hits = [keyword for keyword in keywords if keyword in lowered]
        if hits:
            signals.append(
                {
                    "domain": domain,
                    "score": len(hits),
                    "signals": hits[:4],
                }
            )
    signals.sort(key=lambda item: item["score"], reverse=True)
    return signals[:4]


def summarize_analysis_context(analysis):
    summary_lines = []
    entities = analysis.get("entities", {})
    if entities.get("tcodes"):
        summary_lines.append(f"Detected T-codes: {', '.join(entities['tcodes'])}")
    if entities.get("transports"):
        summary_lines.append(f"Detected transports: {', '.join(entities['transports'])}")
    if entities.get("status_codes"):
        summary_lines.append(f"Detected statuses: {', '.join(entities['status_codes'])}")
    if entities.get("http_codes"):
        summary_lines.append(f"Detected HTTP errors: {', '.join(entities['http_codes'])}")
    if entities.get("users"):
        summary_lines.append(f"Detected users: {', '.join(entities['users'])}")
    if entities.get("idocs"):
        summary_lines.append(f"Detected IDocs: {', '.join(entities['idocs'])}")

    domain_signals = analysis.get("domain_signals", [])
    if domain_signals:
        top_domain = domain_signals[0]
        summary_lines.append(
            f"Top NLP domain signal: {top_domain['domain']} ({', '.join(top_domain['signals'])})"
        )

    semantic = analysis.get("semantic_matches", [])
    if semantic:
        top_matches = ", ".join(
            f"{item['label']} [{item['type']}, {item['score']}]"
            for item in semantic[:3]
        )
        summary_lines.append(f"Top neural matches: {top_matches}")

    return summary_lines[:6]


def build_resolver_evidence_text(analysis):
    evidence_lines = []
    if analysis.get("ocr_text"):
        evidence_lines.append(f"OCR evidence: {analysis['ocr_text'][:800]}")
    for line in analysis.get("summary_lines", []):
        evidence_lines.append(line)
    return "\n".join(evidence_lines[:8]).strip()


def analyze_issue_evidence(text="", image_bytes=None, filename=None):
    base_text = normalize_whitespace(text)
    image_findings = []
    warnings = []
    ocr_text = ""

    if image_bytes:
        ocr_text, image_notes = extract_text_from_image(image_bytes)
        image_findings.extend(image_notes)
        if filename:
            image_findings.insert(0, f"Attached image: {filename}")
        if ocr_text:
            image_findings.append(f"Extracted OCR text length: {len(ocr_text)} characters")

    combined_text = "\n".join(part for part in [base_text, ocr_text] if part).strip()
    entities = extract_entities(combined_text)
    domain_signals = detect_domain_signals(combined_text)
    semantic = semantic_matches(combined_text) if combined_text else []

    analysis = {
        "original_text": base_text,
        "ocr_text": ocr_text,
        "combined_text": combined_text,
        "entities": entities,
        "domain_signals": domain_signals,
        "semantic_matches": semantic,
        "image_findings": safe_unique(image_findings, limit=8),
        "warnings": safe_unique(warnings, limit=6),
    }
    analysis["summary_lines"] = summarize_analysis_context(analysis)
    analysis["resolver_evidence"] = build_resolver_evidence_text(analysis)
    return analysis
