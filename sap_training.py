import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
from functools import lru_cache
import json
import random
import re
import time
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

from sap_ticket_catalog import TICKET_CATALOG


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / ".cache" / "sap_training"
MODEL_DIR = TRAINING_DIR / "sap_router"
STATUS_PATH = TRAINING_DIR / "status.json"
LOG_PATH = TRAINING_DIR / "events.jsonl"
CHECKPOINT_PATH = MODEL_DIR / "model.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"
SOURCE_FILES = [
    BASE_DIR / "sap_tickets.txt",
    BASE_DIR / "sap_dataset.txt",
    BASE_DIR / "sap_web_data.txt",
]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_/.-]+")
ENVIRONMENTS = ["DEV", "QA", "TEST", "PROD"]
DEFAULT_STATUS = {
    "state": "idle",
    "router_model_ready": False,
    "time_budget_hours": 0.0,
    "best_val_accuracy": 0.0,
    "best_val_macro_f1": 0.0,
    "examples_total": 0,
    "train_examples": 0,
    "val_examples": 0,
    "trials_completed": 0,
    "best_trial": 0,
    "updated_at": "",
    "started_at": "",
    "ended_at": "",
    "message": "No tuning run has been started yet.",
}


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_training_dirs():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def training_runtime_available():
    return torch is not None


def normalize_text(text):
    return " ".join(str(text or "").split())


def tokenize(text):
    normalized = normalize_text(text).lower()
    return TOKEN_PATTERN.findall(normalized)


def clip_text(text, limit=700):
    compact = normalize_text(text)
    if len(compact) <= limit:
        return compact
    shortened = compact[:limit].rsplit(" ", 1)[0].strip()
    return f"{shortened}..."


def read_chunks(path):
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return [clip_text(chunk) for chunk in content.split("\n\n") if normalize_text(chunk)]


def safe_unique(items):
    values = []
    seen = set()
    for item in items or []:
        normalized = normalize_text(item).lower()
        if not normalized or normalized in seen:
            continue
        values.append(normalize_text(item))
        seen.add(normalized)
    return values


def build_ticket_profiles():
    profiles = []
    for ticket_index, ticket in enumerate(TICKET_CATALOG):
        phrases = safe_unique(
            [ticket["title"], ticket["root_cause"], ticket["area"]]
            + ticket.get("keywords", [])
            + ticket.get("error_signals", [])
            + ticket.get("symptoms", [])
            + ticket.get("tcodes", [])
        )
        term_counter = Counter()
        for phrase in phrases:
            for token in tokenize(phrase):
                if len(token) >= 2:
                    term_counter[token] += 1
        profiles.append(
            {
                "ticket_index": ticket_index,
                "title": ticket["title"],
                "area": ticket["area"],
                "phrases": [phrase.lower() for phrase in phrases if phrase],
                "terms": set(term_counter),
                "tcodes": {tcode.lower() for tcode in ticket.get("tcodes", [])},
            }
        )
    return profiles


def build_synthetic_examples(ticket, ticket_index):
    texts = []
    title = ticket["title"]
    area = ticket["area"]
    root_cause = ticket["root_cause"]
    error_signals = ticket.get("error_signals", [])[:3]
    symptoms = ticket.get("symptoms", [])[:3]
    keywords = ticket.get("keywords", [])[:4]
    tcodes = ticket.get("tcodes", [])[:3]
    first_signal = error_signals[0] if error_signals else title
    first_symptom = symptoms[0] if symptoms else root_cause
    first_tcode = tcodes[0] if tcodes else "SAP"
    first_check = ticket.get("checks", ["Capture the exact error text."])[0]
    first_fix = ticket.get("resolution", ["Apply the safest validated fix and retest."])[0]

    for environment in ENVIRONMENTS:
        texts.extend(
            [
                f"{environment} SAP ticket: {title}",
                f"{environment} issue: {first_signal}",
                f"{environment} {first_symptom} in {first_tcode}",
                f"{environment} {area} incident: {title}. Need fix steps.",
                f"How to fix {title} in {environment} SAP system",
                f"{environment} user reports {first_signal}. Check {first_tcode}.",
            ]
        )

    texts.extend(
        [
            f"{title}. Root cause: {root_cause}",
            f"{title}. First check: {first_check}",
            f"{title}. First fix: {first_fix}",
            f"{area} issue. {first_signal}. {first_symptom}.",
            f"{title}. Keywords: {', '.join(keywords)}",
            f"{title}. T-codes: {', '.join(tcodes)}",
        ]
    )

    for signal in error_signals:
        texts.append(f"{signal} while using {first_tcode}. Likely SAP issue.")
        texts.append(f"{title}. Error text: {signal}")
    for symptom in symptoms:
        texts.append(f"SAP symptom: {symptom}. Likely incident: {title}")
    for keyword in keywords:
        texts.append(f"{keyword} problem in SAP. Route to {title}")

    examples = []
    seen = set()
    for text in texts:
        normalized = normalize_text(text)
        if not normalized or normalized.lower() in seen:
            continue
        examples.append(
            {
                "text": normalized,
                "ticket_index": ticket_index,
                "source": "synthetic",
            }
        )
        seen.add(normalized.lower())
    return examples


def weak_label_chunk(chunk, profiles):
    text = normalize_text(chunk)
    if not text:
        return None

    lowered = text.lower()
    tokens = set(tokenize(text))
    scored = []
    for profile in profiles:
        overlap = len(tokens.intersection(profile["terms"]))
        phrase_hits = sum(1 for phrase in profile["phrases"][:8] if phrase in lowered)
        tcode_hits = sum(1 for tcode in profile["tcodes"] if tcode in lowered)
        score = overlap + (phrase_hits * 2) + (tcode_hits * 3)
        if profile["area"].lower() in lowered:
            score += 1
        scored.append((score, profile["ticket_index"]))

    scored.sort(reverse=True)
    if not scored:
        return None

    best_score, best_ticket_index = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    if best_score < 6 or best_score < runner_up + 2:
        return None
    return best_ticket_index, best_score


def build_training_examples(max_weak_per_ticket=20):
    profiles = build_ticket_profiles()
    examples = []
    stats = Counter()

    for ticket_index, ticket in enumerate(TICKET_CATALOG):
        synthetic_rows = build_synthetic_examples(ticket, ticket_index)
        examples.extend(synthetic_rows)
        stats["synthetic_examples"] += len(synthetic_rows)

    weak_counts = Counter()
    for source_path in SOURCE_FILES:
        for chunk in read_chunks(source_path):
            weak_label = weak_label_chunk(chunk, profiles)
            if not weak_label:
                continue
            ticket_index, _ = weak_label
            if weak_counts[ticket_index] >= max_weak_per_ticket:
                continue
            examples.append(
                {
                    "text": chunk,
                    "ticket_index": ticket_index,
                    "source": source_path.name,
                }
            )
            weak_counts[ticket_index] += 1
            stats[f"weak_examples_{source_path.stem}"] += 1

    stats["tickets_covered"] = len({row["ticket_index"] for row in examples})
    stats["examples_total"] = len(examples)
    return examples, dict(stats)


def stratified_split(examples, validation_ratio=0.2, seed=42):
    by_label = defaultdict(list)
    for item in examples:
        by_label[item["ticket_index"]].append(item)

    rng = random.Random(seed)
    train_rows = []
    val_rows = []
    for rows in by_label.values():
        rows = list(rows)
        rng.shuffle(rows)
        if len(rows) == 1:
            train_rows.extend(rows)
            continue
        val_count = max(1, int(round(len(rows) * validation_ratio)))
        if val_count >= len(rows):
            val_count = 1
        val_rows.extend(rows[:val_count])
        train_rows.extend(rows[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def build_vocab(texts, min_freq=2, max_size=12000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.most_common():
        if count < min_freq or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab):
    tokens = tokenize(text)
    if not tokens:
        return [vocab["<unk>"]]
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


class TicketRouterDataset(Dataset):
    def __init__(self, rows, vocab, label_to_id):
        self.rows = rows
        self.vocab = vocab
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "tokens": encode_text(row["text"], self.vocab),
            "label": self.label_to_id[row["ticket_index"]],
        }


class SAPTicketRouter(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_count, dropout):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, label_count)

    def forward(self, tokens, offsets):
        embedded = self.embedding(tokens, offsets)
        hidden = torch.relu(self.hidden(embedded))
        hidden = self.dropout(hidden)
        return self.output(hidden)


def collate_batch(rows):
    flat_tokens = []
    offsets = []
    labels = []
    cursor = 0
    for row in rows:
        token_ids = row["tokens"] or [1]
        flat_tokens.extend(token_ids)
        offsets.append(cursor)
        labels.append(row["label"])
        cursor += len(token_ids)

    return (
        torch.tensor(flat_tokens, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def set_seed(seed):
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def compute_macro_f1(predictions, labels, label_count):
    if not predictions:
        return 0.0

    f1_scores = []
    for label_id in range(label_count):
        true_positive = sum(
            1 for prediction, expected in zip(predictions, labels) if prediction == label_id and expected == label_id
        )
        false_positive = sum(
            1 for prediction, expected in zip(predictions, labels) if prediction == label_id and expected != label_id
        )
        false_negative = sum(
            1 for prediction, expected in zip(predictions, labels) if prediction != label_id and expected == label_id
        )

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * precision * recall) / (precision + recall))
    return sum(f1_scores) / len(f1_scores)


def evaluate_model(model, rows, vocab, label_to_id, batch_size=32):
    if not rows:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    dataset = TicketRouterDataset(rows, vocab, label_to_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    predictions = []
    expected = []
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for tokens, offsets, labels in loader:
            logits = model(tokens, offsets)
            predicted = torch.argmax(logits, dim=1)
            predictions.extend(predicted.tolist())
            expected.extend(labels.tolist())
            correct += int((predicted == labels).sum().item())
            total += int(labels.numel())

    accuracy = correct / total if total else 0.0
    macro_f1 = compute_macro_f1(predictions, expected, len(label_to_id))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def list_trial_configs(max_trials):
    base_grid = [
        {"embedding_dim": 64, "hidden_dim": 96, "dropout": 0.1, "lr": 0.002, "weight_decay": 0.0001, "batch_size": 16, "seed": 7},
        {"embedding_dim": 96, "hidden_dim": 128, "dropout": 0.15, "lr": 0.0015, "weight_decay": 0.0001, "batch_size": 16, "seed": 11},
        {"embedding_dim": 96, "hidden_dim": 160, "dropout": 0.2, "lr": 0.001, "weight_decay": 0.00005, "batch_size": 24, "seed": 17},
        {"embedding_dim": 128, "hidden_dim": 192, "dropout": 0.2, "lr": 0.001, "weight_decay": 0.0001, "batch_size": 24, "seed": 23},
        {"embedding_dim": 128, "hidden_dim": 224, "dropout": 0.25, "lr": 0.0008, "weight_decay": 0.0001, "batch_size": 32, "seed": 29},
    ]
    configs = []
    index = 0
    while len(configs) < max_trials:
        template = base_grid[index % len(base_grid)].copy()
        template["seed"] += (index // len(base_grid)) * 31
        configs.append(template)
        index += 1
    return configs


def load_training_status():
    status = DEFAULT_STATUS.copy()
    if STATUS_PATH.exists():
        try:
            stored = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                status.update(stored)
        except Exception:
            pass
    status["router_model_ready"] = router_model_available()
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            status["best_val_accuracy"] = metadata.get("best_val_accuracy", status["best_val_accuracy"])
            status["best_val_macro_f1"] = metadata.get("best_val_macro_f1", status["best_val_macro_f1"])
            status["last_trained_at"] = metadata.get("trained_at", "")
        except Exception:
            pass
    return status


def write_status(**updates):
    ensure_training_dirs()
    status = load_training_status()
    status.update(updates)
    status["updated_at"] = utc_now_iso()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def append_event(payload):
    ensure_training_dirs()
    event = {"timestamp": utc_now_iso(), **payload}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def router_model_available():
    return CHECKPOINT_PATH.exists() and METADATA_PATH.exists()


def save_model_bundle(bundle):
    ensure_training_dirs()
    torch.save(bundle, CHECKPOINT_PATH)
    metadata = {
        "trained_at": bundle["trained_at"],
        "best_val_accuracy": bundle["metrics"]["accuracy"],
        "best_val_macro_f1": bundle["metrics"]["macro_f1"],
        "config": bundle["config"],
        "examples_total": bundle["examples_total"],
        "source_stats": bundle["source_stats"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def get_checkpoint_mtime():
    if not CHECKPOINT_PATH.exists():
        return 0.0
    return CHECKPOINT_PATH.stat().st_mtime


@lru_cache(maxsize=2)
def load_router_bundle_cached(checkpoint_mtime):
    if not checkpoint_mtime or not training_runtime_available():
        return None

    bundle = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = bundle["config"]
    model = SAPTicketRouter(
        vocab_size=len(bundle["vocab"]),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        label_count=len(bundle["ticket_indices"]),
        dropout=config["dropout"],
    )
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return bundle, model


def load_router_bundle():
    checkpoint_mtime = get_checkpoint_mtime()
    if not checkpoint_mtime:
        return None
    return load_router_bundle_cached(checkpoint_mtime)


def predict_ticket_candidates(text, top_k=5):
    if not training_runtime_available() or not router_model_available():
        return []

    loaded = load_router_bundle()
    if not loaded:
        return []

    bundle, model = loaded
    token_ids = encode_text(text, bundle["vocab"])
    tokens = torch.tensor(token_ids, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        logits = model(tokens, offsets)
        probabilities = torch.softmax(logits, dim=1)[0]

    count = min(top_k, probabilities.shape[0])
    values, indices = torch.topk(probabilities, k=count)
    results = []
    for probability, label_id in zip(values.tolist(), indices.tolist()):
        ticket_index = bundle["ticket_indices"][label_id]
        ticket = TICKET_CATALOG[ticket_index]
        results.append(
            {
                "ticket_index": ticket_index,
                "title": ticket["title"],
                "area": ticket["area"],
                "probability": round(float(probability), 4),
            }
        )
    return results


def train_single_trial(config, train_rows, val_rows, vocab, label_to_id, deadline, max_epochs, patience):
    set_seed(config["seed"])
    model = SAPTicketRouter(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        label_count=len(label_to_id),
        dropout=config["dropout"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    dataset = TicketRouterDataset(train_rows, vocab, label_to_id)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_batch)

    best_metric = -1.0
    best_metrics = {"accuracy": 0.0, "macro_f1": 0.0}
    best_state = None
    epochs_without_improvement = 0
    epoch_logs = []

    for epoch in range(1, max_epochs + 1):
        if time.time() >= deadline:
            break

        model.train()
        total_loss = 0.0
        batches = 0
        for tokens, offsets, labels in loader:
            optimizer.zero_grad()
            logits = model(tokens, offsets)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        metrics = evaluate_model(model, val_rows, vocab, label_to_id, batch_size=config["batch_size"])
        mean_loss = total_loss / batches if batches else 0.0
        combined_metric = (metrics["accuracy"] * 0.55) + (metrics["macro_f1"] * 0.45)
        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": round(mean_loss, 4),
                "val_accuracy": round(metrics["accuracy"], 4),
                "val_macro_f1": round(metrics["macro_f1"], 4),
            }
        )

        if combined_metric > best_metric:
            best_metric = combined_metric
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return {
        "best_state": best_state,
        "metrics": best_metrics,
        "epochs_ran": len(epoch_logs),
        "epoch_logs": epoch_logs,
    }


def run_training_job(time_budget_hours=8.0, max_trials=32, max_epochs=18, patience=4, max_weak_per_ticket=20):
    if not training_runtime_available():
        write_status(state="failed", message="PyTorch is not installed, so SAP router tuning cannot run.")
        return {"ok": False, "reason": "missing_torch"}

    if time_budget_hours <= 0:
        write_status(state="failed", message="Time budget must be greater than zero.")
        return {"ok": False, "reason": "invalid_time_budget"}

    start_time = time.time()
    deadline = start_time + (time_budget_hours * 3600.0)
    examples, source_stats = build_training_examples(max_weak_per_ticket=max_weak_per_ticket)
    train_rows, val_rows = stratified_split(examples, validation_ratio=0.2, seed=42)

    if len(train_rows) < 24 or len(val_rows) < 12:
        write_status(
            state="failed",
            message="Not enough SAP examples were generated to tune the router safely.",
            examples_total=len(examples),
            train_examples=len(train_rows),
            val_examples=len(val_rows),
        )
        return {"ok": False, "reason": "insufficient_examples"}

    ticket_indices = sorted({row["ticket_index"] for row in examples})
    label_to_id = {ticket_index: label_id for label_id, ticket_index in enumerate(ticket_indices)}
    vocab = build_vocab([row["text"] for row in train_rows])
    best_bundle = None
    best_metric = -1.0

    write_status(
        state="running",
        started_at=utc_now_iso(),
        ended_at="",
        time_budget_hours=time_budget_hours,
        examples_total=len(examples),
        train_examples=len(train_rows),
        val_examples=len(val_rows),
        trials_completed=0,
        best_trial=0,
        message="SAP router tuning is running.",
        source_stats=source_stats,
    )
    append_event(
        {
            "event": "training_started",
            "time_budget_hours": time_budget_hours,
            "examples_total": len(examples),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "source_stats": source_stats,
        }
    )

    for trial_number, config in enumerate(list_trial_configs(max_trials), start=1):
        if time.time() >= deadline:
            break

        trial_result = train_single_trial(
            config=config,
            train_rows=train_rows,
            val_rows=val_rows,
            vocab=vocab,
            label_to_id=label_to_id,
            deadline=deadline,
            max_epochs=max_epochs,
            patience=patience,
        )

        metrics = trial_result["metrics"]
        combined_metric = (metrics["accuracy"] * 0.55) + (metrics["macro_f1"] * 0.45)
        append_event(
            {
                "event": "trial_completed",
                "trial": trial_number,
                "config": config,
                "epochs_ran": trial_result["epochs_ran"],
                "metrics": metrics,
            }
        )

        if combined_metric > best_metric and trial_result["best_state"] is not None:
            best_metric = combined_metric
            best_bundle = {
                "model_state": trial_result["best_state"],
                "config": config,
                "metrics": metrics,
                "vocab": vocab,
                "ticket_indices": ticket_indices,
                "trained_at": utc_now_iso(),
                "examples_total": len(examples),
                "source_stats": source_stats,
            }
            save_model_bundle(best_bundle)

        write_status(
            state="running",
            time_budget_hours=time_budget_hours,
            examples_total=len(examples),
            train_examples=len(train_rows),
            val_examples=len(val_rows),
            trials_completed=trial_number,
            best_trial=trial_number if combined_metric >= best_metric else load_training_status().get("best_trial", 0),
            best_val_accuracy=round(best_bundle["metrics"]["accuracy"], 4) if best_bundle else 0.0,
            best_val_macro_f1=round(best_bundle["metrics"]["macro_f1"], 4) if best_bundle else 0.0,
            elapsed_minutes=round((time.time() - start_time) / 60.0, 1),
            message="SAP router tuning is running.",
            source_stats=source_stats,
        )

    ended_at = utc_now_iso()
    if best_bundle is None:
        write_status(
            state="failed",
            ended_at=ended_at,
            message="SAP router tuning finished without a usable checkpoint.",
        )
        append_event({"event": "training_failed"})
        return {"ok": False, "reason": "no_checkpoint"}

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    write_status(
        state="completed",
        ended_at=ended_at,
        best_trial=load_training_status().get("best_trial", 0) or 1,
        best_val_accuracy=round(metadata.get("best_val_accuracy", 0.0), 4),
        best_val_macro_f1=round(metadata.get("best_val_macro_f1", 0.0), 4),
        elapsed_minutes=round((time.time() - start_time) / 60.0, 1),
        message="SAP router tuning completed.",
    )
    append_event(
        {
            "event": "training_completed",
            "best_val_accuracy": metadata.get("best_val_accuracy", 0.0),
            "best_val_macro_f1": metadata.get("best_val_macro_f1", 0.0),
        }
    )
    return {"ok": True, "metrics": metadata}


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight SAP ticket router for better runbook matching.")
    parser.add_argument("--time-budget-hours", type=float, default=8.0, help="Maximum wall-clock hours to use for tuning.")
    parser.add_argument("--max-trials", type=int, default=32, help="Maximum number of hyperparameter trials.")
    parser.add_argument("--max-epochs", type=int, default=18, help="Maximum epochs per trial.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience in epochs.")
    parser.add_argument("--max-weak-per-ticket", type=int, default=20, help="Maximum weakly-labeled corpus examples per ticket.")
    args = parser.parse_args()

    result = run_training_job(
        time_budget_hours=args.time_budget_hours,
        max_trials=args.max_trials,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_weak_per_ticket=args.max_weak_per_ticket,
    )
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
