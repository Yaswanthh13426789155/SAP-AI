from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path("sap_dataset.txt")


def export_dataset(output_path=OUTPUT_PATH):
    dataset = load_dataset("orkungedik/SAP-basis-dataset")
    rows = []

    for item in dataset["train"]:
        lines = []
        for key, value in item.items():
            text = str(value).strip()
            if text:
                lines.append(f"{key}: {text}")
        if lines:
            rows.append("\n".join(lines))

    output_path.write_text("\n\n".join(rows), encoding="utf-8")
    return len(rows), output_path


if __name__ == "__main__":
    count, path = export_dataset()
    print(f"Exported {count} dataset records to {path}")
