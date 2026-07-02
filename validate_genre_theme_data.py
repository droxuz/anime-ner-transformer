import json
import re
import sys
from collections import Counter
from pathlib import Path


DATA_DIR = Path("data/anime_training_data")
ALLOWED_LABELS = {"O", "B-TITLE", "I-TITLE", "B-GENRE", "I-GENRE", "B-THEME", "I-THEME"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|'[A-Za-z]+|[^\w\s]")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def validate_bio_sequence(labels):
    errors = []
    previous_type = None

    for index, label in enumerate(labels):
        if label == "O":
            previous_type = None
            continue

        prefix, label_type = label.split("-", 1)
        if prefix == "I" and previous_type != label_type:
            errors.append(f"I-{label_type} at token {index} does not follow {label_type}")

        previous_type = label_type

    return errors


def row_errors(split_name, row_index, row):
    errors = []
    required = {"prompt", "tokens", "labels", "genre_terms", "theme_terms", "liked_titles"}
    missing = required.difference(row)
    if missing:
        errors.append(f"{split_name}[{row_index}] missing fields {sorted(missing)}")
        return errors

    if len(row["tokens"]) != len(row["labels"]):
        errors.append(
            f"{split_name}[{row_index}] has {len(row['tokens'])} tokens but {len(row['labels'])} labels"
        )

    unknown = set(row["labels"]) - ALLOWED_LABELS
    if unknown:
        errors.append(f"{split_name}[{row_index}] unknown labels {sorted(unknown)}")

    prompt_tokens = TOKEN_PATTERN.findall(row["prompt"])
    if prompt_tokens != row["tokens"]:
        errors.append(f"{split_name}[{row_index}] prompt does not retokenize to stored tokens")

    errors.extend(
        f"{split_name}[{row_index}] {error}"
        for error in validate_bio_sequence(row["labels"])
    )
    return errors


def summarize(split_name, rows):
    label_counts = Counter(label for row in rows for label in row["labels"])
    lengths = [len(row["tokens"]) for row in rows]
    print(f"\n{split_name}: {len(rows):,} rows")
    print(f"  token length min/avg/max: {min(lengths)} / {sum(lengths) / len(lengths):.2f} / {max(lengths)}")
    for label in sorted(ALLOWED_LABELS):
        print(f"  {label:8} {label_counts[label]:9,d}")


def main():
    all_errors = []
    for split_name in ("train", "val", "test"):
        rows = load_jsonl(DATA_DIR / f"{split_name}.jsonl")
        summarize(split_name, rows)
        for row_index, row in enumerate(rows):
            all_errors.extend(row_errors(split_name, row_index, row))

    if all_errors:
        print(f"\nFAILED: {len(all_errors)} validation errors")
        for error in all_errors[:50]:
            print(f"  - {error}")
        if len(all_errors) > 50:
            print(f"  ...and {len(all_errors) - 50} more")
        return 1

    print("\nPASSED: genre/theme training data is structurally valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
