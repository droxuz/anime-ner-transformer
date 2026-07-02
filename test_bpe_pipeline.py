"""
Diagnostic tests for the anime BPE token-classification pipeline.

Run from the project root:

    python test_bpe_pipeline.py

Optional:

    python test_bpe_pipeline.py --max-len 100 --alignment-samples 500
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from bpe_tokenizer import load_bpe_tokenizer
from config import get_config
from dataset import AnimeBPEDataset, load_jsonl


def load_project_config():
    values = get_config()
    if len(values) != 6:
        raise ValueError(
            "get_config() must return: training_path, testing_path, val_path, "
            "label_map_path, save_path, label_to_id"
        )

    training_path, testing_path, val_path, label_map_path, save_path, label_to_id = values
    return (
        Path(training_path),
        Path(testing_path),
        Path(val_path),
        Path(label_map_path),
        Path(save_path),
        label_to_id,
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def get_label_tensor(sample: dict[str, torch.Tensor]) -> torch.Tensor:
    if "labels" in sample:
        return sample["labels"]
    if "label_ids" in sample:
        return sample["label_ids"]
    raise KeyError("Dataset sample has neither 'labels' nor 'label_ids'.")


def make_dataset(data, tokenizer, label_to_id, max_len):
    try:
        return AnimeBPEDataset(
            data,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            max_len=max_len,
        )
    except TypeError:
        return AnimeBPEDataset(
            data,
            tokenizer=tokenizer,
            max_len=max_len,
        )


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def validate_rows(split_name, data, label_to_id):
    errors = []
    required = {"prompt", "tokens", "labels"}

    for index, row in enumerate(data):
        missing = required.difference(row)
        if missing:
            errors.append(f"{split_name}[{index}] missing fields: {sorted(missing)}")
            continue

        if len(row["tokens"]) != len(row["labels"]):
            errors.append(
                f"{split_name}[{index}] has {len(row['tokens'])} tokens but "
                f"{len(row['labels'])} labels"
            )

        for label in row["labels"]:
            if label not in label_to_id:
                errors.append(f"{split_name}[{index}] has unknown label {label!r}")

    return errors


def bpe_stats(data, tokenizer, max_len):
    unk_id = tokenizer.token_to_id("<UNK>")
    if unk_id is None:
        raise ValueError("Tokenizer has no <UNK> token")

    lengths = []
    total_subwords = 0
    unk_count = 0

    for row in data:
        encoding = tokenizer.encode(
            row["tokens"],
            is_pretokenized=True,
            add_special_tokens=False,
        )
        lengths.append(len(encoding.ids))
        total_subwords += len(encoding.ids)
        unk_count += sum(token_id == unk_id for token_id in encoding.ids)

    truncated = sum(length > max_len for length in lengths)

    return {
        "examples": len(lengths),
        "total_subwords": total_subwords,
        "unk_count": unk_count,
        "unk_rate": unk_count / total_subwords if total_subwords else 0.0,
        "min": min(lengths) if lengths else 0,
        "median": percentile(lengths, 0.50) if lengths else 0,
        "p95": percentile(lengths, 0.95) if lengths else 0,
        "p99": percentile(lengths, 0.99) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "truncated": truncated,
        "truncated_rate": truncated / len(lengths) if lengths else 0.0,
    }


def independently_align(row, tokenizer, label_to_id):
    encoding = tokenizer.encode(
        row["tokens"],
        is_pretokenized=True,
        add_special_tokens=False,
    )

    aligned = []
    previous_word_id = None

    for word_id in encoding.word_ids:
        if word_id is None:
            aligned.append(-100)
            previous_word_id = None
            continue

        original_label = row["labels"][word_id]
        first_subword = word_id != previous_word_id

        if original_label == "O":
            aligned_label = "O"
        elif first_subword:
            aligned_label = original_label
        else:
            entity_type = original_label.split("-", 1)[1]
            aligned_label = f"I-{entity_type}"

        aligned.append(label_to_id[aligned_label])
        previous_word_id = word_id

    return encoding, aligned


def test_dataset_alignment(
    split_name,
    data,
    dataset,
    tokenizer,
    label_to_id,
    max_len,
    sample_count,
    seed,
):
    errors = []
    if not data:
        return [f"{split_name} is empty"]

    rng = random.Random(seed)
    indices = list(range(len(data)))
    if sample_count < len(indices):
        indices = rng.sample(indices, sample_count)

    pad_id = tokenizer.token_to_id("<PAD>")
    if pad_id is None:
        return ["Tokenizer has no <PAD> token"]

    for index in indices:
        sample = dataset[index]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = get_label_tensor(sample)

        if not (
            len(input_ids) == len(attention_mask) == len(labels) == max_len
        ):
            errors.append(f"{split_name}[{index}] final lengths do not equal {max_len}")
            continue

        valid_length = int(attention_mask.sum().item())
        encoding, expected_labels = independently_align(
            data[index], tokenizer, label_to_id
        )

        expected_ids = encoding.ids[:max_len]
        expected_labels = expected_labels[:max_len]

        if input_ids[:valid_length].tolist() != expected_ids:
            errors.append(f"{split_name}[{index}] input IDs do not match BPE output")

        if labels[:valid_length].tolist() != expected_labels:
            errors.append(f"{split_name}[{index}] BIO/BPE label alignment mismatch")

        if any(value != pad_id for value in input_ids[valid_length:].tolist()):
            errors.append(f"{split_name}[{index}] incorrect input padding")

        if any(value != 0 for value in attention_mask[valid_length:].tolist()):
            errors.append(f"{split_name}[{index}] incorrect attention padding")

        if any(value != -100 for value in labels[valid_length:].tolist()):
            errors.append(f"{split_name}[{index}] incorrect label padding")

    return errors


def label_distribution(data):
    return Counter(label for row in data for label in row["labels"])


def print_label_distribution(split_name, data, ordered_labels):
    counts = label_distribution(data)
    total = sum(counts.values())
    print(f"\n{split_name} label distribution:")

    for label in ordered_labels:
        count = counts[label]
        percentage = 100 * count / total if total else 0
        print(f"  {label:22} {count:9,d}  ({percentage:6.2f}%)")


def prompt_overlap(train_data, target_data):
    train_prompts = {normalize_text(row["prompt"]) for row in train_data}
    target_prompts = [normalize_text(row["prompt"]) for row in target_data]
    count = sum(prompt in train_prompts for prompt in target_prompts)
    return count, count / len(target_prompts) if target_prompts else 0.0


def row_template(row):
    result = []
    index = 0

    while index < len(row["tokens"]):
        label = row["labels"][index]

        if label == "O":
            result.append(normalize_text(row["tokens"][index]))
            index += 1
            continue

        entity_type = label.split("-", 1)[1]
        result.append(f"<{entity_type}>")
        index += 1

        while (
            index < len(row["tokens"])
            and row["labels"][index] == f"I-{entity_type}"
        ):
            index += 1

    return " ".join(result)


def template_coverage(train_data, target_data):
    train_templates = {row_template(row) for row in train_data}
    target_templates = [row_template(row) for row in target_data]
    covered = sum(template in train_templates for template in target_templates)
    unseen = Counter(
        template for template in target_templates if template not in train_templates
    )
    rate = covered / len(target_templates) if target_templates else 0.0
    return covered, rate, unseen


def extract_liked_titles(row):
    if isinstance(row.get("liked_titles"), list):
        return [
            normalize_text(str(title))
            for title in row["liked_titles"]
            if str(title).strip()
        ]

    titles = []
    current = []

    for token, label in zip(row["tokens"], row["labels"]):
        if label == "B-TITLE":
            if current:
                titles.append(normalize_text(" ".join(current)))
            current = [token]
        elif label == "I-TITLE" and current:
            current.append(token)
        else:
            if current:
                titles.append(normalize_text(" ".join(current)))
                current = []

    if current:
        titles.append(normalize_text(" ".join(current)))

    return titles


def exposure_rate(train_values, target_values):
    seen = sum(value in train_values for value in target_values)
    total = len(target_values)
    unseen_examples = sorted(
        {value for value in target_values if value not in train_values}
    )[:20]
    return seen, total, seen / total if total else 1.0, unseen_examples


def title_exposure(train_data, target_data):
    train_titles = {
        title for row in train_data for title in extract_liked_titles(row)
    }
    target_titles = [
        title for row in target_data for title in extract_liked_titles(row)
    ]
    return exposure_rate(train_titles, target_titles)


def canonical_terms(data, fields):
    values = []
    for row in data:
        for field in fields:
            if isinstance(row.get(field), list):
                values.extend(
                    normalize_text(str(value))
                    for value in row[field]
                    if str(value).strip()
                )
    return values


def print_bpe_stats(split_name, stats):
    print(f"\n{split_name} BPE statistics:")
    print(f"  Examples:          {stats['examples']:,}")
    print(f"  Total subwords:    {stats['total_subwords']:,}")
    print(f"  UNK count:         {stats['unk_count']:,}")
    print(f"  UNK rate:          {100 * stats['unk_rate']:.4f}%")
    print(f"  Min/median:        {stats['min']} / {stats['median']}")
    print(f"  P95/P99/max:       {stats['p95']} / {stats['p99']} / {stats['max']}")
    print(f"  Truncated rows:    {stats['truncated']:,}")
    print(f"  Truncation rate:   {100 * stats['truncated_rate']:.4f}%")


def print_alignment_examples(data, tokenizer, label_to_id, count):
    id_to_label = {value: key for key, value in label_to_id.items()}
    print("\nSample BPE alignments:")

    for row in data[:count]:
        encoding, aligned = independently_align(row, tokenizer, label_to_id)
        print(f"\nPrompt: {row['prompt']}")
        print(f"{'BPE token':24} {'Word ID':8} Label")

        for token, word_id, label_id in zip(
            encoding.tokens, encoding.word_ids, aligned
        ):
            label = "-100" if label_id == -100 else id_to_label[label_id]
            print(f"{token!r:24} {str(word_id):8} {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--alignment-samples", type=int, default=500)
    parser.add_argument("--show-alignments", type=int, default=3)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    training_path, testing_path, val_path, label_map_path, tokenizer_path, label_to_id = get_config()

    required = [training_path, testing_path, val_path, tokenizer_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  {path}")
        return 1

    print("Loading data and tokenizer...")
    training_data = load_jsonl(training_path)
    val_data = load_jsonl(val_path)
    testing_data = load_jsonl(testing_path)
    tokenizer = load_bpe_tokenizer(str(tokenizer_path))

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"PAD ID: {tokenizer.token_to_id('<PAD>')}")
    print(f"UNK ID: {tokenizer.token_to_id('<UNK>')}")

    critical_errors = []

    print("\n=== 1. JSONL STRUCTURE AND LABEL VALIDITY ===")
    for name, data in (
        ("Train", training_data),
        ("Validation", val_data),
        ("Test", testing_data),
    ):
        errors = validate_rows(name, data, label_to_id)
        critical_errors.extend(errors)
        print(f"{name}: {len(errors)} errors")

    print("\n=== 2. BPE UNK, LENGTH, AND TRUNCATION ===")
    for name, data in (
        ("Train", training_data),
        ("Validation", val_data),
        ("Test", testing_data),
    ):
        print_bpe_stats(name, bpe_stats(data, tokenizer, args.max_len))

    print("\n=== 3. DATASET PADDING AND LABEL ALIGNMENT ===")
    train_dataset = make_dataset(training_data, tokenizer, label_to_id, args.max_len)
    val_dataset = make_dataset(val_data, tokenizer, label_to_id, args.max_len)
    test_dataset = make_dataset(testing_data, tokenizer, label_to_id, args.max_len)

    for name, data, dataset in (
        ("Train", training_data, train_dataset),
        ("Validation", val_data, val_dataset),
        ("Test", testing_data, test_dataset),
    ):
        errors = test_dataset_alignment(
            name,
            data,
            dataset,
            tokenizer,
            label_to_id,
            args.max_len,
            args.alignment_samples,
            args.seed,
        )
        critical_errors.extend(errors)
        print(f"{name}: {len(errors)} alignment/padding errors")

    print_alignment_examples(
        training_data,
        tokenizer,
        label_to_id,
        args.show_alignments,
    )

    print("\n=== 4. LABEL DISTRIBUTIONS ===")
    ordered_labels = [
        label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])
    ]
    print_label_distribution("Train", training_data, ordered_labels)
    print_label_distribution("Validation", val_data, ordered_labels)
    print_label_distribution("Test", testing_data, ordered_labels)

    print("\n=== 5. EXACT PROMPT OVERLAP ===")
    for name, data in (("Validation", val_data), ("Test", testing_data)):
        count, rate = prompt_overlap(training_data, data)
        print(f"Train -> {name}: {count:,}/{len(data):,} ({100 * rate:.2f}%)")

    print("\n=== 6. TEMPLATE COVERAGE ===")
    for name, data in (("Validation", val_data), ("Test", testing_data)):
        covered, rate, unseen = template_coverage(training_data, data)
        print(
            f"Train templates cover {covered:,}/{len(data):,} {name.lower()} "
            f"examples ({100 * rate:.2f}%)"
        )
        if unseen:
            print("  Most common unseen templates:")
            for template, count in unseen.most_common(10):
                print(f"    {count:5,d}  {template}")

    print("\n=== 7. LIKED-TITLE EXPOSURE ===")
    for name, data in (("Validation", val_data), ("Test", testing_data)):
        seen, total, rate, unseen = title_exposure(training_data, data)
        print(
            f"Train -> {name}: {seen:,}/{total:,} title mentions seen "
            f"({100 * rate:.2f}%)"
        )
        if unseen:
            print("  Sample unseen titles:")
            for title in unseen:
                print(f"    {title}")

    print("\n=== 8. GENRE/THEME TERM EXPOSURE ===")
    train_genres = set(canonical_terms(training_data, ("genre_terms",)))
    train_themes = set(canonical_terms(training_data, ("theme_terms",)))

    for name, data in (("Validation", val_data), ("Test", testing_data)):
        target_genres = canonical_terms(data, ("genre_terms",))
        target_themes = canonical_terms(data, ("theme_terms",))

        genre_seen, genre_total, genre_rate, genre_unseen = exposure_rate(
            train_genres, target_genres
        )
        theme_seen, theme_total, theme_rate, theme_unseen = exposure_rate(
            train_themes, target_themes
        )

        print(
            f"Train -> {name} genres: {genre_seen:,}/{genre_total:,} "
            f"({100 * genre_rate:.2f}%)"
        )
        print(
            f"Train -> {name} themes: {theme_seen:,}/{theme_total:,} "
            f"({100 * theme_rate:.2f}%)"
        )
        if genre_unseen:
            print(f"  Sample unseen genres: {genre_unseen[:10]}")
        if theme_unseen:
            print(f"  Sample unseen themes: {theme_unseen[:10]}")

    print("\n=== FINAL RESULT ===")
    if critical_errors:
        print(f"FAILED: {len(critical_errors)} critical errors found")
        for error in critical_errors[:50]:
            print(f"  - {error}")
        if len(critical_errors) > 50:
            print(f"  ...and {len(critical_errors) - 50} more")
        return 1

    print("PASSED: no critical structure, padding, or alignment errors found")
    print(
        "Review template coverage, title exposure, term exposure, and "
        "truncation rates for distribution mismatch."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
