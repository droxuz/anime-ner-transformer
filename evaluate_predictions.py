import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from bpe_tokenizer import load_bpe_tokenizer
from config import get_config
from dataset import load_jsonl
from predict import (
    MAX_LEN,
    apply_gazetteer_title_labels,
    apply_genre_theme_labels,
    convert_to_word_predictions,
    load_genre_theme_lexicon,
    load_trained_model,
)
from gazetteer import AnimeGazetteer


def collapse_entities(tokens, labels):
    entities = []
    index = 0

    while index < len(labels):
        label = labels[index]
        if label == "O":
            index += 1
            continue

        prefix, label_type = label.split("-", 1)
        start = index
        index += 1
        while index < len(labels) and labels[index] == f"I-{label_type}":
            index += 1

        entities.append((label_type, start, index, " ".join(tokens[start:index]).casefold()))

    return entities


def predict_row(row, model, tokenizer, id_to_label, device, gazetteer, tag_lexicon, max_len):
    encoding = tokenizer.encode(row["tokens"], is_pretokenized=True, add_special_tokens=False)
    input_ids = encoding.ids[:max_len]
    word_ids = encoding.word_ids[:max_len]
    real_length = len(input_ids)
    pad_id = tokenizer.token_to_id("<PAD>")

    input_tensor = torch.tensor(
        [input_ids + [pad_id] * (max_len - real_length)],
        dtype=torch.long,
        device=device,
    )
    attention_tensor = torch.tensor(
        [[1] * real_length + [0] * (max_len - real_length)],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(input_tensor, attention_tensor)

    predicted_ids = logits.argmax(dim=-1)[0][:real_length].cpu().tolist()
    word_predictions = convert_to_word_predictions(
        words=row["tokens"],
        word_ids=word_ids,
        predicted_ids=predicted_ids,
        id_to_label=id_to_label,
    )
    word_predictions = apply_gazetteer_title_labels(word_predictions, gazetteer)
    word_predictions = apply_genre_theme_labels(word_predictions, tag_lexicon)
    predicted_labels = [prediction["label"] for prediction in word_predictions]

    return predicted_labels[:len(row["labels"])]


def update_token_counts(counts, gold_labels, predicted_labels):
    for gold, predicted in zip(gold_labels, predicted_labels):
        counts[gold]["support"] += 1
        if gold == predicted:
            counts[gold]["tp"] += 1
        else:
            counts[gold]["fn"] += 1
            counts[predicted]["fp"] += 1


def metric_line(label, values):
    tp = values["tp"]
    fp = values["fp"]
    fn = values["fn"]
    support = values["support"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--model-path", default="best_model.pth")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    training_path, testing_path, val_path, label_map_path, tokenizer_path, label_to_id = get_config()
    split_paths = {"train": training_path, "val": val_path, "test": testing_path}
    rows = load_jsonl(split_paths[args.split])[:args.limit]
    id_to_label = {label_id: label for label, label_id in label_to_id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_bpe_tokenizer(tokenizer_path)
    model = load_trained_model(tokenizer, label_to_id, args.model_path, device)

    gazetteer_path = Path("data/anime_training_data/title_gazetteer.json")
    gazetteer = AnimeGazetteer(gazetteer_path) if gazetteer_path.exists() else None
    tag_lexicon = load_genre_theme_lexicon("data/anime_training_data/genre_theme_lexicon.json")

    token_counts = defaultdict(Counter)
    entity_gold = Counter()
    entity_predicted = Counter()
    entity_true_positive = Counter()
    missed_examples = []

    for row in rows:
        predicted_labels = predict_row(
            row,
            model,
            tokenizer,
            id_to_label,
            device,
            gazetteer,
            tag_lexicon,
            args.max_len,
        )
        gold_labels = row["labels"][:len(predicted_labels)]
        tokens = row["tokens"][:len(predicted_labels)]

        update_token_counts(token_counts, gold_labels, predicted_labels)

        gold_entities = set(collapse_entities(tokens, gold_labels))
        predicted_entities = set(collapse_entities(tokens, predicted_labels))
        for entity in gold_entities:
            entity_gold[entity[0]] += 1
        for entity in predicted_entities:
            entity_predicted[entity[0]] += 1
        for entity in gold_entities & predicted_entities:
            entity_true_positive[entity[0]] += 1

        if gold_entities - predicted_entities and len(missed_examples) < 20:
            missed_examples.append({
                "prompt": row["prompt"],
                "missed": sorted(gold_entities - predicted_entities),
                "extra": sorted(predicted_entities - gold_entities),
            })

    print("\nToken metrics:")
    for label in sorted(label_to_id, key=label_to_id.get):
        metric = metric_line(label, token_counts[label])
        print(
            f"  {label:8} precision={metric['precision']:.3f} "
            f"recall={metric['recall']:.3f} f1={metric['f1']:.3f} "
            f"support={metric['support']:,}"
        )

    print("\nExact entity metrics:")
    for label_type in ("TITLE", "GENRE", "THEME"):
        tp = entity_true_positive[label_type]
        predicted = entity_predicted[label_type]
        gold = entity_gold[label_type]
        precision = tp / predicted if predicted else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"  {label_type:6} precision={precision:.3f} "
            f"recall={recall:.3f} f1={f1:.3f} support={gold:,}"
        )

    print("\nMissed entity examples:")
    print(json.dumps(missed_examples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
