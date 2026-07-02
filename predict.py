import json
import torch
import re
from pathlib import Path
from model import TinyTransformer
from bpe_tokenizer import load_bpe_tokenizer
from config import get_config
from gazetteer import AnimeGazetteer

MAX_LEN = 100
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|'[A-Za-z]+|[^\w\s]")

# Potential Solution to predictions large title spans, partition the text and plug each partition into prediction
def define_partitions(prompt):
    sentences = re.split(r'(?<=[,.!?;])\s', prompt)
    return [s.strip() for s in sentences if s.strip()]

def normalize_bio_labels(labels):
    normalized = []
    previous_entity = None

    for label in labels:
        if label == "O":
            normalized.append(label)
            previous_entity = None
            continue

        prefix, entity = label.split("-", 1)
        if prefix == "I" and previous_entity != entity:
            label = f"B-{entity}"

        normalized.append(label)
        previous_entity = entity

    return normalized


def convert_to_word_predictions(words, word_ids, predicted_ids, id_to_label):
    predictions = []
    seen_word_ids = set()
    labels = normalize_bio_labels([
        id_to_label[predicted_id]
        for predicted_id in predicted_ids
    ])

    for word_id, label in zip(word_ids, labels):

        if word_id is None or word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)

        predictions.append({
            "word": words[word_id],
            "label": label
        })

    return predictions


def apply_gazetteer_title_labels(word_predictions, gazetteer):
    if gazetteer is None:
        return word_predictions

    words = [prediction["word"] for prediction in word_predictions]
    matches = gazetteer.find_matches(words)

    for match in matches:
        start = match["start_token"]
        end = match["end_token"]
        word_predictions[start]["label"] = "B-TITLE"

        for index in range(start + 1, end):
            word_predictions[index]["label"] = "I-TITLE"

    return word_predictions


def normalize_tokens(tokens):
    return [token.casefold() for token in tokens]


def load_genre_theme_lexicon(path):
    path = Path(path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    phrases = []
    for label_type, group_name in (("GENRE", "genres"), ("THEME", "themes")):
        for surfaces in payload.get(group_name, {}).values():
            for surface in surfaces:
                tokens = tokenize_prompt(surface)
                if tokens:
                    phrases.append((normalize_tokens(tokens), label_type))

    phrases.sort(key=lambda item: len(item[0]), reverse=True)
    return phrases


def apply_genre_theme_labels(word_predictions, tag_lexicon):
    if not tag_lexicon:
        return word_predictions

    words = [prediction["word"] for prediction in word_predictions]
    normalized_words = normalize_tokens(words)
    occupied = {
        index
        for index, prediction in enumerate(word_predictions)
        if prediction["label"].endswith("-TITLE")
    }

    for phrase_tokens, label_type in tag_lexicon:
        phrase_length = len(phrase_tokens)
        if phrase_length == 0:
            continue

        for start in range(0, len(normalized_words) - phrase_length + 1):
            end = start + phrase_length
            if any(index in occupied for index in range(start, end)):
                continue
            if normalized_words[start:end] != phrase_tokens:
                continue

            word_predictions[start]["label"] = f"B-{label_type}"
            for index in range(start + 1, end):
                word_predictions[index]["label"] = f"I-{label_type}"
            occupied.update(range(start, end))

    return word_predictions

def load_trained_model(
    tokenizer,
    label_to_id,
    model_path,
    device
):
    metadata_path = Path(f"{model_path}.meta.json")
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("label_to_id") != label_to_id:
            print(
                "Warning: checkpoint label metadata does not match current config. "
                "Retrain before trusting predictions."
            )
    else:
        print(
            "Warning: checkpoint has no label metadata. "
            "If this was trained before the GENRE/THEME label change, retrain it."
        )

    model = TinyTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        num_labels=len(label_to_id),
        max_len=MAX_LEN
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model


def tokenize_prompt(prompt):
    return TOKEN_PATTERN.findall(prompt)

# Predict the Tags for the prompt
def predict_tags(prompt, model, tokenizer, id_to_label, device, max_len=100, gazetteer=None, tag_lexicon=None):
    # Use the same word-tokenization method used when creating the JSONL data.
    words = tokenize_prompt(prompt)
    
    encoding = tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)

    input_ids = encoding.ids[:max_len]
    bpe_tokens = encoding.tokens[:max_len]
    word_ids = encoding.word_ids[:max_len]

    real_length = len(input_ids)

    pad_id = tokenizer.token_to_id("<PAD>")
    padding_length = max_len - real_length

    padded_input_ids = (input_ids + [pad_id] * padding_length)
    attention_mask = ([1] * real_length + [0] * padding_length)

    input_tensor = torch.tensor([padded_input_ids], dtype=torch.long, device=device)
    attention_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_tensor, attention_tensor)

    predicted_ids = (logits.argmax(dim=-1)[0][:real_length].cpu().tolist())

    predictions = []

    for bpe_token, word_id, predicted_id in zip(bpe_tokens, word_ids, predicted_ids):
        word = (
            words[word_id]
            if word_id is not None
            else "<SPECIAL>"
        )

        label = id_to_label[predicted_id]

        predictions.append({
            "bpe_token": bpe_token,
            "word": word,
            "label": label
        })
    word_predictions = convert_to_word_predictions(
        words=words,
        word_ids=word_ids,
        predicted_ids=predicted_ids,
        id_to_label=id_to_label,
    )
    word_predictions = apply_gazetteer_title_labels(word_predictions, gazetteer)
    word_predictions = apply_genre_theme_labels(word_predictions, tag_lexicon)
    return predictions, word_predictions
    
# Partition based input into model
def predict_partition_tags(prompt, model, tokenizer, id_to_label, device, max_len=100, gazetteer=None, tag_lexicon=None):
    partitions = define_partitions(prompt)
    partition_predictions = []
    partition_word_predictions = []
    for partition in partitions:
        if not partition.strip():
            continue
        predictions, word_predictions = predict_tags(
            partition,
            model,
            tokenizer,
            id_to_label,
            device,
            max_len=max_len,
            gazetteer=gazetteer,
            tag_lexicon=tag_lexicon,
        )
        partition_predictions.extend(predictions)
        partition_word_predictions.extend(word_predictions)
    return partition_predictions, partition_word_predictions

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    (training_path, testing_path, val_path, label_map_path, tokenizer_path, label_to_id) = get_config()

    id_to_label = {
        label_id: label
        for label, label_id in label_to_id.items()
    }
    gazetteer_path = Path("data/anime_training_data/title_gazetteer.json")
    gazetteer = AnimeGazetteer(gazetteer_path) if gazetteer_path.exists() else None
    tag_lexicon = load_genre_theme_lexicon("data/anime_training_data/genre_theme_lexicon.json")
    tokenizer = load_bpe_tokenizer(tokenizer_path)

    model = load_trained_model(
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        model_path="best_model.pth",
        device=device
    )

    prompt = input("Enter your prompt: \n")


    predictions, word_predictions = predict_tags(prompt, model, tokenizer, id_to_label, device, max_len=MAX_LEN, gazetteer=gazetteer, tag_lexicon=tag_lexicon)

    partition_predictions, partition_word_predictions = predict_partition_tags(prompt, model, tokenizer, id_to_label, device, max_len=MAX_LEN, gazetteer=gazetteer, tag_lexicon=tag_lexicon)
    partitions = define_partitions(prompt)
    print(partitions)
    for prediction in predictions:
        print(
            f"normal\n"
            f"{prediction['bpe_token']!r:18} "
            f"{prediction['word']:18} "
            f"{prediction['label']}"
        )
    print("\nWord-level predictions:\n")

    for prediction in word_predictions:
        print(
            f"{prediction['word']:18} "
            f"{prediction['label']}"
        )

    for prediction in partition_predictions:
        print(
            f"partition\n"
            f"{prediction['bpe_token']!r:18} "
            f"{prediction['word']:18} "
            f"{prediction['label']}"
        )
    print("\nWord-level predictions:\n")

    for prediction in partition_word_predictions:
        print(
            f"{prediction['word']:18} "
            f"{prediction['label']}"
        )


if __name__ == "__main__":
    main()
