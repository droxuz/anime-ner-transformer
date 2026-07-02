import torch
import re
from model import TinyTransformer
from bpe_tokenizer import load_bpe_tokenizer
from config import get_config
from gazetteer import AnimeGazetteer

MAX_LEN = 100
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|'[A-Za-z]+|[^\w\s]")

# Potential Solution to predictions large title spans, partition the text and plug each partition into prediction
def define_partitions(prompt):
    sentences = re.split(r'(?<=[.!?;])\s', prompt)
    return [s.strip() for s in sentences if s.strip()]

def convert_to_word_predictions(words, word_ids, predicted_ids):
    predictions = []
    seen_word_ids = set()

    for word_id, label in zip(word_ids, predicted_ids):

        if word_id is None or word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)

        predictions.append({
            "word": words[word_id],
            "label": label
        })

    return predictions

def load_trained_model(
    tokenizer,
    label_to_id,
    model_path,
    device
):
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
def predict_tags(prompt, model, tokenizer, id_to_label, device, max_len=100):
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
    word_predictions = convert_to_word_predictions(words=words,word_ids=word_ids,predicted_ids=predicted_ids,)
    return predictions, word_predictions
    
# Partition based input into model
def predict_partition_tags(prompt, model, tokenizer, id_to_label, device, max_len=100):
    partitions = define_partitions(prompt)
    partition_predictions = []
    partition_word_predictions = []
    for partition in partitions:
        if not partition.strip():
            continue
        predictions, word_predictions = predict_tags(partition, model, tokenizer, id_to_label, device, max_len=100)
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
    #gazetteer = AnimeGazetteer("data/anime_training_data/title_gazetteer.json")
    tokenizer = load_bpe_tokenizer(tokenizer_path)

    model = load_trained_model(
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        model_path="best_model.pth",
        device=device
    )

    prompt = input("Enter your prompt: \n")


    predictions, word_predictions = predict_tags(prompt, model, tokenizer, id_to_label, device, max_len=MAX_LEN)

    if len(tokenize_prompt(prompt)) > 50:
        partition_predictions, partition_word_predictions = predict_partition_tags(prompt, model, tokenizer, id_to_label, device, max_len=MAX_LEN)
        partitions = define_partitions(prompt)
        print(partitions)
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

    


if __name__ == "__main__":
    main()