import torch
from finetune_model import NERTransformer
from pretrained_bpe import  train_bpe_tokenizer, get_bpe_tokenizer, load_bpe_tokenizer
from rapidfuzz import process, fuzz
import json
import unicodedata
import re

gazetteer_path = "data/anime_training_data/title_gazetteer.json"
BPE_tokenizer = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"
model_path = "data/anime_training_data/NER_Best_Model.pt"

def get_prompt():
    print(f"What would like to watch: \n")
    prompt = input()
    return prompt

def normalize_prompt(prompt):
    normal = unicodedata.normalize("NFKC", prompt)
    normal = prompt.strip().lower()
    normal = re.sub(r"[:;,.!?()\[\]{}\"'`~_/\\\-]+", " ", normal)
    normal = re.sub(r"\s+", " ", normal).strip()
    return normal

def load_gazetteer(path):
    with open(path, "r", encoding= "UTF-8") as file:
        gazetteer = json.load(file)
    return gazetteer

def get_gazetteer_dict(gazetteer):
    title_dict = {}
    for anime in gazetteer:
        anime_id = anime["mal_id"]
        canon_title = anime["canonical_title"]
        for alias in anime["aliases"]:
            norm_alias = normalize_prompt(alias)
            title_dict[norm_alias] = {
                "anime_id": anime_id,
                "canon_title": canon_title,
            }   
    return title_dict

def create_tokens(prompt):
    tokens = []
    for match in re.finditer(r"\S+", prompt):   
        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end()
        })
    return tokens

def fuzzy_match(prompt, tokens, alias_dict, max_len):
    alias_list = list(alias_dict.keys())
    for i in range(len(tokens)):
        for j in range(i, min(i+max_len, len(tokens))):
            start = tokens[i]["start"]
            end = tokens[j]["end"]
            raw = prompt[start:end]
            norm = normalize_prompt(raw)
            result = process.extractOne(norm, alias_list, scorer= fuzz.ratio, score_cutoff= 85)
            if result is None:
                continue
            matched_alias, score, index = result

def encode_prompt_for_ner(prompt, tokenizer, max_len):
    CLS = tokenizer.token_to_id("[CLS]")
    SEP = tokenizer.token_to_id("[SEP]")
    PAD = tokenizer.token_to_id("[PAD]")

    encoding = tokenizer.encode(prompt)

    token_ids = encoding.ids[:max_len - 2]
    offsets = encoding.offsets[:max_len - 2]
    tokens = encoding.tokens[:max_len - 2]

    input_ids = [CLS] + token_ids + [SEP]
    attention_mask = [1] * len(input_ids)

    # Add fake offsets/tokens for CLS and SEP
    offsets = [(0, 0)] + offsets + [(0, 0)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    padding_length = max_len - len(input_ids)

    input_ids += [PAD] * padding_length
    attention_mask += [0] * padding_length
    offsets += [(0, 0)] * padding_length
    tokens += ["[PAD]"] * padding_length

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "offsets": offsets,
        "tokens": tokens
    }

def predict_ner(prompt, model, tokenizer, id_to_label, device):
    encoded = encode_prompt_for_ner(
        prompt=prompt,
        tokenizer=tokenizer,
        max_len=model.position_embedding.num_embeddings
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=-1)

    prediction_ids = predictions[0].cpu().tolist()

    results = []

    for token, offset, pred_id, mask in zip(
        encoded["tokens"],
        encoded["offsets"],
        prediction_ids,
        encoded["attention_mask"][0].tolist()
    ):
        if mask == 0:
            continue

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        label = id_to_label[pred_id]

        results.append({
            "token": token,
            "offset": offset,
            "text": prompt[offset[0]:offset[1]],
            "label": label
        })

    return results
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"
    tokenizer = load_bpe_tokenizer(tokenizer_path)
    checkpoint = torch.load(model_path, device)
    config = checkpoint["config"]
    BIO_LIB = checkpoint["bio_lib"]
    id_to_label = {v: k for k, v in BIO_LIB.items()}
    prompt = get_prompt()
    model = NERTransformer(vocab_size = config["vocab_size"],
            max_len= config["max_len"],
            pad_id= config["pad_id"],
            d_model= config["d_model"],
            nhead= config["nhead"],
            num_encoder_layers= config["num_encoder_layers"],
            dim_feedforward= config["dim_feedforward"],
            dropout= config["dropout"],
            num_labels= config["num_labels"]
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    predictions = predict_ner(prompt, model, tokenizer, id_to_label, device)
    for item in predictions:
        print(item["text"], item["token"], item["label"])


    # max_len = 8
    
    # norm_prompt = normalize_prompt(prompt) # Normalize prompt for Fuzzy matching
    # gazetteer = load_gazetteer(gazetteer_path)
    # alias_dict = get_gazetteer_dict(gazetteer)
   
if __name__ == "__main__":
    main()