import torch
import json
training_path = "data/anime_training_data/train.jsonl"
testing_path = "data/anime_training_data/test.jsonl"
val_path = "data/anime_training_data/val.jsonl"
label_map_path = "data/anime_training_data/label_map.json"

pad_token = "<PAD>"
unk_token = "<UNK>"

id_to_label = {
    0: 'O', 
    1: 'B-LIKED_TITLE', 
    2: 'I-LIKED_TITLE', 
    3: 'B-POSITIVE_TERM', 
    4: 'I-POSITIVE_TERM', 
    5: 'B-NEGATIVE_TERM', 
    6: 'I-NEGATIVE_TERM'
}

label_to_id = {value: key for key, value in id_to_label.items()}

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding = 'UTF-8') as file:
        for line in file:
            row = json.loads(line)
            data.append(row)
    return data

#Might use the JSON probably not
#def load_label_map(path): 
    with open(path, 'r', encoding = 'UTF-8') as file:
        data = json.load(file)
        label_to_id = data["label_to_id"]
        id_to_label = data["id_to_label"]
        id_to_label = {int(key): value for key, value in id_to_label.items()}
    return label_to_id, id_to_label



training_data = load_jsonl(training_path)

print(id_to_label)
print(label_to_id)
