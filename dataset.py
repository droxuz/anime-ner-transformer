import torch
from torch.utils.data import Dataset
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

# Creates a vocabulary dictionary with unique ID's for each word
def build_vocab(data, min_freq = 1):
    word_to_id = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    for row in data:
        for token in row["tokens"]:
            token = token.lower()

            if token not in word_to_id:
                word_to_id[token] = len(word_to_id)

    return word_to_id

class AnimePromptDataset(Dataset):
    def __init__(self, data, vocab_dict, max_len=40):
        self.data = data
        self.vocab_dict = vocab_dict
        self.max_len = max_len

    def __len__(self):
        # return number of examples
        return len(self.data)

    def __getitem__(self, index):
        # get one row
        # convert tokens to input_ids
        # convert labels to label_ids
        # truncate
        # create attention_mask
        # pad input_ids, attention_mask, label_ids
        # return tensors
        row = self.data[index]
        input_ids = []
        label_ids = []

        # Places ids into respective lists per row
        for tokens in row["tokens"]:
            token_id = self.vocab_dict.get(tokens.lower(), self.vocab_dict["<UNK>"])
            input_ids.append(token_id)
        for labels in row["labels"]:
            label_id = label_to_id[labels]
            label_ids.append(label_id)

        # Truncate length of ids to fit max    
        input_ids = input_ids[:self.max_len] 
        label_ids = label_ids[:self.max_len]

        # Create attention masking from padding mask
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_len - len(input_ids)
        attention_mask = attention_mask + [0] * padding_length

        # Padding for input_ids, label_ids
        input_ids = input_ids + [self.vocab_dict["<PAD>"]] * padding_length
        label_ids = label_ids + [label_to_id[-100]] * padding_length
        
        # Tensors
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
 
training_data = load_jsonl(training_path)
vocab_dict = build_vocab(training_data)



#print(id_to_label)
#print(label_to_id)
#print(training_data[0])
#print(len(vocab_dict))
#print(vocab_dict["action"])
#print(vocab_dict["<PAD>"])
#print(vocab_dict["<UNK>"])


