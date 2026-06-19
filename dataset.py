import torch
from config import get_config
from torch.utils.data import Dataset
import json
# Loads config
training_path, testing_path, val_path, label_map_path, save_path, label_to_id = get_config()

# Loads json into dict
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding = 'UTF-8') as file:
        for line in file:
            row = json.loads(line)
            data.append(row)
    return data

# Creates a vocabulary dictionary with unique ID's for each word-tokens
# def build_vocab(data, min_freq = 1):
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

#Dataset
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
        label_ids = label_ids + [-100] * padding_length
        
        # Tensors
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class AnimeBPEDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_len= 100):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self):
        return 
    
    def __getitem__():
        pass

