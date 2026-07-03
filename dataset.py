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

# Dataset
class AnimeBPEDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_len= 100):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("<PAD>")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        original_tokens = row["tokens"]
        original_labels = row["labels"]

        # Split pre tokens into subwords using the tokenizer
        token_encoding = self.tokenizer.encode(original_tokens, is_pretokenized= True, add_special_tokens= False)
        input_ids = token_encoding.ids[:self.max_len]
        word_ids = token_encoding.word_ids[:self.max_len]

        # Assignment of label to id using the dict on the BPE subword
        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            # If word_id is not in BPE then assign -100 
            if word_id is None:
                label_ids.append(-100)
            # Assignment only for beginning of word transition from tag and prefix of subword BPE
            elif word_id != previous_word_id:
                original_label = original_labels[word_id]
                label_ids.append(self.label_to_id[original_label])
            else:
                label_ids.append(-100)
                
            previous_word_id = word_id


        #Create a attention mask from the new BPE prompt
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_len - len(input_ids)
        attention_mask = attention_mask + [0] * padding_length

        #Create padding for input_ids, and label_ids
        input_ids = input_ids + [self.pad_id] * padding_length
        label_ids = label_ids + [-100] * padding_length


        #Tensors
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
