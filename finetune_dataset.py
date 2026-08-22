import torch
from torch.utils.data import Dataset
import re
import random
import json



class FTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, bio_lib):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bio_lib = bio_lib
        self.CLS = self.tokenizer.token_to_id("[CLS]")
        self.SEP = self.tokenizer.token_to_id("[SEP]")
        self.PAD = self.tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.data)

    def bio_labels_offset(self, text, offsets, entities):
        labels = [self.bio_lib["O"]] * len(offsets)
        entities = sorted(entities, key= lambda x: (x["start"], x["end"]))
        for entity in entities:
            entity_start = entity["start"]
            entity_end = entity["end"]
            entity_label = entity["label"]
            matched_token_indices = []

            for token, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end:
                    continue
                token_text = text[token_start:token_end]

                if token_text.strip() == "":
                    continue

                # if the entity is greater
                overlap = token_start < entity_end and token_end > entity_start
                if overlap:
                    matched_token_indices.append(token)
            if not matched_token_indices:
                continue
            first_index = matched_token_indices[0]
            labels[first_index] = self.bio_lib[f"B-{entity_label}"]
            for token_index in matched_token_indices[1:]:
                labels[token_index] = self.bio_lib[f"I-{entity_label}"]
        return labels
            
    # Get Attention Mask, Word to Tokens, and Label Mapping per token
    def __getitem__(self, idx):
        x_recommendation = self.data[idx]
        text = x_recommendation["text"]
        entities = x_recommendation["entities"]
        encoding = self.tokenizer.encode(text)
        
        recommendation_ids = encoding.ids
        recommendation_offset = encoding.offsets
        
        recommendation_labels = self.bio_labels_offset(text, recommendation_offset, entities)

        # Truncate IDS and labels to max
        recommendation_ids = recommendation_ids[:self.max_len-2]
        recommendation_labels = recommendation_labels[:self.max_len-2]

        # Adding CLS and SEP for EOL and SOL 
        recommendation_ids = [self.CLS] + recommendation_ids + [self.SEP]
        recommendation_labels = [-100] + recommendation_labels + [-100]

        # Attention Mask
        attention_mask = [1] * len(recommendation_ids)

        # Padding and Length Matching
        padding_length = self.max_len - len(recommendation_ids)
        recommendation_ids = recommendation_ids + [self.PAD] * padding_length
        recommendation_labels = recommendation_labels + [-100] * padding_length 
        attention_mask = attention_mask + [0] * padding_length

        return{
            "input_ids": torch.tensor(recommendation_ids, dtype= torch.long),
            "labels": torch.tensor(recommendation_labels, dtype= torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype= torch.long)
        }