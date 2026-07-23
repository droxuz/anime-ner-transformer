import torch
from torch.utils.data import Dataset
import re
import random
class MLMDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, mask_probability):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_probability = mask_probability
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.special_ids = {
            self.cls_id,
            self.sep_id,
            self.pad_id,
            self.mask_id,
            self.unk_id,
        }
        self.normal_vocab_id = [token_id for token_id in range(self.tokenizer.get_vocab_size()) if token_id not in self.special_ids]

    def __len__(self):
        return len(self.data)


    def getspan(self, text):
        pattern = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
        spans = []
        for match in re.finditer(pattern, text):
            spans.append(match)
        return spans
    
    # Using pointers maybe keep them to fit the offset
    def clean_white_space(self, text, left, right):
        while left < right and text[left].isspace():
            left += 1

        while right > left and text[right - 1].isspace():
            right -= 1

        return left, right
    
    def overlaps(self, token_start, token_end, word_start, word_end):
        return token_start < word_end and token_end > token_start
    
    def build_word_groups(self, text, offsets):
        word_spans = self._get_word_spans(text)
        word_groups = [[] for _ in word_spans]

        for token_pos, (start, end) in enumerate(offsets):
            if start == end:
                continue

            start, end = self._trim_offset_whitespace(text, start, end)

            if start == end:
                continue

            token_text = text[start:end]

            # Skip punctuation-only token pieces.
            if not re.search(r"[A-Za-z0-9]", token_text):
                continue

            for word_index, (word_start, word_end) in enumerate(word_spans):
                if self._overlaps(start, end, word_start, word_end):
                    word_groups[word_index].append(token_pos)
                    break

        # Remove empty groups.
        word_groups = [group for group in word_groups if group]

        return word_groups
    
    def select_word_groups_to_mask(self, word_groups):
        total_maskable_tokens = sum(len(group) for group in word_groups)

        if total_maskable_tokens == 0:
            return []

        target_count = max(1, round(total_maskable_tokens * self.mask_probability))

        shuffled_groups = word_groups[:]
        random.shuffle(shuffled_groups)

        selected_positions = []
        selected_count = 0

        for group in shuffled_groups:
            if selected_count >= target_count:
                break

            selected_positions.extend(group)
            selected_count += len(group)

        return selected_positions
        
    def __getitem__(self, idx):
        

        # Adding of the [SEP] [CLS] [UNK] Tags to the synopses
        # Max len of synopsis is 325 will use 384 max_len 
        x_synopsis = self.data[idx]
        synopsis_token_object = self.tokenizer.encode(x_synopsis)
        synopsis_tokens = synopsis_token_object.tokens
        synopsis_ids = synopsis_token_object.ids
        synopsis_attention = synopsis_token_object.attention_mask
        synopsis_offsets = synopsis_token_object.offsets

        # Lossy will lose information on this (Maybe remove the entire entry? if so)
        if len(synopsis_ids) > (self.max_len - 2):
            synopsis_tokens = synopsis_tokens[:self.max_len-2]
            synopsis_offsets = synopsis_offsets[:self.max_len-2]
            synopsis_ids = synopsis_ids[:self.max_len-2]
        # Need to create a word grouping for dynamic masking because of BPE
        synopsis_word_group = self.build_word_groups(x_synopsis, synopsis_offsets)
        synopsis_word_group = [[pos + 1 for pos in group] for group in synopsis_word_group]
        
        synopsis_tokens = ["[CLS]"] + synopsis_tokens + ["[SEP]"]
        synopsis_ids = [self.cls_id] + synopsis_ids + [self.sep_id]
        original_ids = synopsis_ids.copy()
        synopsis_labels = [-100] * len(synopsis_ids)

        # Dynamically  randomly mask tokens based upon selected tokens
        mask_positions = self.select_word_groups_to_mask(synopsis_word_group)
        for position in mask_positions:
            origin_id = original_ids[position]

            if origin_id in self.normal_vocab_id:
                continue

            synopsis_labels[position] = origin_id
            probability = random.random()

            if probability < 0.80:
                synopsis_ids[position] = self.mask_id

            elif probability < 0.90:
                synopsis_ids[position] = random.choice

            else: 
                synopsis_ids[position] = origin_id


        # Padding the synopsis IDs
        padding_len = self.max_len - len(synopsis_ids) 
        synopsis_ids = synopsis_ids + [self.pad_id] * padding_len

        # Attention Mask
        attention_mask = [1] * len(synopsis_ids)
        attention_mask = attention_mask + [0] * padding_len

        # Adding labels
        synopsis_labels = synopsis_labels + [-100] * padding_len
        

        return {
            "input_ids": torch.tensor(synopsis_ids, dtype= torch.long),
            "label_ids": torch.tensor(synopsis_labels, dtype= torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype= torch.long)
            }








