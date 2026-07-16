import torch
from torch.utils.data import Dataset

class MLMDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, mask_probability):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_probability = mask_probability
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cls_id = self.tokenizer.token_to_id("[CLS]")
        sep_id = self.tokenizer.token_to_id("[SEP]")
        pad_id = self.tokenizer.token_to_id("[PAD]")
        mask_id = self.tokenizer.token_to_id("[MASK]")
        unk_id = self.tokenizer.token_to_id("[UNK]")

        # Adding of the [SEP] [CLS] [UNK] Tags to the synopses
        # Max len of synopsis is 325 will use 384 max_len 
        x_synopsis = self.data[idx]
        synopsis_token_object = self.tokenizer.encode(x_synopsis)
        synopsis_tokens = synopsis_token_object.tokens
        synopsis_ids = synopsis_token_object.ids
        synopsis_attention = synopsis_token_object.attention_mask

        # Lossy will lose information on this (Maybe remove the entire entry? if so)
        if len(synopsis_ids) > (self.max_len - 2):
            synopsis_tokens = synopsis_tokens[:self.max_len-2]
            synopsis_ids = synopsis_ids[:self.max_len-2]
            synopsis_attention = synopsis_attention[:self.max_len-2]
        synopsis_ids = [cls_id] + synopsis_ids + [sep_id]
        synopsis_tokens = ["[CLS]"] + synopsis_tokens + ["[SEP]"]
        synopsis_token_length = len(synopsis_ids)

        # Add the padding to the entire max_len of the synopsis tokens
        padding_len = self.max_len - synopsis_token_length 
        padding_tokens = ["[PAD]"] * padding_len
        padding_ids = [pad_id] * padding_len
        synopsis_tokens = synopsis_tokens + padding_tokens
        synopsis_ids = synopsis_ids + padding_ids

        # Attention Masking
        attention_mask = ["-100"] * padding_len
        synopsis_attention = synopsis_attention + attention_mask

        return {
            "synopsis_tokens": torch.tensor(synopsis_tokens, dtype= torch.StringType), 
            "synopsis_ids": torch.tensor(synopsis_ids, dtype= torch.long),
            "synopsis_attention": torch.tensor(synopsis_attention, dtype= torch.long)
            }








