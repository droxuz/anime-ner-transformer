from mlm_dataset import MLMDataset, split_data, load_jsonl
from mlm_model import MLMTransformer
from pretrained_bpe import load_bpe_tokenizer

from torch.utils.data import DataLoader
import torch
import random

# Config
BATCH_SIZE = 16
MAX_LEN = 360
MASK_PROBABILITY = 0.15
D_MODEL = 256
NHEAD = 8
DIM_FEEDFORWARD = 1024
NUM_ENCODER_LAYERS = 4
DROPOUT = 0.2
EPOCH = 30
LR = 3e-4

# Paths
BPE_tokenizer = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"
mlm_model_path = "data/anime_training_data/MLM_Model.pt"


# Tokenizer
tokenizer = load_bpe_tokenizer(BPE_tokenizer)
VOCAB_SIZE = tokenizer.get_vocab_size()
CLS_ID = tokenizer.token_to_id("[CLS]")
SEP_ID = tokenizer.token_to_id("[SEP]")
PAD_ID = tokenizer.token_to_id("[PAD]")
MASK_ID = tokenizer.token_to_id("[MASK]")
UNK_ID = tokenizer.token_to_id("[UNK]")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load Model
model = MLMTransformer(VOCAB_SIZE, MAX_LEN, PAD_ID, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)
mlm_model_state = torch.load(mlm_model_path)
model.load_state_dict(mlm_model_state)
model.eval()

example = "The young boy [MASK] on a journey around the world."

# Tokenize example, place boundary tokens, find MASK token, and predict
# Input: string, tokenizer
# Output: list of tokens
def prediction(example, tokenizer):
    tokens = tokenizer.encode(example)
    token_ids = tokens.ids[:MAX_LEN-2]
    inputs = [CLS_ID] + token_ids + [SEP_ID]

    mask_positions = inputs.index(MASK_ID) 

    attention_mask = [1] * len(inputs)
    padding_len = MAX_LEN - len(inputs)
       
    inputs = inputs + [PAD_ID] * padding_len
    attention_mask = attention_mask + [0] * padding_len

    inputs = torch.tensor([inputs], dtype= torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype= torch.long).to(device)

    with torch.no_grad():
        logits = model(inputs, attention_mask)

    mask_logits = logits[0, mask_positions]
    probability = torch.softmax(mask_logits, dim= -1)
    best_probabilities, best_choices = torch.topk(mask_logits, 10)

    for probability, choice in zip(best_probabilities, best_choices):
        choice = choice.item()
        token = tokenizer.decode([choice])
        raw = tokenizer.id_to_token(choice)
        print(f"{token!r:15s} raw={raw!r:15s} prob={probability.item():.6f}")

prediction(example, tokenizer)