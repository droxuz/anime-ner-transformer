from mlm_dataset import MLMDataset, split_data, load_jsonl
from mlm_model import MLMTransformer 
from torch.utils.data import DataLoader
from pretrained_bpe import  train_bpe_tokenizer, get_bpe_tokenizer, load_bpe_tokenizer
import torch.nn as nn
import torch
import matplotlib.pyplot as mp
from pathlib import Path

# Config
BATCH_SIZE = 64
MAX_LEN = 360
MASK_PROBABILITY = 0.15
D_MODEL = 512
NHEAD = 8
DIM_FEEDFORWARD = 2048
NUM_ENCODER_LAYERS = 8
DROPOUT = 0.1
EPOCH = 15
LR = 3e-4

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Paths
mal_synopses = "data/anime_training_data/mal_synopsis.jsonl"
BPE_tokenizer = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"
mlm_model = "data/anime_training_data/MLM_Model.pth"
# Data Splits
synopses = load_jsonl(mal_synopses)
train_data, val_data = split_data(synopses)

# BPE Tokenizer
tokenizer = get_bpe_tokenizer(train_data, BPE_tokenizer)
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_ID = tokenizer.token_to_id("[PAD]")

# Dataloading
training_dataset = MLMDataset(train_data, tokenizer, MAX_LEN)
validation_dataset = MLMDataset(val_data, tokenizer, MAX_LEN)
training_dataload = DataLoader(training_dataset, BATCH_SIZE, shuffle= True)
validation_dataload = DataLoader(validation_dataset, BATCH_SIZE, shuffle= True)


# Model
torch.manual_seed(321)
MLMModel = MLMTransformer(VOCAB_SIZE, MAX_LEN, PAD_ID, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT)
entropyloss = nn.CrossEntropyLoss(ignore_index= -100)
optimizer = torch.optim.AdamW(MLMTransformer.parameters(), lr=1e-4)

def training_model():
    
    for epoch in range(EPOCH):



