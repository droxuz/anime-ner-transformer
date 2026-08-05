from mlm_dataset import MLMDataset, split_data, load_jsonl
from mlm_model import MLMTransformer 
from torch.utils.data import DataLoader
from pretrained_bpe import  train_bpe_tokenizer, get_bpe_tokenizer, load_bpe_tokenizer
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from pathlib import Path

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

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Paths
mal_synopses = "data/anime_training_data/mal_synopsis.jsonl"
BPE_tokenizer = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"
mlm_model = "data/anime_training_data/MLM_Model.pt"
# Data Splits
synopses = load_jsonl(mal_synopses)
train_data, val_data = split_data(synopses)

# BPE Tokenizer
tokenizer = get_bpe_tokenizer(train_data, BPE_tokenizer)
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_ID = tokenizer.token_to_id("[PAD]")

# Dataloading
training_dataset = MLMDataset(train_data, tokenizer, MAX_LEN, MASK_PROBABILITY, fixed_masking= False)
validation_dataset = MLMDataset(val_data, tokenizer, MAX_LEN, MASK_PROBABILITY, fixed_masking= True)

training_dataload = DataLoader(training_dataset, BATCH_SIZE, shuffle= True)
validation_dataload = DataLoader(validation_dataset, BATCH_SIZE, shuffle= False)


# Model
torch.manual_seed(321)
MLMModel = MLMTransformer(VOCAB_SIZE, MAX_LEN, PAD_ID, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)
entropyloss = nn.CrossEntropyLoss(ignore_index= -100)
optimizer = torch.optim.AdamW(MLMModel.parameters(), lr=LR, weight_decay= 0.01)

# Stalling in terms of learning need to change learning rate on stall
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor= 0.5, patience= 2)

# Training loop for training data
# Trains the model from Forward, Loss, Backward, and Optimize
# Returns average Losses 
def train_loop(model, train_load, entropyloss, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in train_load:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(input_ids, attention_mask)

        # Loss on the masked tokens ignores -100
        # Takes logits of batch, seq_len, vocab into batch * seq_len, vocab
        # Takes batch, seq_len into batch * seq_len
        loss = entropyloss(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_load)
    return avg_loss

# Validation loss calculations
# Takes validation dataloader, calculates crossentropyloss
# Returns average loss for comparison
def validation(model, val_load, entropyloss, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_load:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Model forward pass on validation data
            logits = model(input_ids, attention_mask)

            loss = entropyloss(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / len(val_load)
        return avg_loss
    
# Training Loop to train the model
# Loops through epochs to train the model and calculates training, validation losses 
def complete_training_loop(model, train_load, val_load, entropyloss, optimizer, scheduler, savepath, epochs, device):
    train_loss = []
    val_loss = []
    best_val_loss = float("inf")
    timeout = 0

    for epoch in range(epochs):

        if timeout > 4:
            break

        t_loss = train_loop(model, train_load, entropyloss, optimizer, device)
        train_loss.append(t_loss)

        v_loss = validation(model, val_load, entropyloss, device)
        val_loss.append(v_loss)
        scheduler.step(v_loss)

        # Select best model from epochs
        print(f"Epoch: {epoch}")
        print(f"Validation Loss: {v_loss:.4f}")
        print(f"Training Loss: {t_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            # Save best model
            torch.save(model.state_dict(), savepath)
            timeout = 0
        else:
            timeout += 1

    return train_loss, val_loss

def plot_losses(train_loss, val_loss):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses per epoch")
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.savefig("data/anime_training_data/training_model.png")
    plt.show()
    

train_loss, val_loss = complete_training_loop(MLMModel, training_dataload, validation_dataload, entropyloss, optimizer, scheduler, mlm_model, EPOCH, device)
plot_losses(train_loss, val_loss)
