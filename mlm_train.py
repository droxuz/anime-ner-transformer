from mlm_dataset import MLMDataset, split_data, load_jsonl
from mlm_model import MLMTransformer 
from torch.utils.data import DataLoader
from pretrained_bpe import  train_bpe_tokenizer, get_bpe_tokenizer, load_bpe_tokenizer
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Config
BATCH_SIZE = 32
MAX_LEN = 360
MASK_PROBABILITY = 0.15
D_MODEL = 256
NHEAD = 8
DIM_FEEDFORWARD = 1024
NUM_ENCODER_LAYERS = 6
DROPOUT = 0.2
EPOCH = 50
LR = 3e-4
config = {
    "BATCH_SIZE" : BATCH_SIZE,
    "MAX_LEN" : MAX_LEN,
    "MASK_PROBABILITY" : MASK_PROBABILITY,
    "D_MODEL" : D_MODEL,
    "NHEAD" : NHEAD,
    "DIM_FEEDFORWARD" : DIM_FEEDFORWARD,
    "NUM_ENCODER_LAYERS" : NUM_ENCODER_LAYERS,
    "DROPOUT" : DROPOUT,
    "EPOCH" : EPOCH,
    "LR" : LR
}

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Paths
mal_synopses = "data/anime_training_data/mal_synopsis.jsonl"
BPE_tokenizer = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"
latest_model = "data/anime_training_data/MLM_Latest_Model.pt"
best_model = "data/anime_training_data/MLM_Best_Model.pt"

# Data Splits
synopses = load_jsonl(mal_synopses)
train_data, val_data = split_data(synopses)

# BPE Tokenizer
tokenizer = get_bpe_tokenizer(train_data, BPE_tokenizer)
print(tokenizer.get_vocab_size())
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor= 0.5, patience= 3, threshold= 0.01, threshold_mode= "abs", cooldown= 1, min_lr=5e-5)

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
        loss = entropyloss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
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

            loss = entropyloss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()

        avg_loss = total_loss / len(val_load)
        return avg_loss

def load_best(best_model):
    path = Path(best_model)
    if not path.exists():
        return float("inf")
    best_checkpoint = torch.load(best_model,map_location= "cpu")
    if isinstance(best_checkpoint, dict) and "best_val_loss" in best_checkpoint:
        return best_checkpoint["best_val_loss"]

    return float("inf")

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, train_loss, val_loss, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None, "best_val_loss": best_val_loss, "train_loss": train_loss, "val_loss": val_loss, "config": config}, path)

# Training Loop to train the model
# Loops through epochs to train the model and calculates training, validation losses 
def complete_training_loop(model, train_load, val_load, entropyloss, optimizer, scheduler, latestpath, bestpath, epochs, config, device):
    train_loss = []
    val_loss = []

    # Best loss from previous saved runs
    global_best_val_loss = load_best(bestpath)

    # Best loss from this current run only
    run_best_val_loss = float("inf")

    timeout = 0
    patience = 8
    min_delta = 0.01

    for epoch in range(epochs):

        if timeout >= patience:
            print("Early stopping.")
            break

        t_loss = train_loop(model, train_load, entropyloss, optimizer, device)
        train_loss.append(t_loss)

        v_loss = validation(model, val_load, entropyloss, device)
        val_loss.append(v_loss)

        if scheduler is not None:
            scheduler.step(v_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Current LR: {current_lr}")
        print(f"Epoch: {epoch}")
        print(f"Validation Loss: {v_loss:.4f}")
        print(f"Training Loss: {t_loss:.4f}")

        # Always save latest checkpoint from current run
        save_checkpoint(bestpath, model, optimizer, scheduler, epoch, global_best_val_loss, train_loss, val_loss, config)

        # Early stopping should compare only against this run's best
        if v_loss < run_best_val_loss - min_delta:
            run_best_val_loss = v_loss
            timeout = 0
        else:
            timeout += 1

        # Global best checkpoint should only update if this run beats old saved best
        if v_loss < global_best_val_loss:
            global_best_val_loss = v_loss

            save_checkpoint(bestpath, model, optimizer, scheduler, epoch, global_best_val_loss, train_loss, val_loss, config)

            print(f"Saved new global best model: {global_best_val_loss:.4f}")

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
    
PROFILE_MEMORY = False
MEM_EPOCH = 1
if PROFILE_MEMORY and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
        max_entries=100000,
        clear_history=True
    )

try:
    train_loss, val_loss = complete_training_loop(MLMModel, training_dataload, validation_dataload, entropyloss, optimizer, scheduler, latest_model, best_model, MEM_EPOCH if PROFILE_MEMORY else EPOCH, config ,device)

finally:
    if PROFILE_MEMORY and torch.cuda.is_available():
        torch.cuda.synchronize()

        print("Peak allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)
        print("Peak reserved GB:", torch.cuda.max_memory_reserved() / 1024**3)

        torch.cuda.memory._dump_snapshot("data/anime_training_data/mem_alloc.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
if not PROFILE_MEMORY:
    plot_losses(train_loss, val_loss)


