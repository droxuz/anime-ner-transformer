from dataset import load_jsonl
from finetune_dataset import FTDataset
from finetune_model import NERTransformer
from torch.utils.data import DataLoader
from pretrained_bpe import  train_bpe_tokenizer, get_bpe_tokenizer, load_bpe_tokenizer
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from pathlib import Path

BIO_LIB = {
    "O": 0,
    "B-TITLE": 1,
    "I-TITLE": 2,
    "B-ATTRIBUTE": 3,
    "I-ATTRIBUTE": 4,
    "B-FORMAT": 5,
    "I-FORMAT": 6,
    "B-CONSTRAINT": 7,
    "I-CONSTRAINT": 8
}

# Config
NUM_LABELS = len(BIO_LIB)
BATCH_SIZE = 32
MAX_LEN = 360
MASK_PROBABILITY = 0.15
D_MODEL = 256
NHEAD = 8
DIM_FEEDFORWARD = 1024
NUM_ENCODER_LAYERS = 6
DROPOUT = 0.2
EPOCH = 20
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

def load_mlm_encoder_weights(ner_model, mlm_checkpoint_path, device):
    checkpoint = torch.load(mlm_checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        mlm_state = checkpoint["model_state_dict"]
    else:
        mlm_state = checkpoint

    ner_state = ner_model.state_dict()
    filtered_state = {}

    for key, value in mlm_state.items():
        # Skip MLM prediction head
        if key.startswith("mlm_head"):
            continue
        # Only load matching names/shapes
        if key in ner_state and value.shape == ner_state[key].shape:
            filtered_state[key] = value

    ner_state.update(filtered_state)
    ner_model.load_state_dict(ner_state)

    print(f"Loaded {len(filtered_state)} pretrained encoder weights.")
    return ner_model

train_data = load_jsonl("data/anime_training_data/anime_ner_train_bio.jsonl")
val_data = load_jsonl("data/anime_training_data/anime_ner_val_bio.jsonl")
tokenizer_path = "data/anime_training_data/synopsis_pretrained_bpe_tokenizer"
tokenizer = load_bpe_tokenizer(tokenizer_path)

train_dataset = FTDataset(train_data, tokenizer, MAX_LEN, BIO_LIB)
val_dataset = FTDataset(val_data, tokenizer, MAX_LEN, BIO_LIB)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_ID = tokenizer.token_to_id("[PAD]")

ner_model = NERTransformer(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    pad_id=PAD_ID,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    num_labels=NUM_LABELS
).to(device)

ner_model = load_mlm_encoder_weights(ner_model, "data/anime_training_data/MLM_Best_Model.pt", device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(ner_model.parameters(), lr=LR, weight_decay=0.01)

def train_ner_loop(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate_ner_loop(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)

            mask = labels != -100

            correct += (predictions[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy

def fine_tune_ner(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path):
    best_val_loss = float("inf")
    patience = 8
    timeout = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        if timeout >= patience:
            print("Early stopping.")
            break

        train_loss = train_ner_loop(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate_ner_loop(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Token Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timeout = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "bio_lib": BIO_LIB,
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "max_len": MAX_LEN,
                        "pad_id": PAD_ID,
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "dim_feedforward": DIM_FEEDFORWARD,
                        "num_encoder_layers": NUM_ENCODER_LAYERS,
                        "dropout": DROPOUT,
                        "num_labels": NUM_LABELS,
                    }
                },
                save_path
            )

            print(f"Saved best NER model: {best_val_loss:.4f}")

        else:
            timeout += 1

    return train_losses, val_losses

train_losses, val_losses = fine_tune_ner(ner_model, train_loader, val_loader, criterion, optimizer, EPOCH, device, "data/anime_training_data/NER_Best_Model.pt")