from dataset import load_jsonl, build_vocab, AnimePromptDataset
from model import TinyTransformer
from config import get_config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

training_path, testing_path, val_path, label_map_path, save_path, label_to_id = get_config()

training_data = load_jsonl(training_path)
vocab_lib = build_vocab(training_data)
val_data = load_jsonl(val_path)


testing_data = load_jsonl(testing_path)


training_dataset = AnimePromptDataset(training_data, vocab_lib, max_len= 40)
testing_dataset = AnimePromptDataset(testing_data, vocab_lib, max_len= 40)
val_dataset = AnimePromptDataset(val_data, vocab_lib, max_len=40)
#print(x) #Rows of the entire dataset
#print(testing) #3 Tensors for inputID, attentionMask, labels

training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

tiny_model = TinyTransformer(vocab_size= len(vocab_lib), num_labels= len(label_to_id), max_len= 40).to(device)

cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
num_epochs = 20
train_losses = []
val_losses = []

def unk_rate(data, vocab):
    total = 0
    unk = 0

    for row in data:
        for token in row["tokens"]:
            total += 1
            if token.lower() not in vocab:
                unk += 1

    return unk / total

print("Train UNK rate:", unk_rate(training_data, vocab_lib))
print("Val UNK rate:", unk_rate(val_data, vocab_lib))
print("Test UNK rate:", unk_rate(testing_data, vocab_lib))

for epoch in range(num_epochs):
    
    tiny_model.train()
    total_loss = 0

    for batch in training_dataloader:
        #Pulls tensors from batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = tiny_model(input_ids, attention_mask)
        loss = cross_entropy_loss(logits.reshape(-1, len(label_to_id)), labels.reshape(-1)) # per logit token create a 7 element mapping to each label, compare with the labels for loss

        loss.backward() # Backprop
        optimizer.step() # Updates weights

        total_loss += loss.item()
    
    avg_loss = total_loss / len(training_dataloader)
    # Validation
    tiny_model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = tiny_model(input_ids, attention_mask)

            loss = cross_entropy_loss(logits.reshape(-1, len(label_to_id)), labels.reshape(-1))

            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} | " f"Train loss: {avg_loss:.4f} | " f"Val loss: {avg_val_loss:.4f}")

plt.plot(train_losses, label="Training Losses")
plt.plot(val_losses, label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()