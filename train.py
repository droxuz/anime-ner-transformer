from dataset import load_jsonl, AnimePromptDataset, AnimeBPEDataset
from bpe_tokenizer import train_bpe_tokenizer, get_bpe_tokenizer
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
val_data = load_jsonl(val_path)
testing_data = load_jsonl(testing_path)

trained_bpe = get_bpe_tokenizer(training_data, save_path, vocab_size = 8000)

training_dataset = AnimeBPEDataset(training_data, tokenizer= trained_bpe, label_to_id= label_to_id, max_len = 100)
testing_dataset = AnimeBPEDataset(testing_data, tokenizer= trained_bpe, label_to_id= label_to_id, max_len= 100)
val_dataset = AnimeBPEDataset(val_data, tokenizer= trained_bpe, label_to_id= label_to_id, max_len= 100)

#training_dataset = AnimePromptDataset(training_data, vocab_lib, max_len= 40)
#testing_dataset = AnimePromptDataset(testing_data, vocab_lib, max_len= 40)
#val_dataset = AnimePromptDataset(val_data, vocab_lib, max_len=40)
#print(x) #Rows of the entire dataset
#print(testing) #3 Tensors for inputID, attentionMask, labels

training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

torch.manual_seed(321)
tiny_model = TinyTransformer(vocab_size= trained_bpe.get_vocab_size(), num_labels= len(label_to_id), max_len= 100).to(device)

cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
num_epochs = 20
train_losses = []
val_losses = []

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

    #Val Loss reduce overfitting. 
    

plt.plot(train_losses, label="Training Losses")
plt.plot(val_losses, label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()