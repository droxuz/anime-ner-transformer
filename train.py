from dataset import load_jsonl, AnimeBPEDataset
from bpe_tokenizer import get_bpe_tokenizer
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

training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

torch.manual_seed(321)
tiny_model = TinyTransformer(vocab_size= trained_bpe.get_vocab_size(), num_labels= len(label_to_id), max_len= 100).to(device)

cross_entropy_weight = torch.tensor([
        0.5, 2, 1.5, 2, 1.5, 2, 1.5
    ], dtype = torch.float).to(device)

cross_entropy_loss = nn.CrossEntropyLoss(weight= cross_entropy_weight, ignore_index=-100)
optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)


def train_model(model, training_dataloader, val_dataloader, optimizer, loss_function, device, num_epochs, patience=3, save_path="best_model.pth"):
    best_val_loss = float("inf")
    no_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0

        for batch in training_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)

            loss = loss_function(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = (
            total_train_loss / len(training_dataloader)
        )

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)

                loss = loss_function(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1)
                )

                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0

            torch.save(model.state_dict(), save_path)
            print("Saved best model")
        else:
            no_improvement += 1

            if no_improvement >= patience:
                print("Early stopping")
                break

    # Reload the best checkpoint, not the final epoch
    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model, train_losses, val_losses
    
num_epochs = 10
tiny_model, train_losses, val_losses = train_model(model=tiny_model, training_dataloader=training_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, loss_function=cross_entropy_loss, device=device, num_epochs=num_epochs, patience=3, save_path="best_model.pth")
plt.plot(train_losses, label="Training Losses")
plt.plot(val_losses, label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()