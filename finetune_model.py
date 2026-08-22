import torch
import torch.nn as nn


class NERTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        pad_id,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        num_labels
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Replaces MLM head
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        src_key_padding_mask = attention_mask == 0

        encoded = self.encoder_stack(x, src_key_padding_mask=src_key_padding_mask)

        logits = self.classifier(encoded)

        return logits