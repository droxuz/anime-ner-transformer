import torch 
import torch.nn as nn


class MLMTransformer(nn.module):
    def __init__(self, vocab_size, max_len, pad_id, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()

        # Embeddings, Layers, and Layer stack
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx= pad_id)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first= True)
        self.encoder_stack = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):

        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device= input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        token_embed = self.token_embeddings(input_ids)
        positions_embed = self.position_embeddings(positions)
        x = token_embed + positions_embed
        x = self.dropout(x)
        src_key_padding_mask = attention_mask == 0
        encoded = self.encoder_stack(x, src_key_padding_mask = src_key_padding_mask)
        logits = self.mlm_head(encoded)

        return logits