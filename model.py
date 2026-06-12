import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_labels, max_len = 40, d_model = 128):
        super().__init__()
        
        # define layers here
        self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)# Creates a token representation with 128 values
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)# Creates a 0-39 position representation using 128 Values
        encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= 8, dim_feedforward= 256, dropout= 0.1, batch_first= True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers= 2)
        self.classifier = nn.Linear(d_model, num_labels)


    def forward(self, input_ids, attention_mask = None):
        # pass data through layers here
        pass