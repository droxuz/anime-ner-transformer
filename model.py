import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_labels, max_len = 100, d_model = 128):
        super().__init__()
        
        # define layers here
        self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)# Creates lookup for tokens
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)# Creates Lookup for positional
        encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= 8, dim_feedforward= 256, dropout= 0.2, batch_first= True)# Singular encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers= 2)# Creates a encoder of 2 encoding layers
        self.classifier = nn.Linear(d_model, num_labels)# Classifier to use with the labels from data


    def forward(self, input_ids, attention_mask = None):
        # pass data through layers here
        rows, columns = input_ids.shape

        token_embed = self.token_embeddings(input_ids)# Creates a token representation with 128 values
        position_id = torch.arange(columns,device= input_ids.device)
        position_id = position_id.unsqueeze(0).expand(rows, columns)
        position_embed = self.position_embeddings(position_id)# Creates a 0 to max_len position representation using 128 Values

        #Combine the position and token embeds
        pos_tok_embedding = token_embed + position_embed

        #Place into the encoder layer
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        encoder_data = self.encoder(pos_tok_embedding, src_key_padding_mask = key_padding_mask)

        #Utilize the classifier on data
        logits = self.classifier(encoder_data)
    
        return logits