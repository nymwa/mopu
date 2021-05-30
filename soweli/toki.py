import torch
import torch.nn as nn
from .embedding import TransformerEmbedding
from .lmlayer import TransformerLanguageModelDecoder

class SoweliToki(nn.Module):
    def __init__(self, d_vocab, d_model, nhead, dim_feedforward,
            num_layers, attention_dropout, dropout):
        super().__init__()
        self.decoder = TransformerLanguageModelDecoder(d_model, nhead, dim_feedforward, num_layers, attention_dropout, dropout)
        self.embedding, self.projection = self.make_embeddings(d_vocab, d_model, dropout)

    def make_embeddings(self, d_vocab, d_model, dropout):
        embedding = TransformerEmbedding(d_vocab, d_model, dropout)
        projection = nn.Linear(d_model, d_vocab)
        projection.weight = embedding.token_embedding.weight
        return embedding, projection

    def decode(self, x, attention_mask = None, padding_mask = None):
        x = self.embedding(x)
        x = self.decoder(x, attention_mask = attention_mask, padding_mask = padding_mask)
        x = self.projection(x)
        return x

    def forward(self, batch):
        x = self.decode(batch.inputs,
                attention_mask = batch.attention_mask,
                padding_mask = batch.padding_mask)
        return x


