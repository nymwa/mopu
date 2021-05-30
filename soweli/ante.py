import torch
import torch.nn as nn
from .embedding import TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder

class SoweliAnte(nn.Module):
    def __init__(self, d_vocab, d_model, nhead, dim_feedforward,
            num_encoder_layers, num_decoder_layers, attention_dropout, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, nhead, dim_feedforward, num_encoder_layers, attention_dropout, dropout)
        self.decoder = Decoder(d_model, nhead, dim_feedforward, num_decoder_layers, attention_dropout, dropout)
        self.embedding, self.projection = self.make_embeddings(d_vocab, d_model, dropout)

    def make_embeddings(self, d_vocab, d_model, dropout):
        embedding = TransformerEmbedding(d_vocab, d_model, dropout)
        projection = nn.Linear(d_model, d_vocab)
        projection.weight = embedding.token_embedding.weight
        return embedding, projection

    def encode(self, x, padding_mask=None):
        x = self.embedding(x)
        x = self.encoder(x, padding_mask = padding_mask)
        return x

    def decode(self, x, mem,
            attention_mask = None,
            encoder_padding_mask = None,
            decoder_padding_mask = None):
        x = self.embedding(x)
        x = self.decoder(x, mem,
                attention_mask = attention_mask,
                encoder_padding_mask = encoder_padding_mask,
                decoder_padding_mask = decoder_padding_mask)
        x = self.projection(x)
        return x

    def forward(self, batch):
        mem = self.encode(batch.encoder_inputs, padding_mask = batch.encoder_padding_mask)
        x = self.decode(batch.decoder_inputs, mem,
                attention_mask = batch.attention_mask,
                encoder_padding_mask = batch.encoder_padding_mask,
                decoder_padding_mask = batch.decoder_padding_mask)
        return x

