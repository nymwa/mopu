import torch
import torch.nn as nn
from .embedding import TransformerEmbedding
from .block import AttentionSubLayer, FeedForwardSubLayer

class KuleAttentionLayer(AttentionSubLayer):
    def forward(self, x, mask, padding_mask):
        x = self.norm(x)
        x, score = self.attn(x, x, x, attn_mask = mask, key_padding_mask = padding_mask)
        x = self.dropout(x)
        return x, score

class KuleEncoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, attention_dropout, dropout):
        super().__init__()
        self.pre_feed_forward_layer = FeedForwardSubLayer(d_model, dim_feedforward, dropout)
        self.post_feed_forward_layer = FeedForwardSubLayer(d_model, dim_feedforward, dropout)
        self.attn_layer = KuleAttentionLayer(d_model, 1, attention_dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        x = self.pre_feed_forward_layer(x)
        x, score = self.attn_layer(x, None, padding_mask)
        x = self.post_feed_forward_layer(x)
        x = self.norm(x)
        return x, score

class SoweliKule(nn.Module):
    def __init__(self, d_vocab, d_model, dim_feedforward, attention_dropout, dropout):
        super().__init__()
        self.encoder = KuleEncoder(d_model, dim_feedforward, attention_dropout, dropout)
        self.embedding, self.projection = self.make_embeddings(d_vocab, d_model, dropout)

    def make_embeddings(self, d_vocab, d_model, dropout):
        embedding = TransformerEmbedding(d_vocab, d_model, dropout)
        projection = nn.Linear(d_model, d_vocab)
        projection.weight = embedding.token_embedding.weight
        return embedding, projection

    def encode(self, x, padding_mask=None):
        x = self.embedding(x)
        x, score = self.encoder(x, padding_mask = padding_mask)
        return x, score

    def forward(self, batch):
        x, _ = self.encode(batch.encoder_inputs,
                padding_mask = batch.encoder_padding_mask)
        x = self.projection(x)
        return x

