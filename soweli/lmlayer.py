import torch
import torch.nn as nn
from .block import SelfAttentionSubLayer, FeedForwardSubLayer

class TransformerLanguageModelDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attention_dropout, dropout):
        super().__init__()
        self.self_attn_layer = SelfAttentionSubLayer(d_model, nhead, attention_dropout)
        self.feed_forward_layer = FeedForwardSubLayer(d_model, dim_feedforward, dropout)

    def forward(self, x, attention_mask = None, padding_mask = None):
        x = self.self_attn_layer(x, attention_mask, padding_mask)
        x = self.feed_forward_layer(x)
        return x

class TransformerLanguageModelDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, attention_dropout, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLanguageModelDecoderLayer(d_model, nhead, dim_feedforward, attention_dropout, dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask = None, padding_mask = None):
        for layer in self.layers:
            x = layer(x, attention_mask, padding_mask)
        x = self.norm(x)
        return x

