import torch
import torch.nn as nn
import math

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim)
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask=None):
        # Multi-head self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        # Feedforward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        return self.ln2(x)

class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.max_seq_len = config.max_seq_len

    def forward(self, x):
        B, T = x.size()
        assert T <= self.max_seq_len, "Input too long"

        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)

        attn_mask = torch.tril(torch.ones(T, T)).to(x.device)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        x = self.blocks(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        return self.head(x)
