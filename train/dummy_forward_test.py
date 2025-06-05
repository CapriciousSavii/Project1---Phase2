import torch
from config.config import GPTConfig
from models.tiny_gpt import TinyGPT

config = GPTConfig()
model = TinyGPT(config)

x = torch.randint(0, config.vocab_size, (4, config.max_seq_len))  # (B, T)
logits = model(x)

print("Output shape:", logits.shape)  # should be (B, T, vocab_size)
