class GPTConfig:
    def __init__(self,
                 vocab_size=1000,
                 max_seq_len=128,
                 embed_dim=128,
                 num_heads=4,
                 num_layers=4,
                 dropout=0.1):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
