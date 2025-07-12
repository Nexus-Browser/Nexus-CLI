import torch
import torch.nn as nn

class NexusGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_embedding = nn.Parameter(torch.zeros(1, config["n_positions"], config["n_embd"]))
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config["n_embd"],
                nhead=config["n_head"],
                dim_feedforward=4*config["n_embd"]
            ) for _ in range(config["n_layer"])
        ])
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.head = nn.Linear(config["n_embd"], config["vocab_size"])

    def forward(self, idx):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)  # add batch dimension if missing
        elif idx.dim() > 2:
            idx = idx.squeeze(0)  # remove accidental extra dims

        

        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                   # [B, T, C]
        pos_emb = self.pos_embedding[:, :T, :]                # [1, T, C]
        x = tok_emb + pos_emb                                 # [B, T, C]

        x = x.transpose(0, 1)  # [T, B, C] for nn.Transformer
        for block in self.transformer_blocks:
            x = block(x)                                       # [T, B, C]
        x = x.transpose(0, 1)                                  # back to [B, T, C]

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens based on the input context.
        """
        for _ in range(max_new_tokens):
            # Get the last token's logits
            logits = self(idx)[:, -1, :] / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
