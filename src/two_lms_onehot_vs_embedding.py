# Compare two language models: one using one-hot inputs, the other using embeddings.
# Generated vy ChatGPT GPT-5 Thinking 2025-08-29.
# Modified by Ken to use mps.

# two_lms_onehot_vs_embedding.py
# pip install torch==2.*  (PyTorch CPU is fine)

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math, random

# -----------------------
# Config (tweak freely)
# -----------------------
@dataclass
class Config:
    block_size: int = 128     # context length
    batch_size: int = 64
    steps: int = 2000         # training iterations per model
    eval_every: int = 200
    lr: float = 3e-3
    hidden_size: int = 256
    num_layers: int = 2
    embed_dim: int = 128      # only used by embedding model
    temperature: float = 1.0
    top_k: int | None = 50

cfg = Config()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# Ken replaced previous line with the following to support MPS (Mac).
device = "mps" if torch.backends.mps.is_available() \
    else "cuda" if torch.cuda.is_available() \
    else "cpu"
print (f"Using device: {device}")

torch.manual_seed(42)

# -----------------------
# Toy corpus (replace with your own)
# -----------------------
TEXT = (
           "To be, or not to be, that is the question:\n"
           "Whether 'tis nobler in the mind to suffer\n"
           "The slings and arrows of outrageous fortune,\n"
           "Or to take arms against a sea of troubles\n"
           "And by opposing end them.\n"
       ) * 200  # repeat to make it less tiny

# Build char vocab
vocab = sorted(list(set(TEXT)))
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for ch, i in stoi.items()}
V = len(vocab)

def encode(s: str):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids: torch.Tensor):
    return "".join(itos[int(i)] for i in ids)

# Data as indices
data = encode(TEXT)
n = int(0.9 * len(data))
train_ids, val_ids = data[:n], data[n:]

def get_batch(split: str):
    src = train_ids if split == "train" else val_ids
    ix = torch.randint(len(src) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([src[i:i+cfg.block_size] for i in ix])
    y = torch.stack([src[i+1:i+cfg.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------
# One-hot model
# -----------------------
class OneHotGRULM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        # GRU takes one-hot directly (input_size = vocab_size)
        self.rnn = nn.GRU(input_size=vocab_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, h=None):
        # idx: (B, T) int64
        x = F.one_hot(idx, num_classes=V).float()  # (B,T,V)
        out, h = self.rnn(x, h)                    # (B,T,H)
        logits = self.head(out)                    # (B,T,V)
        return logits, h

# -----------------------
# Embedding model
# -----------------------
class EmbeddingGRULM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(input_size=embed_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, h=None):
        x = self.embed(idx)             # (B,T,E)
        out, h = self.rnn(x, h)         # (B,T,H)
        logits = self.head(out)         # (B,T,V)
        return logits, h

# -----------------------
# Training / evaluation
# -----------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(5):
            xb, yb = get_batch(split)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

def train_model(model, steps: int):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for step in range(1, steps + 1):
        xb, yb = get_batch("train")
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.eval_every == 0 or step == 1:
            el = estimate_loss(model)
            print(f"step {step:4d} | train {el['train']:.3f} | val {el['val']:.3f}")
    return model

# -----------------------
# Sampling
# -----------------------
@torch.no_grad()
def sample(model, prompt: str, max_new_tokens=400, temperature=cfg.temperature, top_k=cfg.top_k):
    model.eval()
    idx = encode(prompt).unsqueeze(0).to(device)  # (1,T)
    h = None
    out_ids = [int(i) for i in idx[0]]

    for _ in range(max_new_tokens):
        # keep only last block_size tokens for conditioning
        idx_cond = idx[:, -cfg.block_size:]

        logits, h = model(idx_cond, h)  # logits: (1,T,V)
        logits = logits[:, -1, :] / max(1e-8, temperature)  # last step
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)

        out_ids.append(int(next_id))
        idx = torch.cat([idx, next_id], dim=1)

    return decode(torch.tensor(out_ids))

# -----------------------
# Train both and demo
# -----------------------
if __name__ == "__main__":
    print(f"Vocab size: {V}, device: {device}")

    print("\n=== Training One-Hot model ===")
    onehot_model = OneHotGRULM(V, cfg.hidden_size, cfg.num_layers)
    onehot_model = train_model(onehot_model, steps=cfg.steps)

    print("\n=== Training Embedding model ===")
    emb_model = EmbeddingGRULM(V, cfg.embed_dim, cfg.hidden_size, cfg.num_layers)
    emb_model = train_model(emb_model, steps=cfg.steps)

    # Generate from a prompt
    prompt = "To be"
    print("\n--- One-Hot continuation ---")
    print(sample(onehot_model, prompt, max_new_tokens=300))

    print("\n--- Embedding continuation ---")
    print(sample(emb_model, prompt, max_new_tokens=300))