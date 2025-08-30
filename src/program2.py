# Compare two language models: one using one-hot inputs, the other using embeddings.
# Generated vy ChatGPT GPT-5 Thinking 2025-08-30.
# Modified by Ken to use mps.

# two_lms_onehot_vs_embedding_shakespeare.py
# Python 3.10+ | pip install torch==2.*
# Notes:
# - Downloads the complete works of Shakespeare from Project Gutenberg (public domain in the US).
# - Caches in ./data/shakespeare.txt for reuse.
# - Character-level GRU LMs: one-hot vs. embedding.

import os, pathlib, urllib.request, ssl, random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Config
# -----------------------
@dataclass
class Config:
    block_size: int = 256
    batch_size: int = 64
    steps: int = 4000          # increase for better results; try 20_000+ if you have time/compute
    eval_every: int = 250
    lr: float = 3e-3
    hidden_size: int = 512
    num_layers: int = 2
    embed_dim: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = 50
    data_dir: str = "data"
    cache_file: str = "shakespeare.txt"

cfg = Config()
#device = "cuda" if torch.cuda.is_available() else "cpu"
# Ken replaced previous line with the following to support MPS (Mac).
device = "mps" if torch.backends.mps.is_available() \
    else "cuda" if torch.cuda.is_available() \
    else "cpu"
print (f"Using device: {device}")

torch.manual_seed(42)
random.seed(42)

# -----------------------
# Data: Shakespeare (Project Gutenberg #100)
# -----------------------
GUTENBERG_URLS = [
    # UTF-8 text, preferred
    "https://www.gutenberg.org/files/100/100-0.txt",
    # Alternate cache location
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
]

def _strip_gutenberg_boilerplate(text: str) -> str:
    # Remove Project Gutenberg header/footer if present
    start_token = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_token = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    lo = text.find(start_token)
    hi = text.find(end_token)
    if lo != -1 and hi != -1 and hi > lo:
        # find the end of the line for start, start of the line for end
        start_idx = text.find("\n", lo)
        end_idx = text.rfind("\n", 0, hi)
        if start_idx != -1 and end_idx != -1:
            return text[start_idx+1:end_idx].strip()
    return text

def _download_shakespeare(dst_path: pathlib.Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    last_err = None
    for url in GUTENBERG_URLS:
        try:
            print(f"Downloading Shakespeare corpus from {url} ...")
            with urllib.request.urlopen(url, context=ctx, timeout=60) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            cleaned = _strip_gutenberg_boilerplate(raw)
            with open(dst_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"Saved to {dst_path} ({len(cleaned):,} characters).")
            return
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(f"Could not download corpus from known URLs. Last error: {last_err}")

def load_corpus() -> str:
    path = pathlib.Path(cfg.data_dir) / cfg.cache_file
    if not path.exists():
        _download_shakespeare(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Optional: downsample for quick tests by uncommenting:
    # text = text[:1_000_000]
    print(f"Corpus loaded: {len(text):,} characters.")
    return text

TEXT = load_corpus()

# -----------------------
# Vocab (character-level)
# -----------------------
vocab = sorted(list(set(TEXT)))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
V = len(vocab)
print(f"Vocab size (chars): {V}")

def encode(s: str):
    return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)

def decode(ids: torch.Tensor):
    return "".join(itos[int(i)] for i in ids)

# Data as indices
data = encode(TEXT)
n = int(0.95 * len(data))
train_ids, val_ids = data[:n], data[n:]

def get_batch(split: str):
    src = train_ids if split == "train" else val_ids
    # ensure we can draw a batch even on very small splits
    max_start = max(1, len(src) - cfg.block_size - 1)
    ix = torch.randint(max_start, (cfg.batch_size,))
    x = torch.stack([src[i:i+cfg.block_size] for i in ix])
    y = torch.stack([src[i+1:i+cfg.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------
# Models
# -----------------------
class OneHotGRULM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=vocab_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, h=None):
        x = F.one_hot(idx, num_classes=V).float()  # (B,T,V)
        out, h = self.rnn(x, h)                    # (B,T,H)
        logits = self.head(out)                    # (B,T,V)
        return logits, h

class EmbeddingGRULM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, h=None):
        x = self.embed(idx)              # (B,T,E)
        out, h = self.rnn(x, h)          # (B,T,H)
        logits = self.head(out)          # (B,T,V)
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
            print(f"step {step:5d} | train {el['train']:.3f} | val {el['val']:.3f}")
    return model

# -----------------------
# Sampling
# -----------------------
@torch.no_grad()
def sample(model, prompt: str, max_new_tokens=400, temperature=cfg.temperature, top_k=cfg.top_k):
    model.eval()
    if not prompt:
        prompt = "ROMEO:\n"
    idx = encode(prompt)
    if len(idx) == 0:  # if unknown chars stripped, seed with a space
        idx = torch.tensor([stoi.get(" ", 0)], dtype=torch.long)
    idx = idx.unsqueeze(0).to(device)  # (1,T)
    h = None
    out_ids = [int(i) for i in idx[0]]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits, h = model(idx_cond, h)  # (1,T,V)
        logits = logits[:, -1, :] / max(1e-8, temperature)
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
    print(f"Device: {device}")

    print("\n=== Training One-Hot model on Shakespeare ===")
    onehot_model = OneHotGRULM(V, cfg.hidden_size, cfg.num_layers)
    onehot_model = train_model(onehot_model, steps=cfg.steps)

    print("\n=== Training Embedding model on Shakespeare ===")
    emb_model = EmbeddingGRULM(V, cfg.embed_dim, cfg.hidden_size, cfg.num_layers)
    emb_model = train_model(emb_model, steps=cfg.steps)

    # Generate from prompts
    prompts = [
        "ROMEO:\n",
        "To be, or not to be",
        "MACBETH:\n",
    ]

    for p in prompts:
        print("\n--- One-Hot continuation ---")
        print(sample(onehot_model, p, max_new_tokens=400))
        print("\n--- Embedding continuation ---")
        print(sample(emb_model, p, max_new_tokens=400))