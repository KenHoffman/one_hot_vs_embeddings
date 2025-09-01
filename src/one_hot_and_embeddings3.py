# two_word_lms_onehot_vs_embedding_shakespeare.py
# Python 3.10+ | pip install torch==2.*
# - Downloads Project Gutenberg Shakespeare (public domain in the US) to ./data/shakespeare.txt
# - Word-level tokenization (keeps punctuation and newlines as tokens)
# - Two GRU LMs: OneHot vs Embedding
# - Works on CUDA / MPS (Mac) / CPU

import os, pathlib, urllib.request, ssl, random, re
from dataclasses import dataclass
from typing import Optional, List
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Config
# -----------------------
@dataclass
class Config:
    block_size: int = 64                 # tokens of context
    batch_size: int = 16
    steps: int = 3000                    # increase (e.g., 20_000+) for better quality
    eval_every: int = 250
    lr: float = 3e-3
    hidden_size: int = 256               # keep modest for the one-hot model
    num_layers: int = 2
    embed_dim: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = 50
    data_dir: str = "data"
    cache_file: str = "shakespeare.txt"
    # Vocab control: keep top-N frequent tokens (rest -> <unk>)
    vocab_size_limit: int = 30000        # reduce if memory is tight
    min_freq: int = 1                    # raise to shrink vocab

cfg = Config()

# Device: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)
random.seed(42)

# -----------------------
# Data: Shakespeare (Project Gutenberg #100)
# -----------------------
GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/100/100-0.txt",
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
]

def _strip_gutenberg_boilerplate(text: str) -> str:
    start_token = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_token = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    lo = text.find(start_token)
    hi = text.find(end_token)
    if lo != -1 and hi != -1 and hi > lo:
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
            print(f"Saved to {dst_path} ({len(cleaned):,} chars).")
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
    print(f"Corpus loaded: {len(text):,} chars")
    return text

TEXT = load_corpus()

# -----------------------
# Tokenization (word-level)
#   - words incl. internal apostrophes (’ or ')
#   - numbers
#   - punctuation as separate tokens
#   - newlines as tokens
# -----------------------
TOK_REGEX = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)*"  # words with apostrophes
    r"|[0-9]+"                      # numbers
    r"|[^\w\s]"                     # punctuation (each char)
    r"|\n+"                         # one or more newlines as one token
)

def tokenize(text: str) -> List[str]:
    return TOK_REGEX.findall(text)

def detokenize(tokens: List[str]) -> str:
    # Simple detokenizer: attach some punctuation without a leading space; preserve newlines.
    no_space_before = set(list(".,!?:;)]}”’'\""))
    no_space_after = set(list("([“\""))
    out = []
    buf = ""
    for tok in tokens:
        if "\n" in tok:
            # flush buffer and add newlines directly
            if buf:
                out.append(buf)
                buf = ""
            out.append(tok)
            continue
        if not buf:
            buf = tok
        else:
            if tok in no_space_before:
                buf += tok
            elif buf and buf[-1:] in no_space_after:
                buf += tok
            else:
                buf += " " + tok
    if buf:
        out.append(buf)
    return "".join(out)

# -----------------------
# Vocab (word-level with <unk>)
# -----------------------
SPECIALS = ["<unk>"]

def build_vocab(tokens: List[str]):
    ctr = Counter(tokens)
    # apply min_freq and top-N cap (reserve 1 for <unk>)
    items = [(tok, freq) for tok, freq in ctr.items() if freq >= cfg.min_freq]
    # sort by freq desc, then token
    items.sort(key=lambda x: (-x[1], x[0]))
    max_keep = cfg.vocab_size_limit - len(SPECIALS)
    kept = items[:max_keep]
    vocab_tokens = SPECIALS + [tok for tok, _ in kept]
    stoi = {tok: i for i, tok in enumerate(vocab_tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos

TOKENS = tokenize(TEXT)
stoi, itos = build_vocab(TOKENS)
V = len(stoi)
UNK = stoi["<unk>"]
print(f"Vocab size (words): {V}")

def encode_tokens(tokens: List[str]) -> torch.Tensor:
    return torch.tensor([stoi.get(t, UNK) for t in tokens], dtype=torch.long)

def decode_ids(ids: torch.Tensor) -> str:
    toks = [itos[int(i)] for i in ids]
    return detokenize(toks)

DATA = encode_tokens(TOKENS)
n = int(0.95 * len(DATA))
train_ids, val_ids = DATA[:n], DATA[n:]

def get_batch(split: str):
    src = train_ids if split == "train" else val_ids
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
        # (B,T) -> one-hot (B,T,V) -> GRU -> logits (B,T,V)
        x = F.one_hot(idx, num_classes=V).float()
        out, h = self.rnn(x, h)
        logits = self.head(out)
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
        x = self.embed(idx)   # (B,T,E)
        out, h = self.rnn(x, h)
        logits = self.head(out)
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
def tokenize_prompt(prompt: str) -> List[str]:
    return tokenize(prompt)

@torch.no_grad()
def sample(model, prompt: str, max_new_tokens=100, temperature=cfg.temperature, top_k=cfg.top_k):
    model.eval()
    toks = tokenize_prompt(prompt) or ["ROMEO", ":"]
    idx = encode_tokens(toks).unsqueeze(0).to(device)
    h = None
    out_ids = [int(i) for i in idx[0]]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits, h = model(idx_cond, h)
        logits = logits[:, -1, :] / max(1e-8, temperature)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        out_ids.append(int(next_id))
        idx = torch.cat([idx, next_id], dim=1)

    return decode_ids(torch.tensor(out_ids))

# -----------------------
# Train both and demo
# -----------------------
if __name__ == "__main__":
    print(f"Device: {device} | Vocab: {V} | Tokens: {len(TOKENS):,}")

    print("\n=== Training One-Hot word model on Shakespeare ===")
    onehot_model = OneHotGRULM(V, cfg.hidden_size, cfg.num_layers)
    onehot_model = train_model(onehot_model, steps=cfg.steps)

    print("\n=== Training Embedding word model on Shakespeare ===")
    emb_model = EmbeddingGRULM(V, cfg.embed_dim, cfg.hidden_size, cfg.num_layers)
    emb_model = train_model(emb_model, steps=cfg.steps)

    # Generate from prompts
    prompts = [
        "ROMEO:",
        "To be, or not to be",
        "MACBETH:",
    ]

    for p in prompts:
        print("\n--- One-Hot continuation ---")
        print(sample(onehot_model, p, max_new_tokens=120))
        print("\n--- Embedding continuation ---")
        print(sample(emb_model, p, max_new_tokens=120))