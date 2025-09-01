# word_lm_onehot_vs_embedding_shakespeare_glove.py
# Python 3.10+ | pip install torch==2.*
# - Shakespeare corpus (Gutenberg) cached to ./data/shakespeare.txt
# - Word-level tokenization
# - Two GRU LMs: One-Hot vs Embedding
# - Optional GloVe initialization, weight tying, and co-occurrence similarity loss
# - Works on CUDA / MPS (Mac) / CPU

import os, pathlib, urllib.request, ssl, random, re, zipfile, io
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Config
# -----------------------
@dataclass
class Config:
    block_size: int = 64
    batch_size: int = 16
    steps: int = 3000
    eval_every: int = 250
    lr: float = 3e-3
    hidden_size: int = 256
    num_layers: int = 2
    embed_dim: int = 200               # set to match chosen GloVe dim if tying weights
    temperature: float = 1.0
    top_k: Optional[int] = 50
    data_dir: str = "../data"
    cache_file: str = "shakespeare.txt"

    # Vocab control
    vocab_size_limit: int = 30000
    min_freq: int = 1

    # Embedding improvements
    use_glove: bool = True            # try pretrained embeddings
    glove_dim: int = 200              # one of {50,100,200,300} for glove.6B.*d.txt
    glove_freeze_steps: int = 0       # e.g., 500–2000 to warm up with frozen embeddings
    glove_finetune: bool = True       # fine-tune embeddings after warmup
    tie_weights: bool = True          # share input embedding and output softmax weights

    # Optional similarity regularizer (co-occurrence window)
    coocc_loss_weight: float = 0.0    # try 0.01–0.05 to gently nudge; 0.0 disables
    coocc_window: int = 2             # +/- window used to form positive pairs

cfg = Config()

# Device selection: CUDA -> MPS -> CPU
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
    start = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    lo = text.find(start)
    hi = text.find(end)
    if lo != -1 and hi != -1 and hi > lo:
        s = text.find("\n", lo)
        e = text.rfind("\n", 0, hi)
        if s != -1 and e != -1:
            return text[s+1:e].strip()
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
    raise RuntimeError(f"Could not download corpus. Last error: {last_err}")

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
# -----------------------
TOK_REGEX = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)*"  # words with apostrophes
    r"|[0-9]+"
    r"|[^\w\s]"
    r"|\n+"
)

def tokenize(text: str) -> List[str]:
    return TOK_REGEX.findall(text)

def detokenize(tokens: List[str]) -> str:
    no_space_before = set(list(".,!?:;)]}”’'\""))
    no_space_after = set(list("([“\""))
    out, buf = [], ""
    for tok in tokens:
        if "\n" in tok:
            if buf: out.append(buf); buf = ""
            out.append(tok); continue
        if not buf:
            buf = tok
        else:
            if tok in no_space_before:
                buf += tok
            elif buf and buf[-1:] in no_space_after:
                buf += tok
            else:
                buf += " " + tok
    if buf: out.append(buf)
    return "".join(out)

# -----------------------
# Vocab (with <unk>)
# -----------------------
SPECIALS = ["<unk>"]

def build_vocab(tokens: List[str]) -> Tuple[Dict[str,int], Dict[int,str]]:
    ctr = Counter(tokens)
    items = [(tok, f) for tok, f in ctr.items() if f >= cfg.min_freq]
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
# GloVe loading (optional)
# -----------------------
# Downloads glove.6B.zip and extracts the requested dimension file.
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"

def load_glove_embeddings(dim: int, vocab: Dict[str,int]) -> torch.Tensor:
    cache_dir = pathlib.Path(cfg.data_dir) / "glove"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "glove.6B.zip"
    txt_name = f"glove.6B.{dim}d.txt"
    txt_path = cache_dir / txt_name

    if not txt_path.exists():
        # download + extract just the target file
        if not zip_path.exists():
            print(f"Downloading {GLOVE_URL} ... (~822MB)")
            with urllib.request.urlopen(GLOVE_URL, timeout=300) as r:
                data = r.read()
            with open(zip_path, "wb") as f:
                f.write(data)
        print(f"Extracting {txt_name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(txt_name) as src, open(txt_path, "wb") as dst:
                dst.write(src.read())

    # build embedding matrix
    emb = torch.empty((len(vocab), dim)).normal_(mean=0.0, std=0.02)
    found = 0
    # read GloVe
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = parts[1:]
            if len(vec) != dim: continue
            if word in vocab:
                emb[vocab[word]] = torch.tensor([float(x) for x in vec])
                found += 1
            # try lowercase match
            lw = word.lower()
            if lw in vocab and emb[vocab[lw]].abs().sum().item() == 0.0:
                emb[vocab[lw]] = torch.tensor([float(x) for x in vec])
                found += 1
    print(f"GloVe init: matched {found} / {len(vocab)} tokens.")
    return emb

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
        x = F.one_hot(idx, num_classes=V).float()
        out, h = self.rnn(x, h)
        logits = self.head(out)
        return logits, h

class EmbeddingGRULM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int, tie_weights: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
        )
        # If tying, hidden_size must equal embed_dim or we project
        if tie_weights and hidden_size != embed_dim:
            self.proj = nn.Linear(hidden_size, embed_dim, bias=False)
            out_dim = embed_dim
        else:
            self.proj = None
            out_dim = hidden_size

        self.head = nn.Linear(out_dim, vocab_size, bias=False)
        self._tie = tie_weights

    def tie_weights_if_possible(self):
        if self._tie:
            if self.proj is None and self.embed.weight.shape[1] == self.head.weight.shape[0]:
                # share weights directly
                self.head.weight = self.embed.weight
            # else: with projection, cannot tie directly; we still benefit from lower out_dim

    def forward(self, idx, h=None):
        x = self.embed(idx)        # (B,T,E)
        out, h = self.rnn(x, h)    # (B,T,H)
        if self.proj is not None:
            out = self.proj(out)   # (B,T,E)
        logits = self.head(out)    # (B,T,V)
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

def _cooccurrence_pairs(x: torch.Tensor, window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: (B,T) integer token ids
    B, T = x.shape
    centers, contexts = [], []
    for shift in range(1, window+1):
        cs = x[:, :-shift].reshape(-1)
        ctx = x[:, shift:].reshape(-1)
        centers.append(cs); contexts.append(ctx)
        cs = x[:, shift:].reshape(-1)
        ctx = x[:, :-shift].reshape(-1)
        centers.append(cs); contexts.append(ctx)
    return torch.cat(centers), torch.cat(contexts)

def cooccurrence_loss(model: nn.Module, x: torch.Tensor, weight: float, window: int) -> torch.Tensor:
    # Only for embedding model
    if weight <= 0.0 or not hasattr(model, "embed"):
        return torch.tensor(0.0, device=x.device)
    centers, contexts = _cooccurrence_pairs(x, window)
    e = model.embed
    c_vecs = F.normalize(e(centers), dim=-1)
    ctx_vecs = F.normalize(e(contexts), dim=-1)
    # Maximize cosine similarity => minimize (1 - cos)
    loss = (1.0 - (c_vecs * ctx_vecs).sum(dim=-1)).mean()
    return weight * loss

def train_model(model, steps: int, freeze_embed_steps: int = 0, finetune_embeddings: bool = True):
    model = model.to(device)
    model.train()
    model.tie_weights_if_possible() if hasattr(model, "tie_weights_if_possible") else None

    # Split params for warmup (if freezing embeddings)
    if hasattr(model, "embed") and freeze_embed_steps > 0:
        embed_params = list(model.embed.parameters())
        other_params = [p for p in model.parameters() if p not in embed_params]
        opt = torch.optim.AdamW(other_params, lr=cfg.lr)
        opt2 = torch.optim.AdamW(model.parameters(), lr=cfg.lr) if finetune_embeddings else None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        opt2 = None

    for step in range(1, steps + 1):
        xb, yb = get_batch("train")
        logits, _ = model(xb)
        nll = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))

        # optional similarity regularizer for embedding model
        sim_loss = cooccurrence_loss(model, xb, cfg.coocc_loss_weight, cfg.coocc_window)
        loss = nll + sim_loss

        opt.zero_grad(set_to_none=True)
        if opt2: opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # choose optimizer depending on warmup phase
        if hasattr(model, "embed") and freeze_embed_steps > 0 and step <= freeze_embed_steps:
            opt.step()
        else:
            if opt2:
                opt2.step()
            else:
                opt.step()

        if step % cfg.eval_every == 0 or step == 1:
            el = estimate_loss(model)
            print(f"step {step:5d} | loss {loss.item():.3f} | train {el['train']:.3f} | val {el['val']:.3f}")

    return model

# -----------------------
# Sampling
# -----------------------
def tokenize_prompt(prompt: str) -> List[str]:
    toks = TOK_REGEX.findall(prompt)
    return toks if toks else ["ROMEO", ":"]

@torch.no_grad()
def sample(model, prompt: str, max_new_tokens=100, temperature=cfg.temperature, top_k=cfg.top_k):
    model.eval()
    idx = encode_tokens(tokenize_prompt(prompt)).unsqueeze(0).to(device)
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
        next_id = torch.multinomial(probs, num_samples=1)
        out_ids.append(int(next_id))
        idx = torch.cat([idx, next_id], dim=1)
    return decode_ids(torch.tensor(out_ids))

# -----------------------
# Train both and demo
# -----------------------
if __name__ == "__main__":
    print(f"Device: {device} | Vocab: {V} | Tokens: {len(TOKENS):,}")

    # --- One-hot baseline ---
    print("\n=== Training One-Hot word model on Shakespeare ===")
    onehot_model = OneHotGRULM(V, cfg.hidden_size, cfg.num_layers)
    onehot_model = train_model(onehot_model, steps=cfg.steps)

    # --- Embedding model (with upgrades) ---
    print("\n=== Building Embedding word model on Shakespeare ===")
    emb_model = EmbeddingGRULM(V, cfg.embed_dim, cfg.hidden_size, cfg.num_layers, tie_weights=cfg.tie_weights)

    # Optional: initialize from GloVe
    if cfg.use_glove:
        try:
            glove_mat = load_glove_embeddings(cfg.glove_dim, stoi)
            if glove_mat.shape[1] != cfg.embed_dim:
                # make embed_dim match glove
                print(f"Adjusting embed_dim from {cfg.embed_dim} -> {glove_mat.shape[1]} to match GloVe")
                emb_model = EmbeddingGRULM(V, glove_mat.shape[1], cfg.hidden_size, cfg.num_layers, tie_weights=cfg.tie_weights)
            with torch.no_grad():
                emb_model.embed.weight[:,:] = glove_mat
            print("Initialized embedding layer from GloVe.")
        except Exception as e:
            print(f"GloVe initialization failed ({e}); falling back to random init.")

    emb_model = train_model(
        emb_model,
        steps=cfg.steps,
        freeze_embed_steps=cfg.glove_freeze_steps if cfg.use_glove else 0,
        finetune_embeddings=cfg.glove_finetune
    )

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