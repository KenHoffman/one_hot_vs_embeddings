# pytorch_phrase_additivity.py
# Learn word & multiword ("phrase") embeddings with SGNS and test additivity:
#   E("new_york")  ?≈  E("new") + E("york")

# Ken 2025-09-01 When I ran this program as-is, I got a ZeroDivisionError:
# Traceback (most recent call last):
#   File "/Users/kenhoffman/Dev/Python/one_hot_vs_embeddings/src/pytorch_phrase_additivity.py", line 178, in <module>
#     avg = total / (len(loader)*BATCH_SIZE)
#           ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~
#           ZeroDivisionError: float division by zero

import math
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 0) Config
# ----------------------------
EMBED_DIM = 100
WINDOW_SIZE = 2               # symmetric context window
NEGATIVE_SAMPLES = 5
MIN_WORD_COUNT = 5
MIN_BIGRAM_COUNT = 10         # only convert bigrams this frequent into single tokens
MAX_VOCAB = 50000             # safety cap
EPOCHS = 3                    # bump for better results on a large corpus
BATCH_SIZE = 1024
LR = 2e-3
SEED = 2025
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# 1) Data: bring your own corpus
# ----------------------------
# Replace 'corpus_text' with your own large text (e.g., Shakespeare, Wikipedia dump, etc.).
# The better the corpus (size + domain), the better the experiment.
corpus_text = """
In New York City there are many neighborhoods. New York is often called NYC.
San Francisco is also a city. New Yorkers love their city; New York pizza is famous.
The city of New York has many parks. San Francisco Bay Area and New York area
are different places. The phrase "New York" appears very frequently in this toy text.
"""

# Basic tokenization (feel free to swap in spaCy or your own)
def tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.replace("\n", " ").split()]

tokens = tokenize(corpus_text)

# ----------------------------
# 2) Detect frequent bigrams and re-tokenize to include them
# ----------------------------
def find_frequent_bigrams(tokens: List[str], min_count: int) -> Counter:
    bigrams = Counter(zip(tokens, tokens[1:]))
    # Keep only those with alphabetical-ish chars to avoid junk, optional
    cleaned = Counter({(a, b): c for (a, b), c in bigrams.items()
                       if c >= min_count and a.isalpha() and b.isalpha()})
    return cleaned

bigram_counts = find_frequent_bigrams(tokens, MIN_BIGRAM_COUNT)

# Greedy merge pass: convert frequent (w_i, w_{i+1}) to 'w_i_w_{i+1}'
def merge_bigrams(tokens: List[str], bigram_counts: Counter) -> List[str]:
    i, merged = 0, []
    frequent_pairs = set(bigram_counts.keys())
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) in frequent_pairs:
            merged.append(tokens[i] + "_" + tokens[i+1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged

tokens_merged = merge_bigrams(tokens, bigram_counts)

# ----------------------------
# 3) Build vocab with subsampling-style frequency filter
# ----------------------------
word_counts = Counter(tokens_merged)
vocab = [w for w, c in word_counts.items() if c >= MIN_WORD_COUNT]
vocab = sorted(vocab, key=lambda w: (-word_counts[w], w))[:MAX_VOCAB]

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

train_tokens = [w for w in tokens_merged if w in stoi]

print(f"Tokens: {len(tokens)} -> after bigram merge: {len(tokens_merged)}")
print(f"Vocab size (≥{MIN_WORD_COUNT}): {len(vocab)}")
if any("_" in w for w in vocab):
    print("Detected multi-word tokens:", [w for w in vocab if "_" in w][:20])

# ----------------------------
# 4) SGNS dataset
# ----------------------------
def make_skipgram_pairs(tokens: List[str], window: int) -> List[Tuple[int, int]]:
    idxs = [stoi[w] for w in tokens if w in stoi]
    pairs = []
    for i, center in enumerate(idxs):
        left = max(0, i - window)
        right = min(len(idxs), i + window + 1)
        for j in range(left, right):
            if j == i:
                continue
            pairs.append((center, idxs[j]))
    return pairs

pairs = make_skipgram_pairs(train_tokens, WINDOW_SIZE)
print(f"Skip-gram training pairs: {len(pairs)}")

# Negative sampling distribution (unigram^0.75)
freqs = torch.tensor([word_counts[itos[i]] for i in range(len(vocab))], dtype=torch.float)
unigram_dist = (freqs ** 0.75) / (freqs ** 0.75).sum()
unigram_alias = torch.distributions.categorical.Categorical(probs=unigram_dist)

class SGNSDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        c, ctx = self.pairs[i]
        negs = unigram_alias.sample((NEGATIVE_SAMPLES,))
        return torch.tensor(c), torch.tensor(ctx), negs

loader = DataLoader(SGNSDataset(pairs), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ----------------------------
# 5) Model: SGNS (target and context embeddings)
# ----------------------------
class SGNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.target = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.target.weight, -0.5/dim, 0.5/dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, center_ids, pos_ctx_ids, neg_ctx_ids):
        # center: (B,), pos_ctx: (B,), neg_ctx: (B, K)
        v = self.target(center_ids)               # (B, D)
        u_pos = self.context(pos_ctx_ids)         # (B, D)
        u_neg = self.context(neg_ctx_ids)         # (B, K, D)

        # Positive term: log sigma(v·u_pos)
        pos_score = torch.sum(v * u_pos, dim=1)       # (B,)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-9)

        # Negative term: sum log sigma(-v·u_neg)
        neg_score = torch.einsum("bd,bkd->bk", v, u_neg)  # (B, K)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-9), dim=1)

        return (pos_loss + neg_loss).mean()

    def get_word_vectors(self):
        # Many word2vec implementations use the sum or the target embeddings; we’ll use target.
        return self.target.weight.detach()

model = SGNS(len(vocab), EMBED_DIM).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ----------------------------
# 6) Train
# ----------------------------
model.train()
for epoch in range(1, EPOCHS+1):
    total = 0.0
    for centers, pos, neg in loader:
        centers = centers.to(DEVICE)
        pos = pos.to(DEVICE)
        neg = neg.to(DEVICE)
        loss = model(centers, pos, neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Ken 2025-09-01 added the following print statement for debugging.
        # It never prints anything.
        print(f"epoch={epoch}, loss.item()={loss.item()}, centers.size(0)={centers.size(0)}")
        total += loss.item() * centers.size(0)
    # avg = total / (len(loader)*BATCH_SIZE)
    # Ken 2025-09-01 replaced the previous line with the following.
    print(f"epoch={epoch}, total={total}, len(loader)={len(loader)}, BATCH_SIZE={BATCH_SIZE}")
    # Output:
    # epoch=1, total=0.0, len(loader)=0, BATCH_SIZE=1024
    # epoch=2, total=0.0, len(loader)=0, BATCH_SIZE=1024
    # epoch=3, total=0.0, len(loader)=0, BATCH_SIZE=1024

    # len(loader) is always 0, so the following code never prints.
    # if len(loader) * BATCH_SIZE != 0:
    #     avg = total / (len(loader)*BATCH_SIZE)
    #     print(f"Epoch {epoch}/{EPOCHS} - loss {avg:.4f}")

# ----------------------------
# 7) Evaluate additivity: E("w1_w2") vs E("w1")+E("w2")
# ----------------------------
model.eval()
W = model.get_word_vectors()  # (V, D)
W = nn.functional.normalize(W, dim=1)  # normalize for cosine comparisons

def vec(tok: str) -> torch.Tensor:
    return W[stoi[tok]] if tok in stoi else None

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a*b).sum().item())

# Collect phrase tokens present in vocab
phrase_tokens = [w for w in vocab if "_" in w]

results = []
for phrase in phrase_tokens:
    parts = phrase.split("_")
    if len(parts) != 2:
        continue  # this script merges bigrams only
    if parts[0] in stoi and parts[1] in stoi:
        v_phrase = vec(phrase)
        v_sum = nn.functional.normalize(vec(parts[0]) + vec(parts[1]), dim=0)
        cos = cosine(v_phrase, v_sum)
        results.append((phrase, parts[0], parts[1], cos))

results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
print("\nPhrase vs Sum(components) — top few by cosine similarity:")
for r in results_sorted[:10]:
    print(f"{r[0]:<20s} ~ {r[1]} + {r[2]} : cos={r[3]:.3f}")

if results:
    cosines = torch.tensor([r[3] for r in results])
    print(f"\n#phrases={len(results)}  mean cos={cosines.mean():.3f}  median cos={cosines.median():.3f}")
else:
    print("\nNo multi-word tokens in vocab (increase MIN_BIGRAM_COUNT or use larger corpus).")

# ----------------------------
# 8) Extra: test on arbitrary strings (no learned single token)
#     Compare 'sum-of-words' vs 'sum-of-words' including stopwords removal.
# ----------------------------
def embed_string_bow(text: str, drop_stops=True) -> torch.Tensor:
    stops = {"the", "of", "and", "a", "an", "to", "in", "on", "is", "are"}
    toks = tokenize(text)
    toks = [t for t in toks if t in stoi and (not drop_stops or t not in stops)]
    if not toks:
        return None
    s = torch.zeros(EMBED_DIM, device=W.device)
    for t in toks:
        s += model.get_word_vectors()[stoi[t]]
    return nn.functional.normalize(s, dim=0)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

example_pairs = [
    ("new york", "new_york"),
    ("san francisco", "san_francisco"),
    ("new york city", "new_york"),
]

print("\nString-to-token sanity checks (when token exists):")
for s, tok in example_pairs:
    v_s = embed_string_bow(s)
    if v_s is None or tok not in stoi:
        continue
    v_tok = W[stoi[tok]]
    print(f"'{s}' vs '{tok}': cos={cosine_sim(v_s, v_tok):.3f}")

print("\nDone.")