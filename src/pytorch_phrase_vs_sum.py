# pytorch_phrase_vs_sum.py
import math, random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities / configuration
# -------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

seed = 42
random.seed(seed)
torch.manual_seed(seed)

device = pick_device()
print("Device:", device)

# Small, fast demo config (bump sizes/epochs for more signal)
VOCAB_SIZE = 400        # number of distinct words
EMB_DIM    = 48         # dimensionality of embeddings
N_PHRASES  = 1500       # number of distinct multi-word strings
MIN_LEN, MAX_LEN = 2, 6 # range of lengths for strings
BATCH_SIZE = 256
EPOCHS     = 60
LR         = 0.3
NOISE_STD  = 0.01       # label noise

# ---------------------------------------
# Synthetic “corpus” of multi-word strings
# ---------------------------------------
def make_phrases(n_phrases:int, vocab_size:int) -> Tuple[List[List[int]], List[str]]:
    phrases_ids: List[List[int]] = []
    phrases_txt: List[str] = []
    for _ in range(n_phrases):
        L = random.randint(MIN_LEN, MAX_LEN)
        ids = [random.randrange(vocab_size) for _ in range(L)]
        phrases_ids.append(ids)
        phrases_txt.append(" ".join(f"w{j}" for j in ids))
    return phrases_ids, phrases_txt

phrases_ids, phrases_txt = make_phrases(N_PHRASES, VOCAB_SIZE)
max_len = max(len(p) for p in phrases_ids)

# pack to tensors (padded with -1)
pad_id = -1
token_mat = torch.full((N_PHRASES, max_len), pad_id, dtype=torch.long)
lengths   = torch.zeros(N_PHRASES, dtype=torch.long)
for i, ids in enumerate(phrases_ids):
    token_mat[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    lengths[i] = len(ids)

phrase_ids = torch.arange(N_PHRASES, dtype=torch.long)

# ------------------------------------------
# Generate “ground truth” target embeddings
# ------------------------------------------
# We’ll pretend each phrase has a hidden ground-truth vector y.
# In the additive regime, y = sum(true_word_vecs).
# In the non-additive regime, y = sum(T_pos[pos] @ true_word_vecs[word])
# (i.e., position-dependent transforms => not representable as a plain sum)
true_word_table = torch.randn(VOCAB_SIZE, EMB_DIM)

pos_mats = torch.stack([torch.randn(EMB_DIM, EMB_DIM) * 0.15 + torch.eye(EMB_DIM)  # near-identity transforms
                        for _ in range(max_len)], dim=0)

def build_targets(additive: bool) -> torch.Tensor:
    ys = []
    for i in range(N_PHRASES):
        ids = token_mat[i]
        ids = ids[ids != pad_id]
        if additive:
            y = true_word_table[ids].sum(dim=0)
        else:
            comps = []
            for pos, wid in enumerate(ids):
                comps.append(pos_mats[pos] @ true_word_table[wid])
            y = torch.stack(comps, dim=0).sum(dim=0)
        y = y + NOISE_STD * torch.randn_like(y)
        ys.append(y)
    return torch.stack(ys, dim=0)

targets_add     = build_targets(additive=True)
targets_nonadd  = build_targets(additive=False)

# ------------------------------------------
# Train two learnable embedding tables:
#   1) word_emb:   word -> vector
#   2) phrase_emb: phrase_id -> vector
#
# Each is trained to regress to the same target y for the phrase.
# After training we will compare:
#    phrase_emb(phrase)  vs  sum_j word_emb(word_j)
# If the world is additive, these should closely match.
# ------------------------------------------
@dataclass
class TrainedEmbeddings:
    word_emb: nn.Embedding
    phrase_emb: nn.Embedding

def train_to_targets(targets: torch.Tensor) -> TrainedEmbeddings:
    word_emb   = nn.Embedding(VOCAB_SIZE, EMB_DIM).to(device)
    phrase_emb = nn.Embedding(N_PHRASES, EMB_DIM).to(device)

    opt = torch.optim.SGD(list(word_emb.parameters()) + list(phrase_emb.parameters()), lr=LR)

    # Precompute tensors on device
    token_mat_dev  = token_mat.to(device)
    lengths_dev    = lengths.to(device)
    phrase_ids_dev = phrase_ids.to(device)
    targets_dev    = targets.to(device)

    n_batches = math.ceil(N_PHRASES / BATCH_SIZE)

    for epoch in range(EPOCHS):
        perm = torch.randperm(N_PHRASES)
        total = 0.0
        for bi in range(n_batches):
            batch_idx = perm[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
            toks  = token_mat_dev[batch_idx]      # (B, L)
            lens  = lengths_dev[batch_idx]        # (B,)
            pids  = phrase_ids_dev[batch_idx]     # (B,)
            y     = targets_dev[batch_idx]        # (B, D)

            # sum of word embeddings with masking
            mask = (toks != pad_id).float().unsqueeze(-1)  # (B, L, 1)
            word_vecs = word_emb(toks.clamp_min(0))        # (B, L, D)  (pad rows ignored via mask)
            word_sum  = (word_vecs * mask).sum(dim=1)      # (B, D)

            # direct phrase embedding
            pvec = phrase_emb(pids)                        # (B, D)

            # regress BOTH to the same target
            loss = F.mse_loss(word_sum, y) + F.mse_loss(pvec, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d} | loss {total/n_batches:.4f}")

    return TrainedEmbeddings(word_emb=word_emb, phrase_emb=phrase_emb)

def evaluate(tr: TrainedEmbeddings, label: str, targets: torch.Tensor):
    with torch.no_grad():
        toks  = token_mat.to(device)
        mask  = (toks != pad_id).float().unsqueeze(-1)
        word_sum = (tr.word_emb(toks.clamp_min(0)) * mask).sum(dim=1)  # (N, D)
        pvec     = tr.phrase_emb(phrase_ids.to(device))                # (N, D)

        # normalize for cosine
        def cos_sim(a, b):
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
            return (a * b).sum(dim=-1)

        cos = cos_sim(word_sum, pvec).detach().cpu()
        mse = F.mse_loss(word_sum, pvec).item()
        mae = (word_sum - pvec).abs().mean().item()

        # also report how well each matches the true target
        y = targets.to(device)
        cos_w_y = cos_sim(word_sum, y).mean().item()
        cos_p_y = cos_sim(pvec, y).mean().item()

        print(f"\n[{label}] Agreement between phrase_emb and SUM(word_emb):")
        print(f"  mean cosine(sim)  : {cos.mean().item():.4f}")
        print(f"  median cosine     : {cos.median().item():.4f}")
        print(f"  MSE difference    : {mse:.4f}")
        print(f"  MAE difference    : {mae:.4f}")
        print(f"  cos( SUM(words), target y ) avg : {cos_w_y:.4f}")
        print(f"  cos( phrase_vec,  target y ) avg : {cos_p_y:.4f}")

        # show 5 example phrases with low/high agreement
        k = 5
        idx_low  = torch.topk(cos, k, largest=False).indices.tolist()
        idx_high = torch.topk(cos, k, largest=True ).indices.tolist()

        def show(idx_list, title):
            print(f"\n  {title}")
            for i in idx_list:
                print(f"    cos={cos[i].item():.4f} | '{phrases_txt[i]}'")
        show(idx_low,  f"Bottom-{k} examples")
        show(idx_high, f"Top-{k} examples")

print("\n=== Training in ADDITIVE world (hypothesis should hold) ===")
tr_add = train_to_targets(targets_add)
evaluate(tr_add, "ADDITIVE", targets_add)

print("\n=== Training in NON-ADDITIVE world (word order/interactions) ===")
tr_non = train_to_targets(targets_nonadd)
evaluate(tr_non, "NON-ADDITIVE", targets_nonadd)

# ------------------------------------------
# How to USE the trained tables:
# ------------------------------------------
# - Get embedding for a word id: tr_add.word_emb(torch.tensor([wid]))
# - Get embedding for a string by SUM of word embeddings:
#       ids = [...]
#       vec = tr_add.word_emb(torch.tensor(ids, device=device)).sum(dim=0, keepdim=True)
# - Get direct embedding for a known phrase id:
#       pid = ...
#       vec = tr_add.phrase_emb(torch.tensor([pid], device=device))