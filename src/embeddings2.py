import torch
import torch.nn as nn

torch.set_printoptions(threshold=10000, linewidth=200)

vocab_size = 10
embedding_dim = 8

TEXT = (
    "To be or not to be"
)

# Build char vocab
text_set = set(TEXT)
print(f"{text_set=}")
vocab = sorted(list(text_set))
print(f"{vocab=}")
stoi = {ch:i for i, ch in enumerate(vocab)}
print(f"{stoi=}")
itos = {i:ch for ch, i in stoi.items()}
print(f"{itos=}")
len_vocab = len(vocab)
print(f"{len_vocab=}")

def encode(s: str):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

# Data as indices
data = encode(TEXT)
print(f"{data=}")
print(f"{data.shape=}")

embed = nn.Embedding(vocab_size, embedding_dim)
print(f"{embed=}")
embed_output = embed(data)
print(f"{embed_output=}")
print(f"{embed_output.shape=}")