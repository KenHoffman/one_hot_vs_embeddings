import torch
import torch.nn as nn

torch.set_printoptions(threshold=10000, linewidth=200)

vocab_size = 1000
embedding_dim = 128

TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them.\n"
)

# Build char vocab
vocab = sorted(list(set(TEXT)))
print(f"{vocab=}")
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for ch, i in stoi.items()}
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