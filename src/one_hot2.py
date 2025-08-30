import torch
import torch.nn.functional as F

torch.set_printoptions(threshold=10000, linewidth=200)

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

def decode(ids: torch.Tensor):
    return "".join(itos[int(i)] for i in ids)

# Data as indices
data = encode(TEXT)
print(f"{data=}")

one_hot_labels = F.one_hot(data, num_classes=len_vocab)
print(f"{one_hot_labels=}")