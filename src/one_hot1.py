import torch
import torch.nn.functional as F

# Example: a tensor of class labels
labels = torch.tensor([0, 2, 1, 0], dtype=torch.long)

# One-hot encode the labels, assuming 3 classes (0, 1, 2)
one_hot_labels = F.one_hot(labels, num_classes=3)
print(one_hot_labels)