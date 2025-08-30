import torch
import torch.nn as nn

# Define the vocabulary size (number of unique words)
vocab_size = 1000  # Example: 1000 unique words in your vocabulary

# Define the embedding dimension (size of each embedding vector)
embedding_dim = 128  # Example: Each word will be represented by a 128-dimensional vector

# Create the embedding layer
# nn.Embedding(num_embeddings, embedding_dim)
# num_embeddings: The size of the dictionary of embeddings (vocab_size)
# embedding_dim: The size of each embedding vector
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Example input: a tensor of word indices
# Let's say we have a sentence represented by these word indices
# For example, if word_to_index['hello'] = 5, word_to_index['world'] = 10, etc.
input_indices = torch.tensor([5, 10, 2, 7, 1])

# Pass the input indices through the embedding layer
# The embedding layer will look up the corresponding embedding vector for each index
embedded_output = embedding_layer(input_indices)

# Print the shape of the output tensor
# The output shape will be (sequence_length, embedding_dim)
print(f"Shape of embedded output: {embedded_output.shape}")

# Print the embedded vectors (optional, for demonstration)
print("Embedded vectors:")
print(embedded_output)

# You can also access and modify the embedding weights directly
# For example, to load pre-trained embeddings:
# pre_trained_weights = torch.randn(vocab_size, embedding_dim) # Replace with your actual pre-trained weights
# embedding_layer.weight.data.copy_(pre_trained_weights)