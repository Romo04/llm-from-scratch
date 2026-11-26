#%%
import torch

print(torch.__version__)
# %%

# 3 training examples, which may represent token IDs in an LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix can be determined by obtaining the largest token ID + 1
# If the highest token ID is 3, then we want 4 rows for token IDs 0, 1, 2, 3
num_idx = max(idx) + 1

# The desired embedding dimension is a hyperparameter
out_dim = 5
# %%
# ------------------
# USING nn.Embedding
# ------------------

# Random seed for reproducibility
# Weights in the embedding layer are init with small random values
torch.manual_seed(123)

embedding = torch.nn.Embedding(num_idx, out_dim)
# %%
embedding.weight
# %%
embedding(torch.tensor([1]))
# %%
embedding(idx)
# %%
# ---------------
# USING nn.Linear
# ---------------

onehot = torch.nn.functional.one_hot(idx)
onehot
# %%
# Carries out matrix multiplication XW^T
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
linear.weight
# %%
# To compare the linear layer weights to the embedding layer we have to use the same small random weight
linear.weight = torch.nn.Parameter(embedding.weight.T)
# %%
linear(onehot.float())
# %%
embedding(idx)

# The use of matrix multiplication on one-hot encodings is equivalent to the embedding layer look-up but can be inefficient if we work with large embedding matrices (too many wasteful mults by zero)
