#%%
import torch

print(torch.__version__)
# %%
# 3 training examples, which may represent token IDs in an LLM context
idx = torch.tensor([2, 3, 1])