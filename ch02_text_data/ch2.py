# %%
from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# %%
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_folder, "..", "data", "the-verdict.txt")

if (os.path.exists(file_path)):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Length of text: ", len(raw_text))
    print(raw_text[:99])
else:
    print(f"File not found at {file_path}, check the location.")

# %%
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)

# %%
result = re.split(r'([,.]|\s)', text)
print(result)

# %%
result = [item for item in result if item.strip()]
print(result)

# %%
text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# %%
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

# %%
print(len(preprocessed))

# %%
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

# %%
vocab = {token:integer for integer, token in enumerate(all_words)}

# %%
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# %%
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# %%
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# %%
tokenizer.decode(ids)
# %%
tokenizer.decode(tokenizer.encode(text))
# %%
