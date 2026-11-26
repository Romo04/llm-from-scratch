# Implementing a simple byte pair encoding algorithm from scratch (for educational purposes)
# %%
text = "This is some text"
byte_arr = bytearray(text, "utf-8")
print(byte_arr)
# %%
ids = list(byte_arr)
print(ids)
# %%
print("Number of characters:", len(text))
print("Number of token IDS:", len(ids))
# %%
import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")
gpt2_tokenizer.encode("This is some text")
# %%
for i in range(300):
    decoded = gpt2_tokenizer.decode([i])
    print(f"{i}: {decoded}")
# %%
from collections import Counter, deque
from functools import lru_cache

class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id to token_str
        self.vocab = {}
        # token_str to token_id
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        # Preprocess: Replace spaces with 'Ġ'
        # Ġ is a particularity of the GPT-2 BPE implementation
        # E.g "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # GPT-4 BPE would tokenize it with " world" instead
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initializing vocab with unique characters, including Ġ if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]

        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)

        # Optionally ensure Ġ is included
        if 'Ġ' not in unique_chars:
            unique_chars.append('Ġ')

        self.vocab = {i: char for i,char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i,char in self.vocab.items()}

        # Adding allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly merging and pairing
        for new_id in range(len(self.vocab), vocab_size):
            if (len(token_ids) < 2):
                break
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break

            updated = self.replace_pair(token_ids, pair_id, new_id)
            if updated == token_ids:
                break

            token_ids = updated
            self.bpe_merges[pair_id] = new_id

        # Build the vocab with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text):
        """
        Encode the input text into a list of token IDs.

        Args:
            text (str): The text to encode.
        Returns:
            List[int]: The list of token IDs.
        """
        tokens = []
        # Split text into tokens, keeping newlines intact
        words = text.replace("\n", " \n ").split()

        for i, word in enumerate(words):
            if i > 0 and not word.startswith("\n"):
                tokens.append("Ġ" + word)
            else:
                tokens.append(word)

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                token_id = self.inverse_vocab[token]
                token_ids.append(token_id)
            else:
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)

        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str)L The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)

                    print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                    i += 2
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.
        Returns:
            str: The decoded string
        """
        decoded_string = ""

        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token

        return decoded_string

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        if (len(token_ids) < 2):
            return None
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)
        return replaced

    
# %%
tok = BPETokenizerSimple()

assert tok.find_freq_pair([]) is None
assert tok.find_freq_pair([42]) is None

tok.train("", vocab_size=270)
tok.train("H", vocab_size=270)
tok.train("He", vocab_size=270)

print("Edge-case checks passed.")
# %%
with open("../data/the-verdict.txt", "r") as f:
    text = f.read()
# %%
tokenizer = BPETokenizerSimple()
tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})
# %%
print(tokenizer.vocab)
print(len(tokenizer.vocab))
#%%
print(len(tokenizer.bpe_merges))
# %%
input_text = "Jack embraced beauty through art and life."
token_ids = tokenizer.encode(input_text)
print(token_ids)
# %%
print("Number of characters:", len(input_text))
print("Number of token IDs:", len(token_ids))
# %%
print(tokenizer.decode(token_ids))
# %%
for token_id in token_ids:
    print(f"{token_id} -> {tokenizer.decode([token_id])}")
# %%
tokenizer.decode(tokenizer.encode("This is some text."))
# %%
