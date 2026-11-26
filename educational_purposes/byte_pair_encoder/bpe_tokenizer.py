# Python bpe tokenizer class that mimics tiktoken
from collections import Counter, deque
from functools import lru_cache
import re
import json

class BPETokenizer:
    def __init__(self):
        self.vocab = {} # {id: str}
        self.inverse_vocab = {} # {str: id}
        self.bpe_merges = {}    # {(token_id1, token_id2): merged_tid}
        
        # Official OpenaAI GPT-2 merges, use a rank dict: {(str_a, str_b): rank}
        # Lower rank = higher priority
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        tokens = self.pretokenize_text(text)

        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted({char for tok in tokens for char in tok})
            if char not in unique_chars
        )

        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i:char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char:i for i, char in self.vocab.items()}

        if allowed_special:
            for tok in allowed_special:
                if tok not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = tok
                    self.inverse_vocab[tok] = new_id

        token_id_sequences = [
            [self.inverse_vocab[char] for char in tok]
            for tok in tokens
        ]

        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_id_sequences, mode="most")
            if pair_id is None:
                break

            token_id_sequences = self.replace_pair(token_id_sequences, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """

        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file) # encoder.json: {token_str: id}
            self.vocab = {int(v):k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k:int(v) for k, v in loaded_vocab.items()}

        # Must have GPT-2's printable newline character 'Ċ' (U+010A) at id 198
        if "Ċ" not in self.inverse_vocab or self.inverse_vocab["Ċ"] != 198:
            raise KeyError("Vocabulary missing GPT-2 newline glyph 'Ċ' at id 198.")

        # Must have <|endoftext|> at 50256
        if "<|endoftext|>" not in self.inverse_vocab or self.inverse_vocab["<|endoftext|>"] != 50256:
            raise KeyError("Vocabulary missing <|endoftext|> at id 50256.")

        if "\n" not in self.inverse_vocab:
            self.inverse_vocab["\n"] = self.inverse_vocab["Ċ"]

        if "\r" not in self.inverse_vocab:
            if 201 in self.vocab:
                self.inverse_vocab["\r"] = 201
            else:
                raise KeyError("Vocabulary missing carriage return token at 201.")

        self.bpe_ranks = {}
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines [1:]

            rank = 0
            for line in lines:
                tok1, *rest = line.strip().split
                if len(rest) != 1:
                    continue

                tok2 = rest[0]

                if tok1 in self.inverse_vocab and tok2 in self.inverse_vocab:
                    self.bpe_ranks[(tok1, tok2)] = rank
                    rank += 1
        
    def encode(self, text, allowed_special=None):
        """
        Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.
    
        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough. If None, special handling is disabled.
    
        Returns:
            List of token IDs.
        """
        # --- Mimicking Tiktoken ---
        specials_in_vocab = [
            tok for tok in self.inverse_vocab
            if tok.startswith("<|") and tok.endswith("|>")
        ]

        if allowed_special is None:
            disallowed = [tok for tok in specials_in_vocab if tok in text]

            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        else:
            disallowed = [tok for tok in specials_in_vocab if tok in text and tok not in allowed_special]

            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        # ---

        token_ids = []
        
        # If some specials are allowed, split around them and passthrough those ids
        if allowed_special is not None and len(allowed_special) > 0:
            special_pattern = "(" + "|".join(
                re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)
            ) + ")"

            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                token_ids.extend*self.encode(prefix, allowed_special=None)

                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                
                last_index = match.end()

            text = text[last_index:]

            disallowed = [
                tok for tok in self.inverse_vocab
                if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special
            ]

            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")

        tokens = self.pretokenize_text(text)

        for tok in tokens:
            if tok in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[tok])
            else:
                token_ids.extend(self.tokenize_with_bpe(tok))

        return token_ids
        
    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """

        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, token_id in zip(token, token_ids) if token_id is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        if not self.bpe_ranks:
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

                        i += 2
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1

                if i < len(token_ids):
                    new_tokens.append(token_ids[i])

                token_ids = new_tokens
            
            return token_ids

        symbols = [self.vocab[id_num] for id_num in token_ids]

        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            min_rank = float("inf")
            bigram = None

            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            if bigram is None or bigram not in self.bpe_ranks:
                break

            first, second, = bigram
            new_symbols = []
            i = 0

            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols

            if len(symbols) == 1:
                break

        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        
        return merged_ids

    def decode(self, token_ids):
        return

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        return

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        return

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return

    @staticmethod
    def pretokenize_text(text):
        tokens = []

        parts = re.split(r'(\r\n|\r|\n)', text)
        for part in parts:
            if part == "":
                continue
            elif part == "\r\n":
                tokens.append("\r")
                tokens.append("\n")
            elif part == "\r":
                tokens.append("\r")
            elif part == "\n":
                tokens.append("\n")

            pending_spaces = 0
            for m in re.finditer(r'( +)|(\S+)', part):
                if m.group(1) is not None:
                    pending_spaces += len(m.group(1))
                else:
                    word = m.group(2)
                    if pending_spaces > 0:
                        for _ in range(pending_spaces - 1):
                            tokens.append("Ġ")

                        tokens.append("Ġ" + word)
                        pending_spaces = 0
                    else:
                        tokens.append(word)

            for _ in range(pending_spaces):
                tokens.append("Ġ")

        return tokens

    @staticmethod
    def find_freq_pair(token_id_sequences, mode="most"):
        return

    @staticmethod
    def replace_pair(token_id, sequences, pair_id, new_id):
        return