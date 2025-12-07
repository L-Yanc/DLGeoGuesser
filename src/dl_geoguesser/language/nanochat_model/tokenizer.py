"""
BPE Tokenizer for text generation (adapted from nanochat).
Uses tiktoken for efficient inference.
"""

import os
import pickle
from functools import lru_cache


SPECIAL_TOKENS = [
    "<|bos|>",  # Beginning of Sequence
    "<|user_start|>",  # User messages
    "<|user_end|>",
    "<|assistant_start|>",  # Assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # Tool use
    "<|python_end|>",
    "<|output_start|>",  # Tool output
    "<|output_end|>",
]


class Tokenizer:
    """
    Tokenizer wrapper around tiktoken for efficient inference.
    """

    def __init__(self, enc, bos_token="<|bos|>"):
        """
        Args:
            enc: tiktoken.Encoding object
            bos_token: Special token for beginning of sequence
        """
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_file(cls, tokenizer_path):
        """
        Load tokenizer from pickle file.

        Args:
            tokenizer_path: Path to tokenizer.pkl file

        Returns:
            Tokenizer instance
        """
        with open(tokenizer_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        Load tokenizer from directory containing tokenizer.pkl.

        Args:
            tokenizer_dir: Directory containing tokenizer.pkl

        Returns:
            Tokenizer instance
        """
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        return cls.from_file(tokenizer_path)

    def get_vocab_size(self):
        """Get vocabulary size."""
        return self.enc.n_vocab

    def get_special_tokens(self):
        """Get set of special tokens."""
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """Convert token ID to token string."""
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """Encode a special token."""
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        """Get BOS token ID."""
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        Encode text to token IDs.

        Args:
            text: String or list of strings to encode
            prepend: Optional token or token ID to prepend
            append: Optional token or token ID to append
            num_threads: Number of threads for batch encoding

        Returns:
            List of token IDs (or list of lists for batch input)
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        """Allow tokenizer to be called as a function."""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self.enc.decode(ids)


def load_tokenizer(tokenizer_dir):
    """
    Load tokenizer from directory.

    Args:
        tokenizer_dir: Directory containing tokenizer.pkl

    Returns:
        Tokenizer instance
    """
    return Tokenizer.from_directory(tokenizer_dir)
