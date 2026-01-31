from .train_bpe import train_bpe
from dataclasses import dataclass
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int,bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str] | None  = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens


    def from_files(cls, vocab_filepath:str, merges_filepath: str, special_tokens=None):
        """Construct and return a tokenizer from a serialized vocab and list of merges"""
        return
    
    def encode(self, text:str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""

        return
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings, return a generator that lazily yields token IDs"""
        return
    

    def decode(self, ids: list[int]) -> str:
        """decode a sequence of tokens IDS into text"""

        return
    


def main():
    return
if __name__ == "__main__":
"""
to test it run pytest -m ttests/test_tokenizer.py

"""