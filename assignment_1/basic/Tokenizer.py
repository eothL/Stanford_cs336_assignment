from .train_bpe import _merge_pair_in_seq
from .pretokenizer import compile_pattern
from .bytes_utils import gpt2_unicode_to_bytes, bytes_to_unicode_text, unicode_token_to_bytes, unicode_counts_to_bytes
from typing import Iterable, Iterator
import json
import os 


class Tokenizer:
    def __init__(self, vocab: dict[int,bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str] | None  = None):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {bytes_token : token_id for token_id, bytes_token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.split_tokens = ["<|endoftext|>"]
        self.special_tokens = {t for t in special_tokens if t!= self.split_tokens} if special_tokens else self.split_tokens
        self.base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._pat = compile_pattern(self.base_pattern, self.special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath: str, special_tokens=None):
        """Construct and return a tokenizer from a serialized vocab and list of merges"""
        with open(vocab_filepath) as f:
            gpt2_vocab = json.load(f)

        vocab = {token_id: unicode_token_to_bytes(token_str) for token_str, token_id in gpt2_vocab.items()}
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                a,b = line.rstrip().split(" ")
                merges.append((unicode_token_to_bytes(a),unicode_token_to_bytes(b)))

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text:str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        cache: dict[bytes, list[int]] = {} # prevent from recomputing already seen pretoken
        seq_id = [] 
        for m in self._pat.finditer(text):
            pretoken = m.group(0)
            pretoken_bytes = pretoken.encode("utf-8")
            if pretoken_bytes in cache:
                seq_id.extend(cache[pretoken_bytes])
                continue

            if pretoken in self.special_tokens:
                ids = [self.token_to_id[pretoken_bytes]]
                cache[pretoken_bytes] = ids
                seq_id.extend(ids)
                continue

            seq = [bytes([b]) for b in pretoken_bytes]
            while True:
                pairs = list(zip(seq,seq[1:]))
                ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]
                if not ranked:
                    break    
                _, best_pair = min(ranked)  
                seq = _merge_pair_in_seq(seq,best_pair[0], best_pair[1], best_pair[0] + best_pair[1])
                        
                        
            ids = [self.token_to_id[s] for s in seq]
            cache[pretoken_bytes] = ids 
            seq_id.extend(ids)
        return seq_id
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings, return a generator that lazily yields token IDs"""
        for text in iterable:
            yield from self.encode(text)
    

    def decode(self, ids: list[int]) -> str:
        """decode a sequence of tokens IDS into text"""
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("utf-8")
    

def test():
    HERE = os.path.dirname(os.path.abspath(__file__)) # ../Stanford/assignment_1/basic
    vocab_path = os.path.join(HERE, "artifacts/vocab_32k.json")
    merge_path = os.path.join(HERE, "artifacts/merges_32k.txt")
      
    tok = Tokenizer.from_files(vocab_path, merge_path)
    print(tok.encode("hello world"))
    print(tok.decode([31469, 920]))

def main():
    HERE = os.path.dirname(os.path.abspath(__file__)) # ../Stanford/assignment_1/basic 
    vocab_path = os.path.join(HERE, "artifacts/vocab_32k.json")
    merge_path = os.path.join(HERE, "artifacts/merges_32k.txt")
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath= merge_path)
    

    return
if __name__ == "__main__":
    test() 
"""
to test it run pytest -m ttests/test_tokenizer.py
"""