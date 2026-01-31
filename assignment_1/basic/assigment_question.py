# this file is to answer question from the assignment 1 pdf file
import .train_bpe 
from .pretokenization import count_pretokens_parallel
import os 
import cProfile 
import pstats
"""train bpe tokenizer on the TinyStories dataset using vocab size of 10 000"""

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = os.cpu_count() - 1
    base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = count_pretokens_parallel(
        data_path=input_path,
        num_processes=num_processes,
        base_pattern=base_pattern,
        special_tokens =special_tokens,
        split_special_token = special_tokens[0]
    )
    vocab, merges = train_bpe.train_bpe_heap(counts=counts,special_tokens=special_tokens,vocab_size=vocab_size,**kwargs)
    # print(merges[620:640])
    return vocab, merges

def main():
    return 

if __name__== "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run_train_bpe()