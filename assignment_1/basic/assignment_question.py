# this file is to answer question from the assignment 1 pdf file
from __future__ import annotations

from .train_bpe import train_bpe_heap
from .pretokenization import count_pretokens_parallel
from .bytes_utils import bytes_to_unicode_text

import json 
from pathlib import Path
import os 
import cProfile 
import pstats
import argparse
import time 
import tracemalloc

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
    print("-"*25,"Pretokenization", "-"*25)

    pretoken_start = time.perf_counter() # time 
    tracemalloc.start() # memory taken
    counts = count_pretokens_parallel(
        data_path=input_path,
        num_processes=num_processes,
        base_pattern=base_pattern,
        special_tokens =special_tokens,
        split_special_token = special_tokens[0]
    )
    current,peak = tracemalloc.get_traced_memory() # return bytes to convert in GB -> N bytes * 1024 (KiB) * 1024 (MiB) * 1024 (GiB)
    tracemalloc.stop()
    pretoken_end = time.perf_counter()

    elapsed_s = pretoken_end-pretoken_start
    print(f"Pretokenizer took {elapsed_s/60} min")
    print(f"Memory taken currently {current/(1024**3)} GB with a pick at {peak/1024**3} GB") 

    
    print("-"*25,"BPE Algorithm", "-"*25)
    bpe_start = time.perf_counter()
    tracemalloc.start() # memory taken
    vocab, merges = train_bpe_heap(counts=counts,special_tokens=special_tokens,vocab_size=vocab_size,**kwargs)
    bpe_end = time.perf_counter()
    current,peak = tracemalloc.get_traced_memory() # return bytes to convert in GB -> N bytes * 1024 (KiB) * 1024 (MiB) * 1024 (GiB)
    tracemalloc.stop()
    elapsed_bpe_s = bpe_end - bpe_start
    print(f"BPE took {elapsed_bpe_s/60} min")
    print(f"Memory taken currently {current/(1024**3)} GB with a pick at {peak/1024**3} GB") 
    
    return vocab, merges


def save_vocab_merges(vocab, merges, vocab_path, merges_path):
    """return a json file of the vocab with mapping {token_str: token_id}"""
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    os.makedirs(os.path.dirname(merges_path), exist_ok=True)
    vocab_json = {bytes_to_unicode_text(token_bytes):token_id for token_id, token_bytes in sorted(vocab.items())}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False) # not escaping non-ASCII characters
        
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{bytes_to_unicode_text(a)} {bytes_to_unicode_text(b)}\n")
        

def train_bpe(data_folder_path, vocab_size, special_tokens, dataset):
    """return vocab and merges for a specific dataset"""
    data_path = os.path.join(data_folder_path, dataset)

    print("*"*15,f"Start training tokenizer on {dataset} dataset", "*"*15)   
    pr_ts = cProfile.Profile()
    pr_ts.enable()
    vocab, merges = run_train_bpe(data_path,vocab_size,special_tokens)
    pr_ts.disable()

    # longest token
    print("longest token in vocab:", max(vocab.values(),key= len))


    print("*"*25,f"result on {dataset} datset", "*"*25)
    result_ts = pstats.Stats(pr_ts)
    result_ts.sort_stats(pstats.SortKey.TIME)
    result_ts.print_stats(10)

    print("-"*25,"Profile Analysis", "-"*25)
    return vocab, merges


def main():
    special_tokens= ["<|endoftext|>"]    
    saving_path = "artifacts"
    HERE = os.path.dirname(os.path.abspath(__file__)) # .../Stanford/assignment_1/basic
    DATA_FOLDER = "data"
    ARTIFACTS_FOLDER = "artifacts"
    data_folder_path = os.path.join(HERE, "..", DATA_FOLDER)
    artifact_folder_path = os.path.join(HERE, ARTIFACTS_FOLDER)
    vocab_path_ts = os.path.join(artifact_folder_path,"vocab_10k.json")
    merge_path_ts =  os.path.join(artifact_folder_path,"merges_10k.txt")
    vocab_path_owt = os.path.join(artifact_folder_path,"vocab_32k.json")
    merge_path_owt =  os.path.join(artifact_folder_path,"merges_32k.txt")
    dataset = ["tinystories_train.txt","openwebtext_train.txt"]
    
    print("="*30,"Start training tokenizer", "="*30)   
    vocab_ts, merge_ts   =  train_bpe(data_folder_path,vocab_size=10000,special_tokens=special_tokens,dataset=dataset[0])
    vocab_owt, merge_owt =  train_bpe(data_folder_path,vocab_size=32000,special_tokens=special_tokens,dataset=dataset[1])
    
    save_vocab_merges(vocab_ts,merge_ts,vocab_path_ts,merge_path_ts)
    save_vocab_merges(vocab_owt,merge_owt,vocab_path_owt,merge_path_owt)

if __name__== "__main__":
    main()

"""
============================== Start training tokenizer ==============================
*************** Start training tokenizer on tinystories_train.txt dataset ***************
------------------------- Pretokenization -------------------------
Pretokenizer took 0.22992387361591682 min
Memory taken currently 0.005991010926663876 GB with a pick at 0.04069757927209139 GB
------------------------- BPE Algorithm -------------------------
Base vocabulary size: 257
final vocab size: 10000
Total merges performed: 9743
BPE took 0.36352416874918464 min
Memory taken currently 0.0015850821509957314 GB with a pick at 0.13660242035984993 GB
longest token in vocab: b' accomplishment'
************************* result on tinystories_train.txt datset *************************
         24145167 function calls (24144962 primitive calls) in 35.612 seconds

   Ordered by: internal time
   List reduced from 563 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       19   13.605    0.716   13.605    0.716 {method 'acquire' of '_thread.lock' objects}
   886082    9.917    0.000   10.018    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:135(sift_down)
   277780    4.900    0.000    7.641    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:170(_update_pair_stats_for_word_heap)
  1710143    1.979    0.000    2.213    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:116(push)
     9743    1.653    0.000   12.467    0.001 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:147(_select_best_pair_heap)
   277780    0.564    0.000    0.723    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
   855236    0.494    0.000   10.483    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:127(pop_top)
  6458310    0.457    0.000    0.457    0.000 {built-in method builtins.len}
     9743    0.432    0.000    8.799    0.001 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:207(_apply_merge_to_sequences_heap)
  4255725    0.270    0.000    0.270    0.000 {method 'get' of 'dict' objects}


------------------------- Profile Analysis -------------------------
Updating the heap and pair_counts is taking most of the time in the tokenizer process
*************** Start training tokenizer on openwebtext_train.txt dataset ***************
------------------------- Pretokenization -------------------------
Pretokenizer took 2.012904306947409 min
Memory taken currently 0.578836752101779 GB with a pick at 2.415622290223837 GB
------------------------- BPE Algorithm -------------------------
Base vocabulary size: 257
final vocab size: 32000
Total merges performed: 31743
BPE took 101.58056926944991 min
Memory taken currently 0.00523912999778986 GB with a pick at 20.344736545346677 GB
longest token in vocab: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
************************* result on openwebtext_train.txt datset *************************
         4458741759 function calls in 6216.156 seconds

   Ordered by: internal time
   List reduced from 249 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
153860798 3299.661    0.000 3318.758    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:135(sift_down)
 36031452 1121.832    0.000 1805.308    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:170(_update_pair_stats_for_word_heap)
317138737  498.141    0.000  541.918    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:116(push)
    31743  468.523    0.015 3922.755    0.124 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:147(_select_best_pair_heap)
 36031452  114.475    0.000  146.357    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
       19  107.146    5.639  107.146    5.639 {method 'acquire' of '_thread.lock' objects}
153772412  102.112    0.000 3424.560    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:127(pop_top)
1194881312   84.389    0.000   84.389    0.000 {built-in method builtins.len}
    31743   78.804    0.002 2030.495    0.064 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:207(_apply_merge_to_sequences_heap)
        1   74.317   74.317 6216.156 6216.156 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py:19(run_train_bpe)


------------------------- Profile Analysis -------------------------
Updating the heap and pair_counts is taking most of the time in the tokenizer process
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py", line 140, in <module>
    
  File "/Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py", line 136, in main
    save_vocab_merges(vocab_owt,merge_owt,vocab_path_owt,merge_path_owt)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py", line 86, in save_vocab_merges
    with open(vocab_path, "w", encoding="utf-8") as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/../artifacts/vocab_10k.json'
"""