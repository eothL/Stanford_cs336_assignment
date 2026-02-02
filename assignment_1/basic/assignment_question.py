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
Pretokenizer took 0.25675726666813714 min
Memory taken currently 0.005991418845951557 GB with a pick at 0.04069809056818485 GB
------------------------- BPE Algorithm -------------------------
Base vocabulary size: 257
final vocab size: 10000
Total merges performed: 9743
BPE took 0.35713185555068777 min
Memory taken currently 0.0015851343050599098 GB with a pick at 0.1366022638976574 GB
longest token in vocab: b' accomplishment'
************************* result on tinystories_train.txt datset *************************
         24145149 function calls (24144944 primitive calls) in 36.839 seconds

   Ordered by: internal time
   List reduced from 563 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       19   15.213    0.801   15.213    0.801 {method 'acquire' of '_thread.lock' objects}
   886082    9.357    0.000    9.457    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:135(sift_down)
   277780    4.992    0.000    7.753    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:170(_update_pair_stats_for_word_heap)
  1710143    1.991    0.000    2.227    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:116(push)
     9743    1.679    0.000   11.946    0.001 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:147(_select_best_pair_heap)
   277780    0.568    0.000    0.725    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
   855236    0.495    0.000    9.924    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:127(pop_top)
  6458310    0.456    0.000    0.456    0.000 {built-in method builtins.len}
     9743    0.430    0.000    8.911    0.001 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:207(_apply_merge_to_sequences_heap)
  4255725    0.276    0.000    0.276    0.000 {method 'get' of 'dict' objects}


------------------------- Profile Analysis -------------------------
It tooks ~40 s to train the bpe algorithm and 0.13 GB at pick usage for the tinystories dataset with vocab size of 10k.
the longest token is the word "accomplishment" which make sense as it is one of the longest word in english and often use in stories.


*************** Start training tokenizer on openwebtext_train.txt dataset ***************
------------------------- Pretokenization -------------------------
Pretokenizer took 1.9625839909810263 min
Memory taken currently 0.5788362743332982 GB with a pick at 2.415621872060001 GB
------------------------- BPE Algorithm -------------------------
Base vocabulary size: 257
final vocab size: 32000
Total merges performed: 31743
BPE took 97.47809947916927 min
Memory taken currently 0.00523912999778986 GB with a pick at 20.344736545346677 GB
longest token in vocab: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
************************* result on openwebtext_train.txt datset *************************
         4458741750 function calls in 5966.994 seconds

   Ordered by: internal time
   List reduced from 249 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
153860798 3106.554    0.000 3126.678    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:135(sift_down)
 36031452 1119.515    0.000 1792.954    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:170(_update_pair_stats_for_word_heap)
317138737  495.373    0.000  540.013    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:116(push)
    31743  436.177    0.014 3697.856    0.116 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:147(_select_best_pair_heap)
 36031452  111.066    0.000  142.763    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
       19  102.865    5.414  102.865    5.414 {method 'acquire' of '_thread.lock' objects}
153772412  101.976    0.000 3232.380    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:127(pop_top)
1194881312   86.378    0.000   86.378    0.000 {built-in method builtins.len}
    31743   74.139    0.002 2009.881    0.063 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:207(_apply_merge_to_sequences_heap)
        1   71.960   71.960 5966.994 5966.994 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py:19(run_train_bpe)


------------------------- Profile Analysis -------------------------
It tooks ~100 min to train the bpe algorithm with 2 min for pretokenization and 20 GB at pick usage for the openwebtext dataset with vocab size of 32k.
the longest token is the token "b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
" which represent "ÃÂÃÂ" which doesn't make sense as it is not a word used in english but this show that tinystories dataset is very clean data and we have a lot of unpurity in openwebtext.

"""