# this file is to answer question from the assignment 1 pdf file
from __future__ import annotations

from .train_bpe import train_bpe_heap
from .pretokenization import count_pretokens_parallel, find_chunk_boundaries
from .bytes_utils import bytes_to_unicode_text
from .Tokenizer import Tokenizer
from .model import SGD 
import torch
import json 
import os 
from pathlib import Path
import cProfile 
import pstats
import time 
import tracemalloc
import random 
import numpy as np
import multiprocessing as mp


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


def save_train_bpe(data_folder_path, dataset, special_tokens,vocab_path_ts, merge_path_ts, vocab_path_owt, merge_path_owt):
    print("="*30,"Start training tokenizer", "="*30)   
    vocab_ts, merge_ts   =  train_bpe(data_folder_path,vocab_size=10000,special_tokens=special_tokens,dataset=dataset[0])
    vocab_owt, merge_owt =  train_bpe(data_folder_path,vocab_size=32000,special_tokens=special_tokens,dataset=dataset[1])
    
    save_vocab_merges(vocab_ts,merge_ts,vocab_path_ts,merge_path_ts)
    save_vocab_merges(vocab_owt,merge_owt,vocab_path_owt,merge_path_owt)


def get_sample(data_path, split_tokens_bytes, num_process = os.cpu_count() - 1, k = 10, seed = 93):
    """return 10 segment of the text separated by a special token"""
    with open(data_path, "rb") as f: 
        boundaries = find_chunk_boundaries(f, num_process, split_tokens_bytes)
        # print(type(boundaries), boundaries)

    rng = random.Random(seed)
    k = min(k, len(boundaries) - 1)
    sample_ids = rng.sample(range(len(boundaries) - 1), k)
    bounds = [(s,e) for s,e in zip(boundaries[:-1], boundaries[1:])]
    print(sample_ids)
    delim = split_tokens_bytes.decode("utf-8")
    sample: list[str] = []
    with open(data_path, "rb") as f:
        for i in sample_ids:
            f.seek(bounds[i][0])
            chunk = f.read(bounds[i][1] - bounds[i][0]).decode("utf-8", errors="replace")
            docs = [d for d in chunk.split(delim) if d]
            sample.extend(docs)
    sample = rng.sample(sample, k)
    return sample


def tokenizer_experiments(data_path, vocab_path_ts, merge_path_ts,  vocab_path_owt, merge_path_owt,split_tokens_bytes,dataset,special_tokens):
    data_path_ts = os.path.join(data_path,dataset[0])
    data_path_owt = os.path.join(data_path,dataset[1])
    tok_ts = Tokenizer.from_files(vocab_path_ts, merge_path_ts, special_tokens)
    tok_owt = Tokenizer.from_files(vocab_path_owt, merge_path_owt, special_tokens)
    sample_ts = get_sample(data_path_ts, split_tokens_bytes)
    sample_owt = get_sample(data_path_owt, split_tokens_bytes,seed=107)
    chunk = "A<|endoftext|>B"
    ids = tok_ts.encode(chunk)
    print([tok_ts.decode([i]) for i in ids])

    print(tok_ts.encode(sample_ts[0]))
    print(tok_ts.decode(tok_ts.encode(sample_ts[0])))

    total_bytes_ts = 0
    total_tokens_ts = 0
    total_bytes_owt = 0
    total_tokens_owt = 0
    for spl_ts, spl_owt in zip(sample_ts, sample_owt):
        total_bytes_ts += len(spl_ts.encode("utf-8"))
        total_tokens_ts += len(tok_ts.encode(spl_ts))
        total_bytes_owt += len(spl_owt.encode("utf-8"))
        total_tokens_owt += len(tok_owt.encode(spl_owt))

    print(f"compression rate of the tokenizer train on {dataset[0]}", total_bytes_ts/total_tokens_ts)
    print(f"compression rate of the tokenizer train on {dataset[1]}", total_bytes_owt/total_tokens_owt)

    total_bytes_tsowt = 0
    total_tokens_tsowt = 0
    for spl_owt in sample_owt:
        total_bytes_tsowt += len(spl_owt.encode("utf-8"))
        total_tokens_tsowt += len(tok_ts.encode(spl_owt))
    print(f"compression rate of the tokenizer train on {dataset[0]} on the dataset {dataset[1]}", total_bytes_tsowt/total_tokens_tsowt)
    
    total_bytes_owtts = 0
    total_tokens_owtts = 0
    for spl_ts in sample_ts:
        total_bytes_owtts += len(spl_ts.encode("utf-8"))
        total_tokens_owtts += len(tok_owt.encode(spl_ts))
    print(f"compression rate of the tokenizer train on {dataset[1]} on the dataset {dataset[0]}", total_bytes_owtts/total_tokens_owtts)

    big_text = "".join(sample_ts[::])
    big_text_owt = "".join(sample_owt[::])

    start_ts = time.perf_counter()
    tokens_ts = tok_ts.encode(big_text)
    end_ts = time.perf_counter()
    time_ts = end_ts - start_ts

    start_owt = time.perf_counter()
    tokens_owt = tok_owt.encode(big_text_owt)
    end_owt = time.perf_counter()
    time_owt = end_owt - start_owt

    total_bytes = len(big_text.encode("utf-8"))
    total_bytes_owt_2 = len(big_text_owt.encode("utf-8"))
    print("size of the file encoded: ", total_bytes)
    print("compression rate for tokens_ts:", total_bytes/len(tokens_ts))
    print("compression rate for tokens_owt:", total_bytes_owt/ len(tokens_owt))
    print(" time to encode for tokens_ts", time_ts)
    print(" time to encode for tokens_owt", time_owt)
    throughput_ts = total_bytes/time_ts
    throughput_owt = total_bytes_owt_2/time_owt
    print("throughput bytes/second for token_ts", throughput_ts)
    print("throughput bytes/second for token_owt", throughput_owt)
    print(f"for 825GB, it will take : {825*1024**3/throughput_ts} s or {(825*1024**3/throughput_ts)/(3600*24)} days")
    print(f"for 825GB, it will take with owt tokenizer: {825*1024**3/throughput_owt} s or {(825*1024**3/throughput_owt)/(3600*24)} days")


def serialization_data_simple(data_path, vocab_path, merge_path,dataset, split_tokens_bytes ,special_tokens):
    data_path = os.path.join(data_path,dataset)
    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)
    ids = []
    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, max(1,os.cpu_count()-1), split_tokens_bytes)
    
    bounds = [(s,e) for s,e in zip(boundaries[:-1],boundaries[1:])]
    with open(data_path, "rb") as f:
        for s,e in bounds:
            f.seek(s)
            chunk = f.read(e - s).decode("utf-8", errors = "replace")
            id_chunk = tokenizer.encode(chunk)
            ids.extend(id_chunk)
    return np.array(ids, dtype=np.uint16)


def serialization_data(data_path, vocab_path, merge_path,dataset, split_tokens_bytes ,special_tokens,artifact_folder_path):
    data_path = os.path.join(data_path,dataset)
    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)

    assert len(tokenizer.vocab) <= 65536
    out_path_name = f"{Path(dataset).stem}.uint16.bin"
    out_path = os.path.join(artifact_folder_path,out_path_name)
    ids_chunk = []
    total_tokens = 0
    # counts number of tokens
    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, max(1,os.cpu_count()-1), split_tokens_bytes)
    
    bounds = [(s,e) for s,e in zip(boundaries[:-1],boundaries[1:])]
    with open(data_path, "rb") as f:
        for s,e in bounds:
            f.seek(s)
            chunk = f.read(e - s).decode("utf-8", errors = "replace")
            ids_chunk = tokenizer.encode(chunk)
            total_tokens += len(ids_chunk)

    # serialized 
    os.makedirs(artifact_folder_path, exist_ok=True)
    arr = np.memmap(out_path, dtype= np.uint16, shape=(total_tokens,), mode="w+")
    
    index = 0 
    with open(data_path, "rb") as f:
        for s,e in bounds:
            f.seek(s)
            chunk = f.read(e - s).decode("utf-8", errors = "replace")
            ids_chunk = tokenizer.encode(chunk)
            n=len(ids_chunk)
            arr[index: index+n]= ids_chunk
            index += n
    arr.flush()

    return arr 

def learning_rate_tuning(lrs:list):
    print("start learning rate tuning")
    for lr in lrs:
        print(f"lr {lr} test")
        weights = torch.nn.Parameter(5 * torch.randn((10,10)))
        opt = SGD([weights], lr =lr)

        for t in range(10):
            opt.zero_grad() # reset the gradeints for all learnanble paramters
            loss = (weights**2).mean() # Compute a scalar loss value
            print(loss.cpu().item())
            loss.backward() # run backward pass which computes gradients
            opt.step() # Run optimizer step

if __name__== "__main__":
    special_tokens= ["<|endoftext|>"]   
    split_tokens = ["<|endoftext|>"]   
      
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
    dataset = ["tinystories_train.txt","openwebtext_train.txt","tinystories_val.txt","openwebtext_val.txt"]
    # save_train_bpe(data_folder_path, dataset, special_tokens,vocab_path_ts, merge_path_ts, vocab_path_owt, merge_path_owt)

    split_tokens_bytes = split_tokens[0].encode("utf-8") 
    #tokenizer_experiments(data_folder_path, vocab_path_ts, merge_path_ts, vocab_path_owt, merge_path_owt, split_tokens_bytes, dataset, special_tokens)
    
    # print("serialization of dataset")
    # for data in dataset[::-1]:
    #     print(f"serialization of {data}")
    #     pr = cProfile.Profile()
    #     pr.enable()
    #     if data == "tinystories_train.txt" or data == "tinystories_val.txt":
    #         serialization_data(data_path=data_folder_path,
    #                         vocab_path=vocab_path_ts,
    #                         merge_path=merge_path_ts,
    #                         dataset=data, 
    #                         split_tokens_bytes=split_tokens_bytes,
    #                         special_tokens=special_tokens,
    #                         artifact_folder_path=artifact_folder_path)
    #     else : 
    #         serialization_data(data_path=data_folder_path,
    #             vocab_path=vocab_path_owt,
    #             merge_path=merge_path_owt,
    #             dataset=data, 
    #             split_tokens_bytes=split_tokens_bytes,
    #             special_tokens=special_tokens,
    #             artifact_folder_path=artifact_folder_path)
    #     pr.disable()
    #     result = pstats.Stats(pr)
    #     result.sort_stats(pstats.SortKey.TIME)
    #     result.print_stats(20)

    learning_rate_tuning([1e1,1e2,1e3])


"""
learning rate
start learning rate tuning
lr 10.0 test
23.419147491455078
14.98825454711914
11.048701286315918
8.644428253173828
7.001987934112549
5.805449962615967
4.896126747131348
4.183879852294922
3.6131093502044678
3.1474196910858154
lr 100.0 test
19.997255325317383
19.99725341796875
3.4309864044189453
0.08211121708154678
1.436719682729309e-16
1.601313098447264e-18
5.392185233667183e-20
3.2121614325484406e-21
2.7555996891975526e-22
3.061777747986087e-23
lr 1000.0 test
33.50859832763672
12096.6025390625
2089273.75
232409376.0
18825158656.0
1188083269632.0
60992310476800.0
2624149855928320.0
9.672056756187955e+16
3.105805063507935e+18

a higher learning rate can converge faster but also diverge.
"""

"""
========================== Transformer LM resource accounting =============================
a)
GPT-2 XL configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600  = the dimensionality of the model embeddings and sublayer outputs
num_heads : 25
d_ff : 6,400

bias parameter off for every layer 
number of trainable paramater =                                                 if we add bias parameter
    Embedding layer : vocab_size * d_model = 50 257 * 1 600 = 80 411 200        
    embedding_dim : dimension of embedding vector = d_model
    Transformer Block : num_layers = 48
        RMSnorm 1: d_model                                  = 1600              + 1600
        MHA_self_attention: num_heads = 25
            q_proj :   d_model * d_model = 1600 * 1600      = 2 560 000         + 1600
                per head size: (num_head, d_model//num_heads)
            k_proj :   d_model * d_model = 1600 * 1600      = 2 560 000         + 1600
                per head size: (num_head, d_model//num_heads)
            v_proj :   d_model * d_model = 1600 * 1600      = 2 560 000         + 1600
                per head size: (num_head, d_model//num_heads)
            o_proj :   d_model * d_model = 1600 * 1600      = 2 560 000         + 1600
        RMSnorm 2: d_model                                  = 1600
        positionwise - FFN:
            w1: d_model * d_ff = 1600 * 6400                = 10 240 000        + 6400
            w3: d_model * d_ff = 1600 * 6400                = 10 240 000        + 6400
            w2: d_ff * d_model = 6400 * 1600                = 10 240 000        + 1600
        ----------------------------------------------------------------
                                                            = 40 963 200 * 48   + 20 800 * 48
                                                            = 1 966 233 600.    + 998 400
    -------------------------------------------------------------------------
                                                            = 1,9 B + 80 M 
                                                            = 2 046 644 800
                                                            = 2 B parameters    + 1 M bias parameters
if one parameter is represented in single precision floating point meaning FP32 or 32 bits(4 bytes):
it will require more than 8GB of memory in FP32
in FP8, we could go down to 2GB of memory.                                                            


b)
the matrix input is size : (context_len * vocab_size)
token embedding : 


"""

"""
============================== Experiments on tokenizer ==============================
seed used: 93 and 107
a)
compression rate of the tokenizer train on tinystories_train.txt 4.0510440835266825
compression rate of the tokenizer train on openwebtext_train.txt 4.237972988056703

b)
compression rate of the tokenizer train on tinystories_train.txt on the dataset openwebtext_train.txt 3.144608249130363
the tokenizer train on tinystories has a worse compression rate due to having less suitable vocabulary for the content of openwebtext.
we have a difference of 100% in the compression rate 

c)
it will take 3.3 days to encode the Pile dataset (825GB of text). with throughput of 311 000 bytes/sec

[7, 9, 5, 1, 11, 10, 8, 6, 3, 0]
[3, 9, 11, 6, 7, 4, 1, 12, 8, 0]
[10, 430, 439, 259, 398, 44, 401, 283, 259, 390, 1760, 825, 46, 285, 825, 502, 266, 3434, 432, 327, 46, 527, 327, 44, 263, 825, 1023, 259, 346, 3533, 46, 285, 3533, 283, 797, 267, 1486, 46, 316, 1931, 259, 346, 1021, 46, 10, 34, 1183, 44, 390, 825, 637, 324, 263, 3533, 46, 317, 1308, 375, 349, 3434, 432, 327, 476, 10, 410, 825, 1468, 44, 317, 73, 3434, 711, 309, 384, 443, 33, 338, 516, 266, 538, 543, 624, 397, 10, 410, 3533, 562, 46, 317, 1100, 384, 259, 558, 6818, 266, 3434, 44, 390, 825, 46, 5862, 1115, 443, 413, 10, 1382, 44, 263, 390, 1760, 825, 3148, 573, 44, 376, 266, 499, 1023, 263, 797, 3533, 46, 285, 825, 3148, 267, 3148, 44, 5118, 661, 543, 624, 586, 327, 46, 717, 362, 432, 631, 988, 928, 889, 46, 10]

Once upon a time, there was a little orange bug. The bug loved to crawl all day. One day, the bug met a big judge. The judge was kind and wise. He wore a big hat.
"Hello, little bug," said the judge. "Why do you crawl all day?"
The bug replied, "I crawl because it's fun! I like to see new things."
The judge smiled. "That's a good reason to crawl, little bug. Keep having fun!"
So, the little orange bug crawled away, happy to have met the kind judge. The bug crawled and crawled, seeing many new things every day. And they all lived happily ever after.

compression rate of the tokenizer train on tinystories_train.txt 4.0510440835266825
compression rate of the tokenizer train on openwebtext_train.txt 4.237972988056703
compression rate of the tokenizer train on tinystories_train.txt on the dataset openwebtext_train.txt 3.144608249130363
compression rate of the tokenizer train on openwebtext_train.txt on the dataset tinystories_train.txt 3.9817559863169896
size of the file encoded:  6984
compression rate for tokens_ts: 4.058105752469494
compression rate for tokens_owt: 4.2384460817146685
 time to encode for tokens_ts 0.002245499985292554
 time to encode for tokens_owt 0.017486375058069825
throughput bytes/second for token_ts 3110220.461252906
throughput bytes/second for token_owt 2171290.4975395724
for 825GB, it will take : 284814.8598582474 s or 3.296468285396382 days

result serialization of dataset
serialization of dataset
serialization of tinystories_train.txt
         3295693463 function calls (3295693360 primitive calls) in 555.825 seconds

   Ordered by: internal time
   List reduced from 300 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       26  416.711   16.027  546.841   21.032 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:40(encode)
1073184336   46.590    0.000   46.590    0.000 {method 'group' of '_regex.Match' objects}
1078619734   40.312    0.000   40.312    0.000 {method 'encode' of 'str' objects}
1078619830   30.697    0.000   30.697    0.000 {method 'extend' of 'list' objects}
        1    7.560    7.560  555.825  555.825 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py:236(serialization_data)
  3271646    3.681    0.000    4.994    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
       26    3.170    0.122    3.170    0.122 {method 'split' of '_regex.Pattern' objects}
  3939068    2.165    0.000    2.165    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:62(<listcomp>)
  5435400    1.157    0.000    1.157    0.000 {method 'finditer' of '_regex.Pattern' objects}
       26    0.924    0.036    0.924    0.036 {method 'decode' of 'bytes' objects}
 32963010    0.799    0.000    0.799    0.000 {built-in method builtins.len}
 14855751    0.515    0.000    0.515    0.000 {method 'append' of 'list' objects}
       50    0.440    0.009    0.440    0.009 {method 'read' of '_io.BufferedReader' objects}
   667422    0.395    0.000    0.395    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:69(<listcomp>)
  3271648    0.326    0.000    0.326    0.000 {built-in method builtins.min}
   667422    0.323    0.000    0.323    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:59(<listcomp>)
        1    0.027    0.027    0.027    0.027 {method 'flush' of 'mmap.mmap' objects}
    29486    0.010    0.000    0.016    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:69(unicode_token_to_bytes)
   145043    0.006    0.000    0.006    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:72(<genexpr>)
        1    0.005    0.005    0.005    0.005 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/.venv/lib/python3.11/site-packages/numpy/_core/memmap.py:216(__new__)


serialization of openwebtext_train.txt
         18994367226 function calls in 3535.543 seconds

   Ordered by: internal time
   List reduced from 106 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       26 2360.347   90.783 3487.933  134.151 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:40(encode)
175120286  243.886    0.000  332.331    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
4943506184  235.782    0.000  235.782    0.000 {method 'group' of '_regex.Match' objects}
4948304978  197.486    0.000  197.486    0.000 {method 'encode' of 'str' objects}
4948304978  174.199    0.000  174.199    0.000 {method 'extend' of 'list' objects}
208102772  118.951    0.000  118.951    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:62(<listcomp>)
2408008027   54.893    0.000   54.893    0.000 {built-in method builtins.len}
        1   36.502   36.502 3535.543 3535.543 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py:236(serialization_data)
1116475614   33.553    0.000   33.553    0.000 {method 'append' of 'list' objects}
175120286   19.352    0.000   19.352    0.000 {built-in method builtins.min}
 32982486   18.747    0.000   18.747    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:69(<listcomp>)
 32982486   16.412    0.000   16.412    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:59(<listcomp>)
       26   13.177    0.507   13.177    0.507 {method 'split' of '_regex.Pattern' objects}
       26    7.433    0.286    7.433    0.286 {method 'decode' of 'bytes' objects}
       61    3.493    0.057    3.493    0.057 {method 'read' of '_io.BufferedReader' objects}
  4798794    1.150    0.000    1.150    0.000 {method 'finditer' of '_regex.Pattern' objects}
        1    0.088    0.088    0.088    0.088 {method 'flush' of 'mmap.mmap' objects}
    95486    0.035    0.000    0.054    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:69(unicode_token_to_bytes)
   500791    0.020    0.000    0.020    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:72(<genexpr>)
        1    0.016    0.016    0.087    0.087 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:24(from_files)


serialization of tinystories_val.txt
         45673142 function calls in 7.787 seconds

   Ordered by: internal time
   List reduced from 106 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       26    4.922    0.189    7.647    0.294 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:40(encode)
   803134    0.725    0.000    0.981    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
 10838002    0.484    0.000    0.484    0.000 {method 'group' of '_regex.Match' objects}
 10893262    0.438    0.000    0.438    0.000 {method 'encode' of 'str' objects}
 10893262    0.315    0.000    0.315    0.000 {method 'extend' of 'list' objects}
   949912    0.310    0.000    0.310    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:62(<listcomp>)
  6893951    0.159    0.000    0.159    0.000 {built-in method builtins.len}
  3055152    0.096    0.000    0.096    0.000 {method 'append' of 'list' objects}
        1    0.090    0.090    7.787    7.787 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py:236(serialization_data)
   803134    0.069    0.000    0.069    0.000 {built-in method builtins.min}
   146778    0.057    0.000    0.057    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:59(<listcomp>)
       26    0.032    0.001    0.032    0.001 {method 'split' of '_regex.Pattern' objects}
   146778    0.030    0.000    0.030    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:69(<listcomp>)
    29486    0.010    0.000    0.016    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:69(unicode_token_to_bytes)
    55262    0.009    0.000    0.009    0.000 {method 'finditer' of '_regex.Pattern' objects}
        1    0.007    0.007    0.007    0.007 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/.venv/lib/python3.11/site-packages/numpy/_core/memmap.py:216(__new__)
       50    0.006    0.000    0.006    0.000 {method 'read' of '_io.BufferedReader' objects}
   145043    0.006    0.000    0.006    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:72(<genexpr>)
       26    0.005    0.000    0.005    0.000 {method 'decode' of 'bytes' objects}
        1    0.004    0.004    0.026    0.026 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:24(from_files)


serialization of openwebtext_val.txt
         774037266 function calls in 133.104 seconds

   Ordered by: internal time
   List reduced from 106 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       26   68.288    2.626  131.752    5.067 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:40(encode)
 20035732   23.329    0.000   31.770    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:40(_merge_pair_in_seq)
 23670780   10.782    0.000   10.782    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:62(<listcomp>)
120274584    6.100    0.000    6.100    0.000 {method 'group' of '_regex.Match' objects}
234115053    5.304    0.000    5.304    0.000 {built-in method builtins.len}
120392702    5.218    0.000    5.218    0.000 {method 'encode' of 'str' objects}
120392702    4.241    0.000    4.241    0.000 {method 'extend' of 'list' objects}
107071404    3.138    0.000    3.138    0.000 {method 'append' of 'list' objects}
 20035732    2.079    0.000    2.079    0.000 {built-in method builtins.min}
  3635048    1.625    0.000    1.625    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:59(<listcomp>)
  3635048    1.308    0.000    1.308    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:69(<listcomp>)
        1    0.985    0.985  133.104  133.104 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/assignment_question.py:236(serialization_data)
       26    0.312    0.012    0.312    0.012 {method 'split' of '_regex.Pattern' objects}
       26    0.183    0.007    0.183    0.007 {method 'decode' of 'bytes' objects}
       75    0.087    0.001    0.087    0.001 {method 'read' of '_io.BufferedReader' objects}
    95486    0.033    0.000    0.052    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:69(unicode_token_to_bytes)
   118118    0.031    0.000    0.031    0.000 {method 'finditer' of '_regex.Pattern' objects}
   500791    0.019    0.000    0.019    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/bytes_utils.py:72(<genexpr>)
        1    0.015    0.015    0.085    0.085 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/Tokenizer.py:24(from_files)
        1    0.006    0.006    0.006    0.006 {method 'flush' of 'mmap.mmap' objects}
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