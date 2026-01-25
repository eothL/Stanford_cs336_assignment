from __future__ import annotations
import os
from typing import BinaryIO, Sequence
import pretokenizer_baseline, pretokenizer 
from collections import Counter
import multiprocessing as mp
from regex import Pattern



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _count_chunck(
        data_path:str,
        start: int,
        end: int,
        PAT: Pattern,
)-> Counter:
    """worker function for multiprocessing return a pretokenize of a chunck"""
    with open(data_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8",errors="ignore")
        counts = pretokenizer.pretokenize_string_count(PAT=PAT,text=chunk)
    return counts

def count_pretokens_parallel(
        data_path: str,
        num_processes: int,
        base_pattern: str,
        special_tokens: list[str],
        split_special_token: str,
        *,
        boundaries: Sequence[int] | None = None, # if it is compute outside of the loop
        debug: bool= False
)->Counter[bytes]:
    """
    pretokenize in parallel the data for the BPE algorithm by leveraging regex matching pattern to count the number of similar word 
    return a Counter of bytes
    """
    if boundaries is None:
        with open(data_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                num_processes,
                split_special_token.encode("utf-8")
                )

    PAT = pretokenizer.compile_pattern(base_pattern,special_tokens)

    tasks = [(data_path, s, e, PAT) for s, e in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(processes=num_processes) as pool:
        counters = pool.starmap(_count_chunck, tasks, chunksize=1) # keep chunk size to one, increasing it reduce performance


    counts_string = Counter()
    for c in counters:
        counts_string.update(c)
    
    counts = pretokenizer.encoding_pretokenizer_counts(counts_string)

    if debug:
        total = sum(counts.values())
        print("data_path:", os.path.abspath(data_path))
        print("total_pretokens:", total)
        print("unique_pretokens:", len(counts))
        print("top10:", counts.most_common(10))
    return counts
 
def main_parallel():
    HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(HERE, "..", "data","openwebtext_train.txt")
    
    num_processes = os.cpu_count() - 1 
    print("num_process:", num_processes)

    # base pattern used in GPT 2
    base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens= ["<|endoftext|>"]

    count_pretokens_parallel(data_path,num_processes,base_pattern, special_tokens,"<|endoftext|>",debug=True )

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(HERE, "..", "data","tinystories_train.txt")
    

    with open(data_path, "rb") as f:
        num_processes = os.cpu_count() - 1 
        print("num_process:", num_processes)
 
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # base pattern used in GPT 2
        base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_tokens= ["<|endoftext|>"]


        pretokenize_total = Counter()

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pattern = pretokenizer_baseline.merging_pattern(base_pattern,special_tokens)
            pretokenised_chunck = pretokenizer_baseline.pretokenize(pattern,text=chunk)
            pretokenize_total.update(pretokenised_chunck)

        total = sum(pretokenize_total.values())
        print("data_path:", os.path.abspath(data_path))
        print("total_pretokens:", total)
        print("unique_pretokens:", len(pretokenize_total))
        print("top10:", pretokenize_total.most_common(10))

## Usage
if __name__ == "__main__":
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    main_parallel()
    # main()  
    pr.disable()
    result = pstats.Stats(pr)
    result.sort_stats(pstats.SortKey.TIME)
    result.print_stats(20)
    """
    on openwebtext_train.txt
    it takes 109 s with 15 588 502 called function
    total_pretokens: 2475971866
    unique_pretokens: 6605727
    
    and 92.04 s with 22 157 501 function calls we have more called function but improved in time
    6605742    0.402    0.000    0.402    0.000 {method 'encode' of 'str' objects}

    training set of tinystories 
    17s using chunking + parallelism + compiled pattern
    17s using chuncking + parallelism + baseline pretokenizer
    same time, don't save that much time with parallelism

    with encoding at the end we have 12.681 : crazy how simple solution makes huges differences
    407370 function calls
    total_pretokens: 539 317 083
    unique_pretokens: 59 921
    and on the validation set : 0.247 s huge improvement lol, the bigger the corpus are the bigger impact
    116764 function calls 

    running under 13 process + TinyStories validation set
    with chunking + baseline pretokeniser
    16 135 569 function calls in 3.12s in average for the tiny stories validation set
    13111 unique tokens
    5446705 pretoken total
    gain in speed 


    with chunking + compiled regex pretokeniser
    16 135 569 function calls in 2.7s in average for the tiny stories validation set
    13111 unique tokens
    5446705 pretoken total
    no significant gain in speed

    with chunking + // + compiled regex pretokeniser
    100781 called functions in 0.281s
    13111 unique tokens
    5446705 pretoken total

    with the baseline,
    we have in average 0.261s and fewer hundred less called function
    it may be due to the fact that compiled regex object are pickleable
    """
        



