from .pretokenization import count_pretokens_parallel
from collections import Counter
import os 

def _update_sequence_list(seq: list[bytes],sym_a: bytes, sym_b:bytes):
    """reconstruct an updated sequence"""
    out = []
    i = 0
    while i < len(seq):
        # update only the merged sequences and reconstruct the rest of the list
        if i+1 < len(seq) and seq[i] == sym_a and seq[i+1] == sym_b:
            out.append(seq[i] + seq[i+1])
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out 

def _most_common_pairs(
        pair_counts:Counter[tuple[bytes,bytes]],
        Ranking_criteria= lambda kv:(kv[1],kv[0])
)-> tuple[tuple[bytes,bytes],int]:
    """Return the most common pair of bytes based on the ranking criterion : (counts, bytes value if tie)"""
        # # get the first most common pairs 
        # (sym_a,sym_b), _ = pair_counts.most_common(1)[0] # return n list of (element, count) pairs sorted by count descending
        # We have to deal with tie count for pairs and having a deterministic rule to choose
        # we can either use min or max for this 
        # (sym_a,sym_b), _ = min(
        #     pair_counts.items(), # all (pairs, counts)
        #     key = lambda kv: (-kv[1],kv[0]) # picking the smallest key value with min()
        #     # find highest count, with (-)-> -higher value = smaller negativer value
        #     # if tie, pick by dictionary order smallest pair
        #     )
        # this solution passed for the test, i think the reference function used max count then lexicograpihcally largest pair
    return max(pair_counts.items(),key= Ranking_criteria)


def _get_pairs_counts(
        sequence_list:list[tuple[list[bytes],int]]
)->Counter[tuple[bytes,bytes],int]:
    """Return a dict of pairs of tuple and theirs frequencies"""
    pair_counts = Counter()

    for seq, seq_count in sequence_list:
        for sym_a, sym_b in zip(seq,seq[1:]):
            pair_counts[(sym_a, sym_b)] += seq_count 

    return pair_counts

# vocab_size = 50304 instead of 50357 to increase occupancy in gpu as 50357 is a multiple of 64
def train_bpe(
        counts: Counter[bytes],
        special_tokens: list[str],
        vocab_size: int = 50307, 
    )-> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    """ train a bpe tokeniser and return a mapping of id to their token"""

    vocab:dict[int,bytes] = {} # mapping of id to token 
    # to easily retrieve each token id, we need the following dict:
    token_to_id: dict[bytes,int] = {}  # mapping of token to id 
    merge_history: list[tuple[bytes,bytes]] = []

    # initialization for the number from 0 to 255
    for id in range(256):
        vocab[id] = bytes([id]) # don't forget the bracket otherwise we bytes of length id instead of converting 
        token_to_id[bytes([id])] = id

    # adding special token to the vocab 
    for tok in special_tokens:
        btok = tok.encode("utf-8")
        token_to_id[btok] = len(vocab)
        vocab[len(vocab)] = btok # stocking in binary

    base_size = 256 + len(special_tokens)
    assert vocab_size >= base_size
    print("Base vocabulary size:", base_size)
    num_merges = vocab_size - base_size

    # every pretoken can be considered as sequence of symbol/character for example "word" = ["w","o","r","d"]
    # decomposing into sequence for the merging
    special_tokens_bytes = {t.encode("utf-8") for t in special_tokens}

    sequence_list =[
        (
            [word] if word in special_tokens_bytes # keep intact special token
            else [bytes([sym]) for sym in word],   # decompose word in a sequence of symbol
            count                                  # frequence of the whole sequence in the corpus
        )
        for word, count in counts.items()       
    ]
    # example (b' that', 23033864) -> ([b" ", b"t", b"h", b"a", b"t"], 23033864)

    for _ in range (num_merges):
        pair_counts = _get_pairs_counts(sequence_list)

        if not pair_counts: 
            break 
        
        (sym_a, sym_b), _ = _most_common_pairs(pair_counts, Ranking_criteria=lambda kv: (kv[1],kv[0]))

        new_bytes = sym_a + sym_b
        new_id = len(vocab)
        
        if new_bytes in token_to_id:
            continue

        vocab[(new_id)]= new_bytes
        token_to_id[new_bytes] = new_id
        merge_history.append((sym_a,sym_b))

        sequence_list = [(_update_sequence_list(seq, sym_a, sym_b),count) for seq, count in sequence_list]
        
    print("final vocab size:", len(vocab))
    print("Total merges performed:", len(merge_history))
    # print("100 first tokens after 256", [vocab[i] for i in range(256,366)])
    return (vocab, merge_history)


def main():  
    dataset = "tinystories_val.txt"
    num_processes = os.cpu_count() - 1 
    print("num_process:", num_processes)
    HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(HERE, "..", "data", dataset)
    base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens= ["<|endoftext|>"]    

    counts = count_pretokens_parallel(
        data_path=data_path,
        num_processes=num_processes,
        base_pattern=base_pattern,
        special_tokens=special_tokens,
        split_special_token=special_tokens[0]
        )
    
    train_bpe(counts,special_tokens)


if __name__ == "__main__":
    import cProfile
    import pstats 
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    result = pstats.Stats(pr)
    result.sort_stats(pstats.SortKey.TIME)
    result.print_stats(20)

    """
    num_process: 13
    Base vocabulary size: 257
    final vocab size: 18000
    Total merges performed: 17743
    1 944 283 416 function calls in 279.721 seconds
    as predicted, updating sequence length is taking the most time
    232628473  116.355    0.000  172.455    0.000 basic/train_bpe_baseline.py:12(_update_sequence_list)
    
    -> should use in-place method 
    instead of using .most_common(), max() function is used to find the best pairs
    result:
    num_process: 13
    Base vocabulary size: 257
    final vocab size: 18016
    Total merges performed: 17759
    2077116716 function calls (2077116307 primitive calls) in 287.666 seconds
    17802/17801    9.666    0.001   14.745    0.001 {built-in method builtins.max}
    """
    