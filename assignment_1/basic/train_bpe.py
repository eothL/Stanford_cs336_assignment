from .pretokenization import count_pretokens_parallel
from collections import Counter, defaultdict
import argparse
import os


def get_compression_rate(vocab: dict[int, bytes], text: str) -> float:
    number_bytes = len(bytes(text, encoding="utf-8"))
    number_tokens = len(vocab)
    return number_bytes / number_tokens


def _build_pair_stats(
    sequences: list[tuple[list[bytes], int]],
) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], set[int]]]:
    """Return pair counts and index of words containing each pair."""
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_word_ids: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    # {(b"h", b"i"): {0, 1}, (b"i", b"!"): {1}}

    for word_id, (seq, seq_count) in enumerate(sequences):
        for sym_a, sym_b in zip(seq, seq[1:]):
            pair = (sym_a, sym_b)
            pair_counts[pair] += seq_count
            pair_to_word_ids[pair].add(word_id)


    return pair_counts, pair_to_word_ids


def _select_best_pair(
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[tuple[bytes, bytes], int]:
    """return the most common pair based on the count and bytes value if there is a tie for the count"""
    return max(pair_counts.items(), key=lambda kv: (kv[1], kv[0])) # compare count then bytes value


def _merge_pair_in_seq(
    seq: list[bytes],
    sym_a: bytes,
    sym_b: bytes,
    new_bytes: bytes,
) -> list[bytes]:
    out: list[bytes] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == sym_a and seq[i + 1] == sym_b:
            out.append(new_bytes)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _update_pair_stats_for_word(
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_word_ids: defaultdict[tuple[bytes, bytes], set[int]],
    old_seq: list[bytes],
    new_seq: list[bytes],
    word_count: int,
    word_id: int,
) -> None:
    for sym_a, sym_b in zip(old_seq, old_seq[1:]):
        old_pair = (sym_a, sym_b)
        pair_counts[old_pair] -= word_count
        if pair_counts[old_pair] <= 0:
            pair_counts.pop(old_pair, None)
        word_ids = pair_to_word_ids.get(old_pair)
        if word_ids:
            word_ids.discard(word_id)
            if not word_ids:
                pair_to_word_ids.pop(old_pair, None)

    for sym_a, sym_b in zip(new_seq, new_seq[1:]):
        new_pair = (sym_a, sym_b)
        pair_counts[new_pair] += word_count
        pair_to_word_ids[new_pair].add(word_id)


def _apply_merge_to_sequences(
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_word_ids: defaultdict[tuple[bytes, bytes], set[int]],
    sequences: list[tuple[list[bytes], int]],
    pair: tuple[bytes, bytes],
    new_bytes: bytes,
) -> bool:
    word_ids = list(pair_to_word_ids.get(pair, ()))
    if not word_ids:
        pair_counts.pop(pair, None)
        pair_to_word_ids.pop(pair, None)
        return False

    sym_a, sym_b = pair
    changed = False
    for word_id in word_ids:
        old_seq, word_count = sequences[word_id]
        new_seq = _merge_pair_in_seq(old_seq, sym_a, sym_b, new_bytes)
        if new_seq != old_seq:
            _update_pair_stats_for_word(
                pair_counts=pair_counts,
                pair_to_word_ids=pair_to_word_ids,
                old_seq=old_seq,
                new_seq=new_seq,
                word_count=word_count,
                word_id=word_id,
            )
            sequences[word_id] = (new_seq, word_count)
            changed = True

    return changed

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

    sequences =[
        (
            [word] if word in special_tokens_bytes # keep intact special token
            else [bytes([sym]) for sym in word],   # decompose word in a sequence of symbol
            count                                  # frequence of the whole sequence in the corpus
        )
        for word, count in counts.items()       
    ]
    # example (b' that', 23033864) -> ([b" ", b"t", b"h", b"a", b"t"], 23033864)
    pair_counts, pair_to_word_ids = _build_pair_stats(sequences)
     
    for _ in range (num_merges):
        if not pair_counts: 
            break 
        
        pair, _ = _select_best_pair(pair_counts)
        new_bytes = pair[0] + pair[1]

        if not _apply_merge_to_sequences(
            pair_counts=pair_counts,
            pair_to_word_ids=pair_to_word_ids,
            sequences=sequences,
            pair=pair,
            new_bytes=new_bytes,
        ):
            continue

        if new_bytes not in token_to_id:
            new_id = len(vocab)
            vocab[new_id] = new_bytes
            token_to_id[new_bytes] = new_id
            merge_history.append(pair)

    print("final vocab size:", len(vocab))
    print("Total merges performed:", len(merge_history))
    # print("100 first tokens after 256", [vocab[i] for i in range(256,366)])
    return (vocab, merge_history)


def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tinystories_val.txt")
    args = parser.parse_args()
    dataset =  args.dataset
    print("Dataset used:",dataset)
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

    # Updating in-place sequences(sequence_list) and updating only the modified pairs
    num_process: 13
    Base vocabulary size: 257
    final vocab size: 18016
    Total merges performed: 17759
    133962607 function calls (133962193 primitive calls) in 15.777 seconds


   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
17802/17801   10.029    0.001   14.943    0.001 {built-in method builtins.max}
131608649    4.914    0.000    4.914    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:32(<lambda>)
    39/37    0.257    0.007    0.258    0.007 {built-in method posix.read}
    77464    0.197    0.000    0.281    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:53(_update_pair_stats_for_word)
    77464    0.075    0.000    0.116    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:35(_merge_pair_in_seq)
    17759    0.045    0.000    0.449    0.000 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:78(_apply_merge_to_sequences)
   385870    0.038    0.000    0.038    0.000 {method 'get' of 'dict' objects}
   299950    0.033    0.000    0.033    0.000 {method 'discard' of 'set' objects}
   695103    0.024    0.000    0.024    0.000 {built-in method builtins.len}
        1    0.023    0.023   15.446   15.446 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:111(train_bpe)
   300073    0.023    0.000    0.023    0.000 {method 'add' of 'set' objects}
   318153    0.019    0.000    0.019    0.000 {method 'append' of 'list' objects}
        1    0.016    0.016    0.022    0.022 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:12(_build_pair_stats)
       20    0.012    0.001    0.292    0.015 {method 'poll' of 'select.poll' objects}
       29    0.007    0.000    0.012    0.000 /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/collections/__init__.py:679(update)
       15    0.007    0.000    0.007    0.000 {built-in method _pickle.loads}
       14    0.006    0.000    0.006    0.000 {built-in method _posixsubprocess.fork_exec}
    63342    0.006    0.000    0.006    0.000 {method 'pop' of 'dict' objects}
    17759    0.005    0.000   14.949    0.001 /Users/theo/Curious/Learning_hub/Stanford/assignment_1/basic/train_bpe.py:29(_select_best_pair)
        9    0.004    0.000    0.004    0.000 {built-in method _imp.create_dynamic}

    """
    
    
