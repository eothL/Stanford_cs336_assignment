# this file is the first version of the pretokenizer
# the main diff with the other one is here we use a list and we don't compile the regex pattern

import regex as re
from collections import Counter

def pretokenize(
        pattern:re.Pattern,
        text: str, 
        ) -> dict[bytes, int]:
    """ based on a pattern and ignoring the special tokens in a text, return a mapping of bytes to integer"""

    pretokenize_list= []    
    for match in re.finditer(pattern, text):
        pretokenize_list.append(match.group(0).encode("utf-8"))

    return Counter(pretokenize_list)

def merging_pattern(
        base_pattern: str,
        special_tokens: str
)->re.Pattern:
        # let's define the final pattern by merging the base pattern with special tokens to group bytes by using regex pattern matching engine
    # if we have a special tokens in our data structure, we want to keep their structure inchanged while pretokenizing
    if special_tokens: 
        # '|' represent the OR in regex and we use re.escape() to match this token literaly (it returns a version of the string s with all regular-expreerssion metacharacters escaped)
        # Sorted the list by length to match the longer ones first 
        special_pattern = '|'.join([re.escape(t) for t in sorted(special_tokens,key = len, reverse= True)])
        pattern = rf"({special_pattern})|{base_pattern}"
    else :
        pattern = base_pattern
    return pattern


def main():
    import os
    from collections import Counter
    
    print("beginning pretokenisation")
    HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(HERE, "..", "data","tinystories_val.txt")

    # base pattern used in GPT-2
    base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens = ["<|endoftext|>"]

    # read a subset so it runs fast (change max_bytes or set to 0 for all)
    max_bytes = 0

    with open(data_path, "rb") as f:
        raw = f.read() if max_bytes == 0 else f.read(max_bytes)
    text = raw.decode("utf-8", errors="ignore")
    pattern = merging_pattern(base_pattern,special_tokens)
    counts = pretokenize(pattern, text)

    total = sum(counts.values())
    print("data_path:", os.path.abspath(data_path))
    print("total_pretokens:", total)
    print("unique_pretokens:", len(counts))
    print("top10:", counts.most_common(10))

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
    without compiled regex
    16 344 379 function calls in 3.0s in average for the tiny stories validation set
    13111 unique tokens
    5446705 pretoken total

    with compiled regex
    10 910 760 function calls in 2.58s in average for the tiny stories validation set
    13111 unique token
    5446705 pretokens
    """
