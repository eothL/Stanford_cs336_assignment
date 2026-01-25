# In this file, I add some upgrade to the function and try some techniques
# re.compiled pattern, caching and simple idea like encoding at the end instead of each time xD


import regex as re
from collections import Counter

def pretokenize(
        PAT: re.Pattern,
        text: str, 
        )-> Counter[bytes]:
    """ based on a pattern and ignoring special tokens in a text, return a mapping of dict of bytes sequences to its frequency in the text"""        
        
    counts : Counter[bytes] = Counter()

    # go through the corpus, and find all the world with similar pattern and add it to the dict 
    for match in PAT.finditer(text):
        counts[match.group(0).encode("utf-8")] += 1

    return counts

# --------------- Others solutions --------------
# Encoding at the end instead at each step
def pretokenize_string_count(
        PAT:re.Pattern,
        text:str,
)-> Counter[str]:
    """encode string after counting at the end of the process"""
    str_counts = Counter(m.group(0) for m in PAT.finditer(text))
    return str_counts

def encoding_pretokenizer_counts(
        str_counts: Counter[str],
):
    """Return encoded in utf-8 of a pretokenizer with string types as key value"""
    return Counter({s.encode("utf-8"): c for s,c  in str_counts.items()})


# cached version but not clear value against encoding at the end
def pretokenize_cached(
        PAT:re.Pattern,
        text:str,
)-> Counter[bytes]:
    """Space complexity of this solution:
    let u be the number of unique tokens"""
    counts = Counter()
    cache :dict[str,bytes] = {}
    for m in PAT.finditer(text):
        s = m.group(0)
        b = cache.get(s)
        if b is None:
            b = cache[s] = s.encode("utf-8")
        counts[b] += 1
    return counts


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

def compile_pattern(base_pattern:str, special_tokens:list[str])-> re.Pattern:
    """Compile the pattern as pattern object, so it is reusable"""
    return re.compile(merging_pattern(base_pattern,special_tokens))


def main():
    import os
    from collections import Counter
    
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

    counts = encoding_pretokenizer_counts(pretokenize_string_count(text=text, PAT= compile_pattern(base_pattern= base_pattern, special_tokens=special_tokens )))
    # counts = pretokenize_cached(PAT= compile_pattern(base_pattern,special_tokens), text=text)
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
    with compiled regex
    10 910 760 function calls in 2.58s in average for the tiny stories validation set
    13111 unique token
    5446705 pretokens

    Using encoding at the end of counting:
    10 910 800 function calls in 2.05 s in average for tiny stories 
    
    Using caching we save few s for this example but for larger corpus we can see more result
    2.54
    10 923 871
    """
      
    
    
    