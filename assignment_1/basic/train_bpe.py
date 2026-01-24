import cProfile
import pstats


# vocab_size = 50304 instead of 50357 to increase occupancy in gpu

def _get_compression_rate(vocab: dict[bytes,int], text: str)-> float:

    number_bytes = len(bytes(text, encoding="utf-8"))
    number_tokens = len(vocab.items) 
    return number_bytes / number_tokens

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str])-> tuple[dict]:
    return